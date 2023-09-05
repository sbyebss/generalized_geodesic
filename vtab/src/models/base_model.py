from os import path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch_ema import ExponentialMovingAverage
from torchmetrics.classification.accuracy import Accuracy

from src.logger.jam_wandb import prefix_metrics_keys
from src.models.loss_zoo import get_cost, label_cost
from src.transfer_learning.train_nist_classifier import get_num_label
from src.utils import lht_utils

log = lht_utils.get_logger(__name__)
# pylint: disable=R0901,abstract-method,too-many-instance-attributes,too-many-function-args,line-too-long

ce_loss = nn.CrossEntropyLoss()


def turn_off_grad(network):
    for param in network.parameters():
        if not param.requires_grad:
            break
        param.requires_grad = False


def turn_on_grad(network):
    for param in network.parameters():
        param.requires_grad = True


def get_feat_label(feat_label):
    if isinstance(feat_label, list):  # image
        return feat_label
    if "datasets" in str(type(feat_label)):  # image
        return feat_label.data, feat_label.targets
    if len(feat_label.shape) == 2:  # vectors
        return feat_label[:, :-1], feat_label[:, -1]
    else:
        return None


class BaseModule(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.num_source_class = get_num_label(self.cfg.source)
        self.cfg.num_target_class = get_num_label(self.cfg.target)
        self.f_net = hydra.utils.instantiate(
            self.cfg.f_net, num_class=self.cfg.num_target_class
        )
        self.map_t = hydra.utils.instantiate(
            self.cfg.T_net, num_classes=self.cfg.num_source_class
        )
        self.classifier = hydra.utils.instantiate(
            self.cfg.classifier, num_class=self.cfg.num_target_class
        )
        self.load_classifier()

        self.iter_count = 0
        self.ema_map = (
            ExponentialMovingAverage(self.map_t.parameters(), decay=0.995)
            if self.cfg.ema
            else None
        )
        self.train_acc = Accuracy()
        self.automatic_optimization = False

    def load_classifier(self):
        if path.exists(self.cfg.classifier_save_path):
            self.classifier.load_state_dict(
                torch.load(self.cfg.classifier_save_path)["model_state_dict"]
            )
            log.info(
                f"Successfully load the pretrained classifier from <{self.cfg.classifier_save_path }>"
            )
            self.pretrain_clsf = False
            turn_off_grad(self.classifier)
        else:
            log.info(
                "Didn't find the pretrained classifier, need to train it from scratch..."
            )
            self.pretrain_clsf = True

    @property
    def feat_cost_func(self):
        return get_cost(self.cfg.feat_cost_type)

    def on_fit_start(self) -> None:
        if self.cfg.ema:
            self.ema_map.to(self.device)

    # pylint: disable=arguments-differ,unused-argument
    def training_step(self, batch, batch_idx):
        source_data, target_data = self.get_real_data(batch)
        # pylint: disable=E0633
        opt_t, opt_f, opt_l = self.optimizers()
        if self.current_epoch < self.cfg.classifier_epoch:
            self.pretrain_feature_map(source_data, opt_t)
            if self.pretrain_clsf:
                self.pretrain_classifier(target_data, opt_l)
            else:
                self.test_classifier(target_data)
        else:
            turn_off_grad(self.classifier)
            self.opt_f_g(source_data, target_data, opt_f, opt_t, opt_l)

    def pretrain_feature_map(self, data, map_opt):
        feat, source_label = get_feat_label(data)
        loss = F.mse_loss(self.map_t(feat, source_label), feat)
        if loss > 1e-3:
            map_opt.zero_grad()
            loss.backward()
            map_opt.step()
            self.log_dict(
                prefix_metrics_keys(
                    {"id_loss": loss},
                    "pretrain_loss",
                )
            )

    def pretrain_classifier(self, target_data, optimizer_l):
        loss, loss_info = self.loss_classify(target_data)
        if loss > 1e-3:
            optimizer_l.zero_grad()
            self.manual_backward(loss)
            optimizer_l.step()
            self.log_dict(loss_info)

    def test_classifier(self, target_data):
        _, loss_info = self.loss_classify(target_data)
        self.log_dict(loss_info)

    def opt_f_g(self, source_data, target_data, f_opt, map_opt, clf_opt):
        if self.global_step % (self.cfg.n_outer_iter + self.cfg.n_inner_iter) == 0:
            self.iter_count = 0

        if self.iter_count < self.cfg.n_outer_iter:
            self.opt_f(source_data, target_data, f_opt)
        else:
            self.opt_map(source_data, map_opt, clf_opt)
        self.iter_count += 1

    def opt_f(self, source_data, target_data, f_opt):
        turn_on_grad(self.f_net)
        turn_off_grad(self.map_t)
        loss, loss_info = self.loss_f(source_data, target_data)
        f_opt.zero_grad()
        self.manual_backward(loss)
        f_opt.step()
        self.log_dict(loss_info)

    def opt_map(self, source_data, map_opt, clf_opt):
        del clf_opt
        turn_on_grad(self.map_t)
        turn_off_grad(self.f_net)
        loss, loss_info = self.loss_map(source_data)
        map_opt.zero_grad()
        self.manual_backward(loss)
        map_opt.step()
        if self.cfg.ema:
            self.ema_map.update()
        self.log_dict(loss_info)

    def loss_classify(self, feat_label):
        target_feat, target_label = get_feat_label(feat_label)
        target_label = target_label.long()
        batch_size = target_feat.shape[0]
        random_label = torch.randint(0, self.cfg.num_target_class, (batch_size,)).to(
            self.device
        )
        # this prob output is unnormalized.
        label_logits = self.classifier(target_feat, random_label)
        loss = ce_loss(label_logits, target_label)
        pred = torch.argmax(label_logits, dim=1)
        log_info = prefix_metrics_keys(
            {"ce_loss": loss, "accuracy": self.train_acc(pred, target_label)},
            "pretrain_loss",
        )
        return loss, log_info

    def loss_f(self, source_feat_label, target_feat_label):
        source_feat, source_label = get_feat_label(source_feat_label)
        with torch.no_grad():
            output_feat = self.map_t(source_feat, source_label)
            self.classifier.eval()
            label_logits = self.classifier(output_feat, source_label)
            label_probs = F.softmax(label_logits, dim=1)
        target_feat, target_label = get_feat_label(target_feat_label)
        target_label = target_label.view(-1).long()
        target_label_onehot = F.one_hot(
            target_label, num_classes=self.cfg.num_target_class
        ).float()
        f_tx, f_y = (
            self.f_net(output_feat, label_probs).mean(),
            self.f_net(target_feat, target_label_onehot).mean(),
        )
        f_loss = f_tx - f_y
        log_info = prefix_metrics_keys(
            {"f_tx": f_tx, "f_y": f_y, "f_tx - f_y": f_tx - f_y},
            "f_loss",
        )
        return f_loss, log_info

    def loss_map(self, source_feat_label, return_pf=False):
        # FIXME: if we setup the model, need to change to map_g.
        source_feat, source_label = get_feat_label(source_feat_label)
        output_feat = self.map_t(source_feat, source_label)
        self.classifier.eval()
        label_logits = self.classifier(output_feat, source_label)
        label_probs = F.softmax(label_logits, dim=1)

        feat_loss = self.cfg.coeff_feat * self.feat_cost_func(source_feat, output_feat)
        label_loss = self.cfg.coeff_label * label_cost(
            self.w_distance_table, source_label, label_probs
        )
        cost_loss = feat_loss + label_loss

        f_tx = self.f_net(output_feat, label_probs).mean()
        map_loss = cost_loss - f_tx
        log_info = prefix_metrics_keys(
            {
                "cost_loss": cost_loss,
                "feat_loss": feat_loss,
                "label_loss": label_loss,
                "f_tx": f_tx,
            },
            "map_loss",
        )
        if return_pf:
            return map_loss, log_info, output_feat, label_probs
        return map_loss, log_info

    def configure_optimizers(self):
        optimizer_map = optim.Adam(
            self.map_t.parameters(),
            lr=self.cfg.lr_T,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )
        optimizer_f = optim.Adam(
            self.f_net.parameters(),
            lr=self.cfg.lr_f,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )
        optimizer_l = optim.Adam(self.classifier.parameters(), lr=self.cfg.lr_l)
        if self.cfg.schedule_learning_rate:
            return [optimizer_map, optimizer_f, optimizer_l], [
                StepLR(
                    optimizer_map,
                    step_size=self.cfg.lr_schedule_epoch,
                    gamma=self.cfg.lr_schedule_scale_t,
                ),
                StepLR(
                    optimizer_f,
                    step_size=self.cfg.lr_schedule_epoch,
                    gamma=self.cfg.lr_schedule_scale_h,
                ),
                StepLR(
                    optimizer_l,
                    step_size=self.cfg.lr_schedule_epoch,
                    gamma=self.cfg.lr_schedule_scale_t,
                ),
            ]
        else:
            return optimizer_map, optimizer_f, optimizer_l

    def test_step(self, *args, **kwargs):
        pass
