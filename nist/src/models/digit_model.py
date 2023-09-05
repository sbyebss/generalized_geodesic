from functools import partial

import torch
from scipy.stats import rankdata
from torchvision.models import resnet18

from src.models.base_model import BaseModule, get_feat_label
from src.models.loss_zoo import get_cost
from src.otdd.pytorch.datasets import CustomTensorDataset
from src.otdd.pytorch.distance import DatasetDistance
from src.viz.img import save_tensor_imgs

# pylint: disable=R0901,abstract-method,too-many-locals


class DigitModule(BaseModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.embedder = resnet18(pretrained=True)
        self.embedder.fc = torch.nn.Identity()
        for param in self.embedder.parameters():
            param.requires_grad = False

    @property
    def feat_cost_func(self):
        return partial(get_cost(self.cfg.feat_cost_type), embedder=self.embedder)

    def draw_batch(self, x_data, y_data):
        save_tensor_imgs(
            self.trainer.datamodule.inverse_data_transform(x_data),
            8,
            self.global_step,
            "batch_source",
        )
        save_tensor_imgs(
            self.trainer.datamodule.inverse_data_transform(y_data),
            8,
            self.global_step,
            "batch_target",
        )

    def get_real_data(self, batch):
        d_fn = self.trainer.datamodule.data_transform
        source_data, target_data = batch
        source_data[0], target_data[0] = d_fn(source_data[0]), d_fn(target_data[0])
        if self.global_step == 1:
            self.draw_batch(source_data[0], target_data[0])
        return source_data, target_data

    # pylint: disable=arguments-differ,unused-argument
    def validation_step(self, batch, batch_idx):
        batch = [batch["source"], batch["target"]]
        source_data, target_data = self.get_real_data(batch)
        _, loss_info, output_feat, label_probs = self.loss_map(
            source_data, return_pf=True
        )
        return loss_info["map_loss/cost_loss"], [output_feat, label_probs], target_data

    def validation_epoch_end(self, outputs):
        total_cost_loss = 0

        for idx, out in enumerate(outputs):
            target_value, pushforward_data, target_data = out
            total_cost_loss += target_value
            pf_feat, pf_probs = pushforward_data
            pf_labels = torch.argmax(pf_probs, dim=1)
            target_feat, target_label = get_feat_label(target_data)
            if idx == 0:
                stacked_pf_feat = pf_feat
                stacked_pf_label = pf_labels
                stacked_target_feat = target_feat
                stacked_target_label = target_label
            else:
                stacked_pf_feat = torch.cat([stacked_pf_feat, pf_feat], dim=0)
                stacked_pf_label = torch.cat([stacked_pf_label, pf_labels])
                stacked_target_feat = torch.cat(
                    [stacked_target_feat, target_feat], dim=0
                )
                stacked_target_label = torch.cat([stacked_target_label, target_label])

        stacked_pf_label = torch.from_numpy(
            rankdata(stacked_pf_label.cpu(), method="dense") - 1
        )
        pf_dataset = CustomTensorDataset(
            [
                stacked_pf_feat.cpu(),
                stacked_pf_label.to(torch.int64).cpu(),
            ]
        )
        target_dataset = CustomTensorDataset(
            [
                stacked_target_feat.cpu(),
                stacked_target_label.to(torch.int64).cpu(),
            ]
        )
        dist = DatasetDistance(
            pf_dataset,
            target_dataset,
            inner_ot_method="exact",
            inner_ot_debiased=True,
            inner_ot_entreg=1e-3,
            entreg=1e-3,
            device=self.device,
            λ_x=self.cfg.coeff_feat,
            λ_y=self.cfg.coeff_label,
        )
        ps_otdd_gap = dist.distance(maxsamples=5000).item()

        num_step = len(outputs)
        total_cost_loss /= num_step
        self.log_dict(
            {
                "otdd/target_value": total_cost_loss,
                "otdd/pf_target_otdd_gap": ps_otdd_gap,
            }
        )
