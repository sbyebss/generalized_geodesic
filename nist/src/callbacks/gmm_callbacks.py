import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
from torch.distributions.categorical import Categorical

from src.models.base_model import get_feat_label
from src.viz.points import draw_trajectory


class DataCb(Callback):
    def on_fit_start(self, trainer, pl_module) -> None:
        if hasattr(trainer.datamodule, "to_device"):
            trainer.datamodule.to_device(pl_module.device)


# pylint: disable=too-many-locals
def generate_geodesic(
    source_label, pf_probs, source_feat, pf_feat, num_source_class, num_target_class
):
    geodesic = []
    source_label = source_label.view(-1).long()
    source_probs = F.one_hot(
        source_label, num_classes=num_source_class + num_target_class
    ).float()
    patch = torch.zeros([pf_probs.shape[0], num_source_class]).to(pf_probs.device)
    pf_cat_probs = torch.cat([patch, pf_probs], axis=1)

    for time in torch.linspace(0, 1, 5):
        label_probs = source_probs * (1 - time) + pf_cat_probs * time
        catg = Categorical(probs=label_probs)
        output_label = catg.sample().reshape(-1, 1)
        mixup_feat = source_feat * (1 - time) + pf_feat * time
        mixup_feat_label = torch.cat([mixup_feat, output_label], dim=1)
        geodesic.append(mixup_feat_label)
    return geodesic


class GMMCb(Callback):
    def __init__(self, log_interval, num_test_sample) -> None:
        super().__init__()
        self.log_interval = log_interval
        self.num_test_sample = num_test_sample

    def on_batch_start(self, trainer, pl_module) -> None:
        if (pl_module.global_step + 1) % self.log_interval == 0:
            with torch.no_grad():
                source, target = trainer.datamodule.get_test_samples(
                    self.num_test_sample
                )
                source = source.to(pl_module.device)
                source_feat, source_label = get_feat_label(source)
                if pl_module.current_epoch < pl_module.cfg.classifier_epoch:
                    output_feat = source_feat
                else:
                    output_feat = pl_module.map_t(source_feat, source_label)

                pf_label_logits = pl_module.classifier(output_feat, source_label)
                pf_label_probs = F.softmax(pf_label_logits, dim=1)
            geodesic = generate_geodesic(
                source_label,
                pf_label_probs,
                source_feat,
                output_feat,
                pl_module.cfg.num_source_class,
                pl_module.cfg.num_target_class,
            )
            draw_trajectory(
                geodesic,
                target,
                f"step{pl_module.global_step}.png",
                pl_module.cfg.num_source_class,
                pl_module.cfg.num_target_class,
            )
