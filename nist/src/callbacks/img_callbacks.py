import pytorch_lightning as pl  # pylint: disable=unused-import
import torch
from pytorch_lightning import Callback
from torch.distributions.categorical import Categorical

from src.models.base_model import get_feat_label
from src.transfer_learning.train_nist_classifier import get_nist_num_label
from src.viz.img import save_tensor_imgs
from src.viz.points import draw_histogram

# pylint: disable=arguments-differ,too-many-instance-attributes,unused-argument


def map_images(source_feat, source_label, map_t, device, data_transform, ema=None):
    source_feat = data_transform(source_feat)
    source_feat = source_feat.to(device)
    source_label = source_label.to(device)
    if ema is not None:
        with ema.average_parameters():
            output_feat = map_t(source_feat, source_label)
    else:
        output_feat = map_t(source_feat, source_label)
    return source_feat, output_feat


class MapViz(Callback):
    def __init__(self, log_interval, source_dataset, map_path: str) -> None:
        super().__init__()
        self.log_interval = log_interval
        num_class = get_nist_num_label(source_dataset)
        self.num_test_sample = num_class * 5
        self.map_path = map_path

    def on_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        if pl_module.global_step % self.log_interval == 0:
            # Plot the "geodesic"
            with torch.no_grad():
                source = trainer.datamodule.get_test_samples(self.num_test_sample)
                source_feat, source_label = get_feat_label(source)
                source_feat, output_feat = map_images(
                    source_feat,
                    source_label,
                    pl_module.map_t,
                    pl_module.device,
                    trainer.datamodule.data_transform,
                    pl_module.ema_map,
                )

            geodesic_feat = source_feat
            for time in [0.5, 1.0]:
                interpl_feat = source_feat * (1 - time) + output_feat * time
                geodesic_feat = torch.cat([geodesic_feat, interpl_feat], dim=0)

            save_tensor_imgs(
                trainer.datamodule.inverse_data_transform(geodesic_feat),
                pl_module.cfg.num_source_class,
                pl_module.global_step,
                "pushforward",
            )
            # Plot the histogram of labels in pushforward distribution.
            with torch.no_grad():
                source = trainer.datamodule.get_test_samples(
                    self.num_test_sample, shuffle=True
                )
                source_feat, source_label = get_feat_label(source)
                _, output_feat = map_images(
                    source_feat,
                    source_label,
                    pl_module.map_t,
                    pl_module.device,
                    trainer.datamodule.data_transform,
                    pl_module.ema_map,
                )
                output_logits = pl_module.classifier(output_feat, source_label)
                catg = Categorical(logits=output_logits)
                output_label = catg.sample()
            output_label = output_label.detach().cpu().numpy()
            draw_histogram(output_label, f"histogram_{pl_module.global_step}.png")

    def on_test_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *args, **kwargs
    ) -> None:
        with torch.no_grad():
            source = trainer.datamodule.get_test_samples(self.num_test_sample)
            source_feat, source_label = get_feat_label(source)
            pl_module.map_t.load_state_dict(torch.load(self.map_path))
            source_feat, output_feat = map_images(
                source_feat,
                source_label,
                pl_module.map_t,
                pl_module.device,
                trainer.datamodule.data_transform,
                None,
            )

        save_tensor_imgs(
            trainer.datamodule.inverse_data_transform(source_feat),
            pl_module.cfg.num_source_class,
            0,
            "source",
        )

        save_tensor_imgs(
            trainer.datamodule.inverse_data_transform(output_feat),
            pl_module.cfg.num_source_class,
            0,
            "pushforward",
        )
