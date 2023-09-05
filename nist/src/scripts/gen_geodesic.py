"""
Train the LeNet classifier on the generalized geodesic:
The existing labelled datasets are three datasets from *NIST.
The external test dataset is a 20-shot dataset from *NIST.
"""
# pylint: disable=line-too-long,not-callable,redefined-outer-name,too-many-locals,no-value-for-parameter,too-many-statements,bare-except,wildcard-import
import os
from collections import defaultdict

from ternary.helpers import simplex_iterator

import wandb
from src.datamodules.few_shot_datamodule import FewShotNIST
from src.networks.classifier import LeNet
from src.scripts import *
from src.transfer_learning.gen_geodesic import tuple_dataloader
from src.transfer_learning.mix_transformation import get_geodesic_mix_by_method
from src.transfer_learning.train_nist_classifier import TupleSampler
from src.viz.stat import draw_ternary_heatmap

log = lht_utils.get_logger(__name__)


@hydra.main(config_path="../../configs/scripts", config_name="gen_geodesic")
def gen_geodesic(cfg: omegaconf.DictConfig):
    best_id = get_gpu_by_utils(1, sleep_sec=10)
    cfg.device = f"cuda:{best_id[0]}"

    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=cfg.logger.project, name=f"tl_{cfg.logger.name}", config=wandb.config
    )

    num_target_classes = [get_num_label(ds) for ds in cfg.train_datasets]
    total_label = sum(num_target_classes)
    target_alias = "".join(ds[0] for ds in cfg.train_datasets)
    num_source_label = get_num_label(cfg.fine_tune_dataset)
    num_seed = len(cfg.seeds)
    accuracies = defaultdict(list)

    for seed in cfg.seeds:
        # Firstly, dump all the three datasets.
        dl_save_path = f"pf_dl_{target_alias}_seed{seed}.pt"
        tuple_dl = tuple_dataloader(
            cfg.method,
            cfg,
            seed=seed,
            dl_path=dl_save_path,
        )
        pf_sampler = TupleSampler(tuple_dl)

        for index, simplex_vector in enumerate(simplex_iterator(cfg.num_segment)):
            classifier_path = f"t{index}_classifier_{target_alias}_seed{seed}.pt"
            # Training for different interpolation time
            # If you already run this part code and save the trained classifier
            # you can skip this part of code.
            if not os.path.exists(classifier_path) or cfg.retrain:
                simplex_vector = np.array(simplex_vector) / cfg.num_segment

                lenet = LeNet(num_class=total_label)
                lenet = lenet.to(cfg.device)
                optimizer = optim.Adam(lenet.parameters(), lr=1e-3)

                data_transformer = get_geodesic_mix_by_method(
                    cfg.method, simplex_vector, num_target_classes, cfg.device
                )
                feat, labels = train_classifier_on_sampler(
                    lenet,
                    optimizer,
                    pf_sampler,
                    data_transformer,
                    cfg.train_iteration,
                    device=cfg.device,
                )
                save_tensor_imgs(
                    inverse_data_transform(feat),
                    8,
                    0,
                    f"train_on_{target_alias}_time_{index}_seed{seed}",
                )
                log.info(f"sanity check labels: {labels}")
                log.info(
                    f"Finish training on one interpolation, the weight is {simplex_vector}"
                )
                torch.save(
                    {"model_state_dict": lenet.state_dict()},
                    classifier_path,
                )
            else:
                log.info(
                    f"Load the pretrained classifier, the weight is {simplex_vector}"
                )

        # Fine-tune dataloder
        nist_few_shot_dataloader = FewShotNIST(
            data_path=cfg.nist_data_path,
            fine_tune_dataset=cfg.fine_tune_dataset,
            num_shot=cfg.num_shot,
            batch_size=cfg.batch_size,
            img_size=cfg.img_size,
            seed=seed,
        )

        # sanity check
        assert nist_few_shot_dataloader.num_data == cfg.num_shot * num_source_label

        # Fine-tuning, and testing for different interpolation time
        for index, simplex_vector in enumerate(simplex_iterator(cfg.num_segment)):
            simplex_vector = np.array(simplex_vector) / cfg.num_segment

            lenet = LeNet(num_class=total_label)
            lenet.load_state_dict(
                torch.load(f"t{index}_classifier_{target_alias}_seed{seed}.pt")[
                    "model_state_dict"
                ]
            )
            # turn_off_grad(lenet)
            # Now we choose to fine-tune all the layers
            # instead of only the last one
            lenet = fine_tune_lenet(
                lenet,
                nist_few_shot_dataloader.fine_tune_loader,
                cfg.fine_tune_epoch,
                num_source_label,
                cfg.device,
                f"fine_tune_{cfg.fine_tune_dataset}_time{index}",
            )

            accuracy = test_lenet(
                lenet, nist_few_shot_dataloader.test_loader, cfg.device
            ).item()
            accuracies[tuple(simplex_vector * cfg.num_segment)[:2]].append(accuracy)

            log.info(
                f"Finish fine tuning on one interpolation, t{index}, seed{seed}, accuracy is",
                accuracy,
            )
    accuracies = {key: sum(acc) / len(acc) for key, acc in accuracies.items()}
    accuracy_save_path = (
        f"train_on_{target_alias}_epoch{cfg.fine_tune_epoch}_repeat{num_seed}.pt"
    )
    torch.save(accuracies, accuracy_save_path)

    plot_save_path = (
        f"train_on_{target_alias}_epoch{cfg.fine_tune_epoch}_repeat{num_seed}.png"
    )
    draw_ternary_heatmap(
        accuracies,
        cfg.num_segment,
        plot_save_path,
        cfg.train_datasets,
    )
    wandb.log({"ternary_plot/acc": wandb.Image(plot_save_path)})


if __name__ == "__main__":
    gen_geodesic()
