# This script solves the accuracy of fine-tuning vs training
# directly on a x-shot dataset. x can be 5, 20, ...
import os

import hydra
import numpy as np
import omegaconf
import torch
from jammy.cli.gpu_sc import get_gpu_by_utils

import wandb
from src.transfer_learning.gen_geodesic import (
    get_best_interp_param,
    get_knn_dl_and_otdd_map,
    get_otdd_stat,
    get_w2_matrix,
    ternary_otdd_interpolation,
    transport_metric,
)
from src.utils import lht_utils
from src.viz.stat import draw_ternary_heatmap

log = lht_utils.get_logger(__name__)


# pylint: disable=too-many-locals,no-value-for-parameter,undefined-loop-variable,too-many-statements,line-too-long,too-many-nested-blocks,too-many-branches
@hydra.main(config_path="../../configs/scripts", config_name="otdd_ternary")
def otdd_ternary_plot(cfg: omegaconf.DictConfig):
    if cfg.auto_gpu:
        best_id = get_gpu_by_utils(1, sleep_sec=10)
        cfg.device = f"cuda:{best_id[0]}"  # cfg.device='cpu'

    if cfg.full_dataset:
        cfg.train_datasets = cfg.all_datasets
        if cfg.fine_tune_dataset in cfg.train_datasets:
            cfg.train_datasets.remove(cfg.fine_tune_dataset)

    if cfg.ds_type == "NIST":
        target_alias = "".join(ds[0] for ds in cfg.train_datasets)
    elif cfg.ds_type == "VTAB":
        target_alias = "".join(ds + "_" for ds in cfg.train_datasets)

    num_train_dataset = len(cfg.train_datasets)
    num_seed = len(cfg.seeds)

    # Then we can use the mapped dataset to calculate OTDD distance
    otdd_distance_path = (
        f"from_{cfg.fine_tune_dataset}2{target_alias}_repeat{num_seed}.pth"
    )
    if os.path.exists(otdd_distance_path):
        otdd_stat = torch.load(otdd_distance_path)
        # The i entry in w2_vector_dict is Wasserstein distance between (\nu,\mu_i)
        # The (i,j) entry in w2_matrix_dict is Wasserstein distance between (\mu_i,\mu_j)
        w2_vector_dict, w2_matrix_dict = (
            otdd_stat["W(nu,mu_i)"],
            otdd_stat["W(mu_i,mu_j)"],
        )
    else:
        w2_matrix_dict = {}
        w2_vector_dict = {}
        best_interpo_params = {}
        for seed in cfg.seeds:
            w2_matrix_internal = np.zeros([num_train_dataset, num_train_dataset])
            w2_between_internal_external = np.zeros(num_train_dataset)
            map_list, classifier_list, external_loader = get_knn_dl_and_otdd_map(
                cfg, seed
            )
            for idx_i in range(num_train_dataset):
                for idx_j in range(num_train_dataset):
                    if idx_i == idx_j:
                        # For the distance between external and any internal dataset
                        # we just load the pre-computed OTDD
                        suffix = f"seed{seed}_{cfg.num_shot}shot.pt"
                        stat = get_otdd_stat(
                            cfg.otdd_dir,
                            cfg.fine_tune_dataset,
                            cfg.train_datasets[idx_i],
                            suffix,
                        )
                        distance = stat[cfg.fine_tune_dataset]["otdd"]
                        w2_between_internal_external[idx_i] = distance
                        log.info(
                            f"Loaded OTDD between source and {idx_i} datasets, distance={distance}"
                        )
                    else:
                        map_i = map_list[idx_i]
                        target_classifier_i = classifier_list[idx_i]
                        map_j = map_list[idx_j]
                        target_classifier_j = classifier_list[idx_j]
                        suffix = "full.pt"
                        w2_matrix_betwen_ij = get_w2_matrix(
                            cfg.otdd_dir,
                            cfg.train_datasets[idx_i],
                            cfg.train_datasets[idx_j],
                            suffix,
                        )
                        distance = transport_metric(
                            external_loader,
                            map_i,
                            map_j,
                            target_classifier_i,
                            target_classifier_j,
                            cfg.coeff_feat,
                            cfg.coeff_label,
                            w2_matrix_betwen_ij,
                            cfg.device,
                        )
                        w2_matrix_internal[idx_i, idx_j] = w2_matrix_internal[
                            idx_j, idx_i
                        ] = distance
                        log.info(
                            f"Finish solving OTDD between {idx_i} and {idx_j} datasets, distance={distance}"
                        )
            key = f"seed{seed}"
            w2_matrix_dict[key] = w2_matrix_internal
            w2_vector_dict[key] = w2_between_internal_external
            best_interpo_params[key] = get_best_interp_param(
                w2_between_internal_external, w2_matrix_internal
            )
            log.info(
                f"Best interpolation parameter for seed{seed} is {list(best_interpo_params[key])}"
            )
        torch.save(
            {
                "W(nu,mu_i)": w2_vector_dict,
                "W(mu_i,mu_j)": w2_matrix_dict,
                "best_interpo_params": best_interpo_params,
            },
            otdd_distance_path,
        )

    if num_train_dataset == 3:
        # Finaly, approximate all OTDD distance
        avg_w2_matrix = sum(w2_matrix_dict.values()) / len(w2_matrix_dict)
        avg_w2_vector = sum(w2_vector_dict.values()) / len(w2_vector_dict)
        otdd_distance = ternary_otdd_interpolation(
            avg_w2_vector, avg_w2_matrix, cfg.num_segment
        )
        plot_save_path = (
            f"from_{cfg.fine_tune_dataset}2{target_alias}_repeat{num_seed}.png"
        )
        draw_ternary_heatmap(
            otdd_distance,
            cfg.num_segment,
            plot_save_path,
            cfg.train_datasets,
        )
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            project=cfg.logger.project,
            name=f"tl_{cfg.logger.name}",
            config=wandb.config,
        )
        wandb.log({"ternary_plot/otdd": wandb.Image(plot_save_path)})


if __name__ == "__main__":
    otdd_ternary_plot()
