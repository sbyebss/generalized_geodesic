import os
from functools import partial

import hydra
import omegaconf
import torch
from jammy.cli.gpu_sc import get_gpu_by_utils

from src.otdd.pytorch.distance import DatasetDistance
from src.transfer_learning.gen_geodesic import (
    get_knn_or_full_emb_vtab_data,
    get_knn_or_full_nist_data,
)
from src.transfer_learning.mix_transformation import transform_ds_feat
from src.transfer_learning.train_nist_classifier import get_num_label

# pylint: disable=no-value-for-parameter,too-many-locals


@hydra.main(config_path="../../configs/scripts", config_name="calculate_otdd")
def precompute_otdd_nist(cfg: omegaconf.DictConfig):
    assert cfg.source != cfg.target, "source and target must be different"
    assert (
        not cfg.source_few_shot or not cfg.target_few_shot
    ), "source and target cannot be few shot in the same time"
    assert not os.path.exists(cfg.otdd_flipped_path[0]) or not os.path.exists(
        cfg.otdd_flipped_path[1]
    ), "This pair of datasets has already been precomputed"
    if cfg.source_few_shot or cfg.target_few_shot:
        assert not os.path.exists(
            cfg.otdd_few_shot_path
        ), "This pair of datasets has already been precomputed long time ago"
    else:
        assert not os.path.exists(
            cfg.otdd_full_dataset_path
        ), "This pair of datasets has already been precomputed long time ago"

    if not os.path.exists(cfg.otdd_dir):
        os.makedirs(cfg.otdd_dir)

    if cfg.auto_gpu:
        best_id = get_gpu_by_utils(1, sleep_sec=10)
        cfg.device = f"cuda:{best_id[0]}"
    # cfg.device='cpu'
    if cfg.source in cfg.nist_datasets or cfg.target in cfg.nist_datasets:
        load_data = partial(
            get_knn_or_full_nist_data,
            nist_data_path=cfg.nist_data_path,
            batch_size=cfg.batch_size,
            img_size=cfg.img_size,
        )
        source_zero_out_ds, source_dl = load_data(
            ds_name=cfg.source,
            knn_data_path=cfg.source_knn_data_path,
            few_shot=cfg.source_few_shot,
        )
        target_zero_out_ds, target_dl = load_data(
            ds_name=cfg.target,
            knn_data_path=cfg.target_knn_data_path,
            few_shot=cfg.target_few_shot,
        )

        source_ds = transform_ds_feat(source_zero_out_ds, source_dl, cfg.batch_size)
        target_ds = transform_ds_feat(target_zero_out_ds, target_dl, cfg.batch_size)
    else:
        load_data = partial(
            get_knn_or_full_emb_vtab_data,
            vtab_data_path=cfg.vtab_data_path,
            batch_size=cfg.batch_size,
        )
        source_ds, source_dl = load_data(
            ds_name=cfg.source,
            knn_data_path=cfg.source_knn_data_path,
            few_shot=cfg.source_few_shot,
        )
        target_ds, target_dl = load_data(
            ds_name=cfg.target,
            knn_data_path=cfg.target_knn_data_path,
            few_shot=cfg.target_few_shot,
        )

    # FIXME: now I fix the lambdas, this can be a problem.
    dist = DatasetDistance(
        source_ds,
        target_ds,
        inner_ot_method="exact",
        inner_ot_debiased=True,
        inner_ot_entreg=1e-3,
        entreg=1e-3,
        device=cfg.device,
        λ_x=cfg.coeff_feat,
        λ_y=cfg.coeff_label,
    )

    distance = []
    for _ in range(6):
        distance.append(dist.distance(maxsamples=10000).item())
        print("Calculate OTDD once:", distance[-1])

    otdd_distance = sum(distance) / len(distance)
    # Here we divide by p=2 because OTDD also divide by p
    # See https://github.com/microsoft/otdd/blob/main/otdd/pytorch/distance.py#L1325
    w2_matrix = dist.pwlabel_stats["dlabs"] / 2
    assert w2_matrix.shape[0] == get_num_label(
        cfg.source
    ), "source label number is not equal to the first dimension of w2 matrix"
    # print("w2 matrix=", w2_matrix)
    print("Done! OTDD=", otdd_distance, "W2 min=", w2_matrix.min().item())
    stat_source = {
        "otdd": otdd_distance,
        "w2_matrix": w2_matrix.cpu(),
        "w2 min": w2_matrix.min().item(),
    }
    stat_target = {
        "otdd": otdd_distance,
        "w2_matrix": w2_matrix.cpu().T,
        "w2 min": w2_matrix.min().item(),
    }
    # Do this check again because we're running in parallel, there could still be some overlapping.
    assert not os.path.exists(cfg.otdd_flipped_path[0]) or not os.path.exists(
        cfg.otdd_flipped_path[1]
    ), "This pair of datasets has already been precomputed"

    if cfg.source_few_shot or cfg.target_few_shot:
        few_shot_ds = cfg.source if cfg.source_few_shot else cfg.target
        stat = stat_source if cfg.source_few_shot else stat_target
        save_path = cfg.otdd_few_shot_path
        if os.path.exists(save_path):
            dict_data = torch.load(save_path)
        else:
            dict_data = {}
        dict_data[few_shot_ds] = stat
    else:
        dict_data = {}
        dict_data[cfg.source] = stat_source
        dict_data[cfg.target] = stat_target
        save_path = cfg.otdd_full_dataset_path
    torch.save(dict_data, save_path)


if __name__ == "__main__":
    precompute_otdd_nist()
