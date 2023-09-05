from copy import deepcopy
from functools import partial

import numpy as np
import torch

from src.otdd.pytorch.datasets import CustomTensorDataset
from src.transfer_learning.gen_geodesic import (
    get_knn_or_full_nist_data,
    get_zero_out_ds_soft_label,
)
from src.transfer_learning.mix_transformation import (
    barycentric_projection,
    transform_ds_feat_label,
)
from src.transfer_learning.train_nist_classifier import (
    LoaderSampler,
    get_nist_num_label,
    simple_transformation,
)
from src.utils import lht_utils

log = lht_utils.get_logger(__name__)


# pylint: disable=bare-except
def target_dataloader(method, cfg, nist_few_shot_dl, **kwargs):
    assert method in ["otdd_map", "mixup", "barycenteric_map"], "method not supported"
    seed = kwargs["seed"]
    dl_path = kwargs["dl_path"]
    try:
        target_dl = torch.load(dl_path)
        log.info(f"{method}: get boosted dataloader in seed {seed}.")
        return target_dl
    except:
        if method == "otdd_map":
            target_dl = otdd_map_target_dataloader(cfg, seed=seed)
        elif method == "mixup":
            # already transformed
            target_dl = mixup_target_dataloader(cfg, nist_few_shot_dl, seed=seed)
        elif method == "barycenteric_map":
            # already transformed
            target_dl = bary_map_target_dataloader(cfg, nist_few_shot_dl, seed=seed)
        torch.save(target_dl, dl_path)
        return target_dl


# --------------------- Mixup ---------------------


def mixup_transform_func(
    feat, labels, few_shot_data_sampler, num_class, device, alpha=0.2
):
    # labels: (batch_size,) hard labels
    del feat, labels
    feat, hard_label = few_shot_data_sampler.sample()
    feat, hard_label = simple_transformation(feat, hard_label)
    feat = feat.to(device)
    hard_label = hard_label.to(device)
    lam = np.random.beta(alpha, alpha)
    soft_label = torch.nn.functional.one_hot(hard_label, num_classes=num_class).float()
    mix_feat = lam * feat + (1 - lam) * feat.flip(0)
    mix_label = lam * soft_label + (1 - lam) * soft_label.flip(0)
    return mix_feat, mix_label


def mixup_target_dataloader(cfg, nist_few_shot_dl, seed=1):
    tf_batch_size = min(cfg.batch_size, nist_few_shot_dl.num_data)
    _, source_loader = get_knn_or_full_nist_data(
        cfg.source_dataset,
        cfg.nist_data_path,
        tf_batch_size,
        cfg.img_size,
        few_shot=False,
    )
    n_class = get_nist_num_label(cfg.mapped_dataset)
    zero_out_ds = get_zero_out_ds_soft_label(source_loader, num_class=n_class)

    few_shot_data_sampler = LoaderSampler(nist_few_shot_dl.fine_tune_loader)
    tf_func = partial(
        mixup_transform_func,
        few_shot_data_sampler=few_shot_data_sampler,
        num_class=n_class,
        device=cfg.device,
    )

    mixed_ds = transform_ds_feat_label(zero_out_ds, source_loader, tf_func=tf_func)
    target_dl = torch.utils.data.DataLoader(
        mixed_ds,
        batch_size=cfg.batch_size,
    )
    log.info(f"mixup: get boosted dataloader in seed {seed}.")

    return target_dl


# --------------------- barycentric map ---------------------


def bary_map_transform_func(feat, label, target_ds, num_class, device):
    # feat: size = (b,c,h,w)
    # label: size = (b,) is hard label
    # target_ds: nist_dataset (target_ds.data, target_ds.targets)
    # also contains hard labels.
    target_ds = deepcopy(target_ds)

    if feat.shape[1] == 1:
        feat = feat.expand(feat.shape[0], 3, *feat.shape[2:])
    source_ds = CustomTensorDataset(
        [
            feat,
            label.to(torch.int64),
        ]
    )
    # There is a problem here: otdd is not automatically making
    # two datasets the same dimension.
    # It only changes the dimension to be (32,32), but don't extend channel.
    # We use coefficients as both 1.0

    # if len(target_ds) < 1000:
    #     target_few_shot = True
    pf_feat, pf_probs = barycentric_projection(
        source_ds,
        target_ds,
        feat.shape,
        num_class,
        device,
        # target_few_shot=target_few_shot,
    )
    return pf_feat * 2 - 1.0, pf_probs


def bary_map_target_dataloader(cfg, nist_few_shot_dl, seed=1):
    batch_size = 10000
    _, source_loader = get_knn_or_full_nist_data(
        cfg.source_dataset,
        cfg.nist_data_path,
        batch_size,
        cfg.img_size,
        few_shot=False,
    )
    n_class = get_nist_num_label(cfg.mapped_dataset)
    zero_out_ds = get_zero_out_ds_soft_label(source_loader, num_class=n_class)

    tf_func = partial(
        bary_map_transform_func,
        target_ds=nist_few_shot_dl.fine_tune_loader.dataset,
        num_class=n_class,
        device=cfg.device,
    )

    mixed_ds = transform_ds_feat_label(zero_out_ds, source_loader, tf_func=tf_func)
    target_dl = torch.utils.data.DataLoader(
        mixed_ds,
        batch_size=cfg.batch_size,
    )

    log.info(f"barycentric mapping: get boosted dataloader in seed {seed}.")

    return target_dl


# pylint: disable=unused-argument


def otdd_map_target_dataloader(cfg, seed=1):
    return "Not implemented"
