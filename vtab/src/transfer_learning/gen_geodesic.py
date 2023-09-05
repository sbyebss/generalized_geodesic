import os
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from jamtorch.data import get_batch
from qpsolvers import solve_qp
from ternary.helpers import simplex_iterator

from src.callbacks.w2_callbacks import knn_data, torchify_targets, transform2torch
from src.datamodules.datasets.small_scale_image_dataset import PyTorchDataset, nist_dataset
from src.models.loss_zoo import label_cost, mse_loss
from src.otdd.pytorch.datasets import CustomTensorDataset
from src.transfer_learning.data_utils import get_vtab_dataset, get_train_test_dataset, get_vtab_train_dataset
from src.transfer_learning.mix_transformation import (
    ConcatDataset,
    bary_map_transform_func,
    map_classifier_list_loader,
    otdd_map_transform_func,
    transform_ds_feat_label,
)
from src.transfer_learning.train_nist_classifier import get_num_label, simple_transformation
from src.utils import lht_utils

log = lht_utils.get_logger(__name__)

nist_list = ["MNIST", "USPS", "FMNIST", "KMNIST", "EMNIST", "MNISTM"]
# pylint: disable=too-many-locals,too-many-arguments,undefined-loop-variable,bare-except,line-too-long


def get_otdd_stat(otdd_dir, ds1, ds2, suffix):
    try:
        stat = torch.load(
            os.path.join(
                otdd_dir,
                f"{ds1}_{ds2}_{suffix}",
            )
        )
    except:
        stat = torch.load(
            os.path.join(
                otdd_dir,
                f"{ds2}_{ds1}_{suffix}",
            )
        )
    return stat


def get_w2_matrix(otdd_dir, ds1, ds2, suffix):
    try:
        w2_matrix_betwen_ij = torch.load(
            os.path.join(
                otdd_dir,
                f"{ds1}_{ds2}_{suffix}",
            )
        )[ds1]["w2_matrix"]
    except:
        w2_matrix_betwen_ij = torch.load(
            os.path.join(
                otdd_dir,
                f"{ds2}_{ds1}_{suffix}",
            )
        )[ds1]["w2_matrix"]
    return w2_matrix_betwen_ij


def get_best_interp_param(w2_between_internal_external, w2_matrix_internal):
    # w2_between_internal_external (# datasets,)
    # w2_matrix_internal (# datasets, # datasets) is a symmetric matrix
    n_ds = w2_matrix_internal.shape[0]
    param = solve_qp(
        -w2_matrix_internal,
        w2_between_internal_external,
        A=np.ones(n_ds),
        b=np.array([1.0]),
        lb=np.zeros(n_ds),
        solver="osqp",
    )
    return np.array(param)


def ternary_otdd_interpolation(
    w2_between_internal_external, w2_matrix_internal, num_segment
):
    otdd_distance = {}
    for simplex_tuple in simplex_iterator(num_segment):
        # This simplex_tuple is not really on a simplex, it's unnormalized.
        simplex_vector = np.array(simplex_tuple) / num_segment
        assert np.isclose(sum(simplex_vector), 1)
        assert w2_between_internal_external.shape == simplex_vector.shape

        simplex_vector_n1 = simplex_vector.reshape(-1, 1)
        otdd_distance[simplex_tuple[:2]] = (
            np.inner(w2_between_internal_external, simplex_vector)
            - (simplex_vector_n1.T @ w2_matrix_internal @ simplex_vector_n1 / 2).item()
        )
    return otdd_distance


def get_zero_out_ds_hard_label(dataloader, dim=32):
    dataset = deepcopy(dataloader.dataset)
    return CustomTensorDataset(
        [
            torch.zeros([len(dataset), 3, dim, dim]),
            dataset.targets.to(torch.int64),
        ]
    )


# Here we don't change the type of labels to be int64,
# because they're soft labels, not hard label.
def get_zero_out_ds_soft_label(dataloader, dim=32, num_channel=3, num_class=10):
    dataset = deepcopy(dataloader.dataset)
    # TODO: this is a hack, we should fix this.
    if type(dataset).__name__ in nist_list:
        return CustomTensorDataset(
            [
                torch.zeros([len(dataset), num_channel, dim, dim]),
                torch.zeros([len(dataset), num_class]),
            ]
        )
    else:
        one_batch = get_batch(dataloader)
        data_shape = one_batch[0][0].shape
        return CustomTensorDataset(
            [
                torch.zeros([len(dataset), *data_shape]),
                torch.zeros([len(dataset), num_class]),
            ]
        )


def get_emb_mae_data(
    vtab_data_path: str,
    ds_name: str,
    fold: str,
    batch_size: int,
    shuffle: bool = True,
    shuffle_ahead: bool = False,
):
    dataset = get_vtab_dataset(vtab_data_path, ds_name, fold)
    dataset = torchify_targets(dataset)
    if shuffle:
        if shuffle_ahead:
            perm = torch.randperm(len(dataset)).tolist()
            shuffled_ds = torch.utils.data.Subset(dataset, perm)
            dataloader = torch.utils.data.DataLoader(
                shuffled_ds, batch_size=batch_size, shuffle=False
            )
            return shuffled_ds, dataloader
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
        return dataset, dataloader
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        return dataset, dataloader


def get_knn_or_full_emb_vtab_data(
    ds_name: str,
    vtab_data_path: str,
    batch_size: int,
    knn_data_path: str = None,
    few_shot: bool = True,
):
    dataset = get_vtab_train_dataset(vtab_data_path, ds_name)
    # dataset = transform2torch(dataset)
    dataset = torchify_targets(dataset)
    # dataset = PyTorchDataset([dataset.data, dataset.targets])
    # breakpoint()
    if few_shot:
        assert knn_data_path, "knn_data_path is not provided."
        dataset = knn_data(dataset, knn_data_path)
        dataset.targets = dataset.targets.to(torch.int64)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return dataset, dataloader


def get_knn_or_full_nist_data(
    ds_name: str,
    nist_data_path: str,
    batch_size: int,
    img_size: int,
    knn_data_path: str = None,
    few_shot: bool = True,
):
    dataset, _ = nist_dataset(ds_name, nist_data_path, img_size)
    dataset = transform2torch(dataset)
    if few_shot:
        assert knn_data_path, "knn_data_path is not provided."
        dataset = knn_data(dataset, knn_data_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    zero_out_dataset = get_zero_out_ds_hard_label(dataloader)
    return zero_out_dataset, dataloader


def neural_transport_metric(
    external_loader,
    map_i,
    map_j,
    target_classifier_i,
    target_classifier_j,
    coeff_feat,
    coeff_label,
    w2_matrix_betwen_ij,
    device,
):
    total_distance = 0.0
    w2_matrix_betwen_ij = w2_matrix_betwen_ij.to(device)
    for (feat, labels) in external_loader:
        feat = feat.to(device)
        labels = labels.to(device)
        feat, labels = simple_transformation(feat, labels)
        with torch.no_grad():
            mapped_feat1 = map_i(feat, labels)
            mapped_logits1 = target_classifier_i(mapped_feat1, None)
            mapped_labels1 = torch.argmax(mapped_logits1, dim=1)
            mapped_feat2 = map_j(feat, labels)
            mapped_logits2 = target_classifier_j(mapped_feat2, None)
            mapped_probs2 = F.softmax(mapped_logits2, dim=1)
        total_distance += (
            coeff_feat * mse_loss(mapped_feat1, mapped_feat2)
            + coeff_label
            * label_cost(w2_matrix_betwen_ij, mapped_labels1, mapped_probs2)
        ) * feat.shape[0]
    return total_distance.item() / len(external_loader.dataset)


def get_knn_or_full_dataloader(cfg, batch_size=None, knn_data_path=None):
    if batch_size is None:
        batch_size = cfg.batch_size
    if cfg.ds_type == "NIST":
        _, external_loader = get_knn_or_full_nist_data(
            cfg.fine_tune_dataset,
            cfg.nist_data_path,
            batch_size,
            cfg.img_size,
            knn_data_path,
        )
    elif cfg.ds_type == "VTAB":
        _, external_loader = get_knn_or_full_emb_vtab_data(
            cfg.fine_tune_dataset,
            cfg.vtab_data_path,
            batch_size,
            knn_data_path,
        )
    return external_loader


def get_knn_dl_and_otdd_map(cfg, seed):
    map_list, classifier_list = map_classifier_list_loader(
        cfg.load_epochs,
        cfg.fine_tune_dataset,
        cfg.train_datasets,
        cfg.device,
        cfg.pretrained_classifier_path,
        cfg.otdd_map_dir,
        seed=seed,
        num_shot=cfg.num_shot,
    )
    # Train Dataloder
    knn_data_path = f"{cfg.work_dir}/data/knn_results/{cfg.fine_tune_dataset}_seed{seed}_{cfg.num_shot}shot.pt"
    external_loader = get_knn_or_full_dataloader(cfg, knn_data_path=knn_data_path)
    return map_list, classifier_list, external_loader


def tuple_dataloader(method, cfg, **kwargs):
    assert method in ["otdd_map", "mixup", "barycenteric_map"], "method not supported"
    seed = kwargs["seed"]
    dl_path = kwargs["dl_path"]
    if method == "otdd_map":
        return otdd_map_tuple_dataloader(dl_path, cfg, seed=seed)
    elif method == "mixup":
        return mixup_tuple_dataloader(cfg, seed=seed)
    elif method == "barycenteric_map":
        return bary_map_tuple_dataloader(dl_path, cfg, seed=seed)
    return "Not implemented"


# ------------------------otdd map -----------------


def otdd_map_tuple_dataloader(dl_save_path, cfg, seed=1):
    # Firstly, dump all the pushforward datasets.
    num_target_classes = [get_num_label(ds) for ds in cfg.train_datasets]
    try:
        pf_dl = torch.load(dl_save_path)
    except:
        map_list, classifier_list, external_loader = get_knn_dl_and_otdd_map(cfg, seed)
        pf_datasets = []
        for index_ds, n_class in enumerate(num_target_classes):
            zero_out_ds = get_zero_out_ds_soft_label(external_loader, num_class=n_class)
            tf_func = partial(
                otdd_map_transform_func,
                feat_map=map_list[index_ds],
                classifier=classifier_list[index_ds],
                device=cfg.device,
            )
            pf_ds = transform_ds_feat_label(
                zero_out_ds, external_loader, tf_func=tf_func
            )
            pf_datasets.append(pf_ds)

        # pf_ds is already transformed, don't need further transformation.
        pf_dl = torch.utils.data.DataLoader(
            ConcatDataset(pf_datasets),
            batch_size=cfg.train_batch_size,
        )
        torch.save(pf_dl, dl_save_path)

    log.info(f"OTDD map: Loaded all pushforward dataloaders in seed {seed}.")

    return pf_dl


# ------------------------ mixup -----------------


def randomize_ds_feat_label(dataset):
    """
    Randomize the order of the data in the dataset.
    """
    perm = torch.randperm(len(dataset))
    return torch.utils.data.Subset(dataset, perm)


def mixup_tuple_dataloader(cfg, seed=1):
    marginal_datasets = []
    for ds_name in cfg.train_datasets:
        dataset, _ = get_train_test_dataset(cfg, ds_name)
        if cfg.ds_type == "VTAB":
            dataset = PyTorchDataset([dataset.data, dataset.targets])
        random_ds = randomize_ds_feat_label(dataset)
        marginal_datasets.append(random_ds)

    # random_ds is not transformed, need further transformation.
    tuple_dl = torch.utils.data.DataLoader(
        ConcatDataset(marginal_datasets),
        batch_size=cfg.train_batch_size,
    )
    log.info(f"mixup: Loaded all existing randomized dataloaders in seed {seed}.")

    return tuple_dl


# ------------------------barycentric map -----------------


def get_num_channel(ds_name):
    return 1 if ds_name in ["MNIST", "USPS", "FMNIST", "KMNIST", "EMNIST"] else 3


def bary_map_tuple_dataloader(dl_save_path, cfg, seed=1):
    num_target_classes = [get_num_label(ds) for ds in cfg.train_datasets]
    try:
        pf_dl = torch.load(dl_save_path)
        # assert 1==0
    except:
        knn_data_path = f"{cfg.work_dir}/data/knn_results/{cfg.fine_tune_dataset}_seed{seed}_{cfg.num_shot}shot.pt"
        batch_size = 10000
        external_loader = get_knn_or_full_dataloader(cfg, batch_size, knn_data_path)
        pf_datasets = []
        for ds_name, n_class in zip(cfg.train_datasets, num_target_classes):
            # We calculate the coupling per batch, and use a relatively large batch size.
            zero_out_ds = get_zero_out_ds_soft_label(
                external_loader,
                num_channel=get_num_channel(cfg.fine_tune_dataset),
                num_class=n_class,
            )
            tg_dataset, _ = get_train_test_dataset(cfg, ds_name)
            if cfg.ds_type == "VTAB":
                tg_dataset = PyTorchDataset([tg_dataset.data, tg_dataset.targets])
            tg_dataset.targets = tg_dataset.targets.to(torch.int64)
            tf_func = partial(
                bary_map_transform_func,
                target_ds=tg_dataset,
                num_target_class=max(tg_dataset.targets) + 1,
                device=cfg.device,
            )
            pf_ds = transform_ds_feat_label(
                zero_out_ds, external_loader, tf_func=tf_func
            )
            pf_datasets.append(pf_ds)

        # pf_ds is already transformed, don't need further transformation.
        pf_dl = torch.utils.data.DataLoader(
            ConcatDataset(pf_datasets),
            batch_size=cfg.train_batch_size,
        )
        torch.save(pf_dl, dl_save_path)

    log.info(f"Barycentric mapping: Loaded all pushforward dataloaders in seed {seed}.")

    return pf_dl
