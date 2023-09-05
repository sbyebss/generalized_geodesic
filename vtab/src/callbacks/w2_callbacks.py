import sys
from copy import deepcopy
from os.path import join

import numpy as np
import scipy.linalg as ln
import torch
from einops import repeat
from pytorch_lightning import Callback
from torch.utils.data import DataLoader

from src.callbacks.otdd_helper import pwdist_exact
from src.datamodules.datasets.small_scale_image_dataset import PyTorchDataset
from src.models.base_model import get_feat_label
from src.utils import lht_utils

log = lht_utils.get_logger(__name__)
# pylint: disable=too-many-locals,unused-argument,too-many-instance-attributes,bare-except


def extract_mean_cov(feats, labels, index):
    indices = labels == index
    data = feats[indices]
    return torch.mean(data, axis=0), torch.cov(data.T).numpy()


def transform2torch(dataset_input):
    dataset = deepcopy(dataset_input)  # This looks bad!!!
    if isinstance(dataset.data, np.ndarray):
        dataset.targets = torch.Tensor(dataset.targets)
    elif isinstance(dataset.data, list):
        dataset.data = torch.Tensor(dataset.data)
        dataset.targets = torch.Tensor(dataset.targets)
    # Some dataset's label starts from 1 such as EMNIST, so we need to shift it
    if isinstance(dataset.targets, torch.Tensor) and min(dataset.targets) == 1:
        dataset.targets -= 1
    return dataset


def torchify_targets(dataset):
    if not isinstance(dataset.targets, torch.Tensor):
        dataset.targets = torch.Tensor(dataset.targets)
    if isinstance(dataset.targets, torch.Tensor) and min(dataset.targets) == 1:
        dataset.targets -= 1
    return dataset


def knn_data(dataset_input, path):
    # dataset = transform2torch(dataset_input)
    dataset = torchify_targets(dataset_input)
    gt_labels = dataset.targets
    knn_dict = torch.load(path)
    dataset.targets = knn_dict["labels"].int()  # .numpy()
    accuracy = (dataset.targets == gt_labels).sum() / dataset.targets.shape[0]
    assert abs(accuracy - knn_dict["train_ds_accuracy"]) < 0.01
    log.info(
        f"Loaded the KNN classified labels with accuracy <{accuracy}> with path {path}"
    )
    return dataset


def few_shot_data(dataset_input, n_shot, seed=1, nist_ds=True):
    dataset = transform2torch(dataset_input)
    max_label = int(max(dataset.targets))
    if "H5" in str(type(dataset)):  # Dataset is an hdf5 file
        few_shot_dataset = PyTorchDataset([dataset.data, dataset.targets])
    elif hasattr(dataset, "dataset") and "H5" in str(type(dataset.dataset)):
        # Dataset is a subset of an hdf5 file dataset
        few_shot_dataset = PyTorchDataset([dataset.data, dataset.targets])
    else:
        few_shot_dataset = deepcopy(dataset)
    few_shot_dataset.data = few_shot_dataset.data[: n_shot * (max_label + 1)]
    few_shot_dataset.targets = few_shot_dataset.targets[: n_shot * (max_label + 1)]
    torch.manual_seed(seed)
    for index in range(max_label + 1):
        all_indices = dataset.targets == index
        selected_indices = torch.randperm(all_indices.sum())[:n_shot]
        (
            few_shot_dataset.data[n_shot * index : n_shot * (index + 1)],
            few_shot_dataset.targets[n_shot * index : n_shot * (index + 1)],
        ) = (
            dataset.data[all_indices][selected_indices],
            dataset.targets[all_indices][selected_indices],
        )
    return few_shot_dataset


def repeated_few_shot_data(dataset_input, n_shot, seed=1):
    rep_ds = transform2torch(dataset_input)
    few_shot_ds = few_shot_data(rep_ds, n_shot, seed)
    tile = int(len(rep_ds) / n_shot)
    rep_ds.data = repeat(few_shot_ds.data, "b c h w -> (tile b) c h w", tile=tile)
    rep_ds.targets = repeat(few_shot_ds.targets, "b -> (tile b)", tile=tile)
    return rep_ds


def extract_class(dataset_input, index):
    dataset = transform2torch(dataset_input)
    indices = dataset.targets == index
    dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]
    return dataset


def squared_bures_wass(cov_s, cov_t, mean_s=None, mean_t=None):
    if mean_t is None:
        mean_t = torch.zeros(cov_s.shape[0])
    if mean_s is None:
        mean_s = torch.zeros(cov_s.shape[0])
    under_root = ln.sqrtm(cov_s) @ cov_t @ ln.sqrtm(cov_s)
    return torch.linalg.norm(mean_s - mean_t) ** 2 + np.trace(
        cov_s + cov_t - 2 * ln.sqrtm(under_root)
    )


def points_bures_wass(source, target):
    source_feat, source_label = get_feat_label(source)
    target_feat, target_label = get_feat_label(target)
    num_s_label, num_t_label = int(max(source_label + 1)), int(max(target_label + 1))
    w_distance_table = torch.zeros([num_s_label, num_t_label])
    for idx_s in range(num_s_label):
        for idx_t in range(num_t_label):
            mean_s, cov_s = extract_mean_cov(source_feat, source_label, idx_s)
            mean_t, cov_t = extract_mean_cov(target_feat, target_label, idx_t)
            w_distance_table[idx_s, idx_t] = squared_bures_wass(
                cov_s, cov_t, mean_s, mean_t
            )
    print("w_distance_table:", w_distance_table)
    return w_distance_table


def label_distance_exact_origin(source, target, label_dist_path, suffix):
    # TODO: if not found, compute the OTDD now.
    # Since here label distance is symmetric,
    # it's only saved in one file, so we need to try both ways.
    try:
        stat = torch.load(join(label_dist_path, f"{source}_{target}_{suffix}"))[source]
    except:
        stat = torch.load(join(label_dist_path, f"{target}_{source}_{suffix}"))[source]
    w_distance_table = stat["w2_matrix"]
    baseline_otdd = stat["otdd"]
    w2_matrix_min = stat["w2 min"]
    w_distance_table -= w_distance_table.min()
    print("w_distance_table:", w_distance_table)
    return w_distance_table, baseline_otdd, w2_matrix_min

