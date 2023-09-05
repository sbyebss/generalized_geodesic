"""
Given few shot labeled data, generate pseudo labels by 1-KNN.
"""
import os

import hydra
import numpy as np
import omegaconf
import torch
from einops import rearrange

from src.callbacks.w2_callbacks import few_shot_data
from src.transfer_learning.data_utils import get_train_test_dataset
from src.utils.knn import knn

# pylint: disable=no-value-for-parameter,line-too-long,too-many-locals


@hydra.main(config_path="../../configs/scripts", config_name="knn_dataset")
def knn_dataset(cfg: omegaconf.DictConfig):
    if not os.path.exists(cfg.knn_data_folder):
        os.makedirs(cfg.knn_data_folder)

    train_dataset, test_dataset = get_train_test_dataset(cfg)
    fine_tune_dataset = few_shot_data(train_dataset, cfg.num_shot, cfg.seed)
    existing_feature = fine_tune_dataset.data
    existing_labels = fine_tune_dataset.targets

    accuracy = [0, 0]
    for idx_ds, unclassified_ds in enumerate([train_dataset, test_dataset]):
        assert len(existing_feature) < len(unclassified_ds)
        gt_labels = unclassified_ds.targets
        knn_labels = torch.tensor([])

        batch_size = len(unclassified_ds)
        # batch_size = 100
        for idx in range(int(len(unclassified_ds) / batch_size)):
            unclassified_feature = unclassified_ds.data[
                idx * batch_size : (idx + 1) * batch_size
            ]
            existing_feature = rearrange(existing_feature, "b ... -> b (...)")
            unclassified_feature = rearrange(unclassified_feature, "b ... -> b (...)")

            if isinstance(existing_feature, np.ndarray):
                existing_feature = torch.from_numpy(existing_feature)
            if isinstance(unclassified_feature, np.ndarray):
                unclassified_feature = torch.from_numpy(unclassified_feature)

            # assign_index returns such a matrix
            # the second row gives the indices of the nearest neighbors in [existing features] for the [unclassified features] index.
            # https://discuss.pytorch.org/t/explanation-of-torch-cluster-knn/148245
            assign_index = knn(existing_feature, unclassified_feature, cfg.num_neighbor)
            all_knn_labels = rearrange(
                existing_labels[assign_index[1]], "(b n) -> b n", n=cfg.num_neighbor
            )
            # This mode is the most frequent label in [num_neighbor] results.
            batch_knn_labels = torch.mode(all_knn_labels, dim=1)[0]
            knn_labels = torch.concat([knn_labels, batch_knn_labels])
            batch_gt_labels = gt_labels[idx * batch_size : (idx + 1) * batch_size]
            if cfg.verbose:
                print(
                    f"Batch accuracy: {torch.sum(batch_knn_labels==batch_gt_labels)/len(batch_gt_labels)}"
                )

        accuracy[idx_ds] = (knn_labels == gt_labels).sum() / knn_labels.shape[0]
        ds_type = "train" if idx_ds == 0 else "test"
        print(
            f"Accuracy on {ds_type} dataset of {cfg.dataset} is ",
            accuracy[idx_ds].item(),
        )
        if idx_ds == 0:
            train_knn_labels = knn_labels

    torch.save(
        {
            "labels": train_knn_labels,
            "train_ds_accuracy": accuracy[0],
            "test_ds_accuracy": accuracy[1],
        },
        cfg.knn_data_path,
    )


if __name__ == "__main__":
    knn_dataset()
