"""
python src/scripts/knn_vtab.py --ds_name "pets"

python src/scripts/knn_vtab.py --ds_name "OxfordIIITPet" --vtab_data_dir "/data/VTAB-mae-embeddings/" --num_neighbor 1

The goal of this file:
Given train800val200 dataset and train dataset,
it classify train dataset with train800val200 dataset.
No need to use cuda, no randomness is involved.
------ Inputs ------
MAE embedding data --> vtab_data_dir
------ Outputs ------
[Path to VTAB-mae-embeddings]/[DATASET]/train_knn/train_knn-mae-xx.hdf5
"""

import argparse
import os

import torch
from einops import rearrange

from src.callbacks.w2_callbacks import torchify_targets
from src.datamodules.datasets.hdf5_dataset import cat_and_dump_hdf5
from src.scripts.vtab_bary_projection import is_not_exists_makedir
from src.transfer_learning.data_utils import NAME_MAP
from src.transfer_learning.gen_geodesic import get_emb_mae_data
from src.utils.knn import knn

# pylint: disable=too-many-locals, missing-function-docstring


def main():
    parser = argparse.ArgumentParser(description="Solve KNN datasets for VTAB.")
    parser.add_argument(
        "--ds_name",
        type=str,
        default="DMLab",
        metavar="D",
        help="reference dataset or the test dataset to use",
    )
    parser.add_argument(
        "--vtab_data_dir",
        type=str,
        default="/home/jfan97/dpdata/datasets/masked_autoencoder/dropbox_file/shuffled_emb",
        help="path to the MAE embedding data",
    )
    parser.add_argument(
        "--num_neighbor", type=int, default=1, help="default: 1-nearest-neighbor"
    )
    parser.add_argument("--batch_size", type=int, default=1000, metavar="N")
    parser.add_argument(
        "--max_dump_size",
        type=int,
        default=10000,
        metavar="N",
        help="maximum number of data in each dumped hdf5 file",
    )
    parser.add_argument("--labeled_fold", type=str, default="train1000_seed0", metavar="S")
    parser.add_argument("--seed", type=int, default=0, metavar="S")

    cfg = parser.parse_args()
    pattern = "knn" + '_' + cfg.labeled_fold.split('-')[-1]
    print(pattern)
    knn_data_folder = os.path.join(cfg.vtab_data_dir, NAME_MAP[cfg.ds_name], pattern)
    is_not_exists_makedir(knn_data_folder)

    _, train_dl = get_emb_mae_data(
        cfg.vtab_data_dir,
        cfg.ds_name,
        fold="train",
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    fine_tune_dataset, _ = get_emb_mae_data(
        cfg.vtab_data_dir,
        cfg.ds_name,
        fold=cfg.labeled_fold,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    fine_tune_dataset = torchify_targets(fine_tune_dataset)

    # since fine_tune_dataset only has one hdf5 file, we can just load it into memory
    existing_feature = torch.from_numpy(fine_tune_dataset.archives[0]["X"][:])
    existing_labels = fine_tune_dataset.targets
    # debug using 100 samples
    # existing_feature = existing_feature[:10]
    # existing_labels = existing_labels[:10]

    file_knn_feat, file_knn_label = [], []
    nsamples = 0
    nfiles = 0
    for batch in train_dl:
        if nsamples + batch[0].shape[0] >= cfg.max_dump_size:
            # Dump to file, reset counters
            cat_and_dump_hdf5(
                file_knn_feat,
                file_knn_label,
                knn_data_folder,
                pattern,
                nfile=nfiles,
            )
            file_knn_feat, file_knn_label = [], []
            nsamples = 0
            nfiles += 1
        unclassified_feature, _ = batch

        # assign_index returns such a matrix
        # the second row gives the indices of the nearest neighbors
        # in [existing features] for the [unclassified features] index.
        # https://discuss.pytorch.org/t/explanation-of-torch-cluster-knn/148245
        print("Running knn")
        assign_index = knn(existing_feature, unclassified_feature, cfg.num_neighbor)
        all_knn_labels = rearrange(
            existing_labels[assign_index[1]], "(b n) -> b n", n=cfg.num_neighbor
        )
        # This mode is the most frequent label in [num_neighbor] results.
        batch_knn_labels = torch.mode(all_knn_labels, dim=1)[0]

        nsamples += unclassified_feature.shape[0]
        file_knn_feat.append(unclassified_feature)
        file_knn_label.append(batch_knn_labels)

    cat_and_dump_hdf5(
        file_knn_feat,
        file_knn_label,
        knn_data_folder,
        pattern,
        nfile=nfiles,
    )


if __name__ == "__main__":
    main()
