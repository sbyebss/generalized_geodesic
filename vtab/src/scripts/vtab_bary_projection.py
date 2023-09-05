"""
The goal of this file:
Given one reference dataset and several training datasets,
it gets the pushforward datasets by barycentric projection and dump them to disk.
The requirement is that the reference dataset and the training datasets have the same order of data.
And it also dumps the label-to-label distance.
------ Inputs ------
MAE embedding data --> vtab_data_dir
------ Outputs ------
the OTDD label distances --> otdd_dir
the pushforward MAE embedding datasets --> output_dir
"""
# pylint: disable=too-many-locals,undefined-loop-variable, too-many-statements, protected-access,line-too-long

import argparse
import os

import torch

from src.datamodules.datasets.hdf5_dataset import cat_and_dump_hdf5
from src.datamodules.datasets.small_scale_image_dataset import PyTorchDataset
from src.otdd import launch_logger
from src.otdd.pytorch.distance import DatasetDistance
from src.transfer_learning.gen_geodesic import get_emb_mae_data
from src.transfer_learning.mix_transformation import barycentric_projection
from src.transfer_learning.train_nist_classifier import get_num_label

logger = launch_logger("info")


def is_not_exists_makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def solve_label_distance(ds1, ds2, ds1_name, ds2_name, device, save_dir, ds1_type):
    sorted_names = sorted([ds1_name + "-" + ds1_type, ds2_name + "-full"])
    label_dist_path = os.path.join(save_dir, f"{sorted_names[0]}_{sorted_names[1]}.pt")

    if os.path.exists(label_dist_path):
        print(
            "Label-to-label distances between {} and {} already exist. Loading from disk".format(
                ds1_name, ds2_name
            )
        )
        label_distances = torch.load(label_dist_path)[ds1_name]["w2_matrix"]
    else:
        dist = DatasetDistance(
            ds1,
            ds2,
            debiased_loss=False,
            batchified="both",
            inner_ot_method="exact",
            inner_ot_debiased=True,  # This changes less compared to debiased_loss.
            maxbatch=1500,
            minbatch=min(128, torch.unique(ds1.targets, return_counts=True)[1].min()),
            p=2,
            inner_ot_entreg=1e-1,
            entreg=1e-1,
            min_labelcount=1,  # Jiaojiao TBD, the default is 2.
            device=device,
            nworkers_dists=16,
            nworkers_stats=8,
            verbose=2,
        )
        label_distances = dist._get_label_distances()
        del dist
        # Output 2: the label to label distance matrix
        torch.save(
            {
                ds1_name: {"w2_matrix": label_distances},
                ds2_name: {"w2_matrix": label_distances.T},
            },
            label_dist_path,
        )
    return label_distances


def main():
    parser = argparse.ArgumentParser(
        description="Solve pushforward datasets by barycentric projection."
    )
    parser.add_argument(
        "--reference_ds_name",
        type=str,
        default="Retinopathy",
        # default="Camelyon",
        metavar="D",
        help="reference dataset or the test dataset to use (default: Retinopathy)",
    )
    parser.add_argument(
        "--reference_ds_fold",
        type=str,
        default="train_knn",  # train, train800val200, knntrain
        help="fold of reference dataset to use (default: train_knn)",
    )
    parser.add_argument(
        "--train_datasets_name",
        nargs="+",
        default=["sNORB-Azim", "DMLab", "Camelyon"],
        # default=["DMLab", "Camelyon"],
        metavar="D",
        help="a list of train datasets to use",
    )
    parser.add_argument(
        "--vtab_data_dir",
        type=str,
        default="/home/jfan97/dpdata/datasets/masked_autoencoder/dropbox_file/shuffled_emb",
        help="path to the MAE embedding data",
    )
    parser.add_argument(
        "--otdd_dir",
        type=str,
        default="data/otdd/vtab",
        help="path to save the OTDD label distances",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/pushforward_datasets/vtab",
        help="path to save the pushforward MAE embedding datasets",
    )
    parser.add_argument("--batch_size", type=int, default=5000, metavar="N")
    parser.add_argument(
        "--max_dump_size",
        type=int,
        default=10000,
        metavar="N",
        help="maximum number of data in each dumped hdf5 file",
    )
    parser.add_argument(
        "--pf_ds_size",
        default="max_train_ds_size",
        help="size of the pushforward dataset: either ref_ds_size or max_train_ds_size",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--device", type=int, default=2, metavar="S", help="cuda device (default: 2)"
    )

    cfg = parser.parse_args()
    cfg.device = (
        f"cuda:{cfg.device}"
        if (torch.cuda.is_available() and not cfg.no_cuda)
        else "cpu"
    )
    torch.manual_seed(cfg.seed)
    assert (
        cfg.max_dump_size > cfg.batch_size
    ), "we need this when dumping projection embeddings"
    is_not_exists_makedir(cfg.output_dir)

    # ----------------- Get reference dataset and training datasets ----------------- #
    reference_ds, reference_dl = get_emb_mae_data(
        cfg.vtab_data_dir,
        cfg.reference_ds_name,
        fold=cfg.reference_ds_fold,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    train_ds_list = []
    for train_ds_name in cfg.train_datasets_name:
        train_ds, train_dl = get_emb_mae_data(
            cfg.vtab_data_dir,
            train_ds_name,
            fold="train",
            batch_size=cfg.batch_size,
            shuffle_ahead=False,
        )
        train_ds_list.append(train_ds)

    del train_ds, train_dl, train_ds_name
    is_not_exists_makedir(cfg.otdd_dir)

    assert cfg.pf_ds_size in [
        "ref_ds_size",
        "max_train_ds_size",
    ], "pf_ds_size should be either ref_ds_size or max_train_ds_size"
    if cfg.pf_ds_size == "ref_ds_size":
        ref_epoch = 1
    elif cfg.pf_ds_size == "max_train_ds_size":
        # We set the pushforward dataset size to be the same as maximum of training datasets
        max_train_ds_size = max([len(train_ds) for train_ds in train_ds_list])
        ref_epoch = int(max_train_ds_size / len(reference_ds))

    # ----------------- Pushforward through the barycentric projection ----------------- #
    for idx, train_ds_name in enumerate(cfg.train_datasets_name):
        # Firstly solve the label-to-label distance
        train_ds = train_ds_list[idx]
        # * the first place using DatasetDistance
        print(f"Solving label distance  {cfg.reference_ds_name}<->{train_ds_name}")
        label_distances = solve_label_distance(
            reference_ds,
            train_ds,
            cfg.reference_ds_name,
            train_ds_name,
            cfg.device,
            cfg.otdd_dir,
            ds1_type="knn",
        )
        print("Done solving label distance.")

        num_target_class = get_num_label(train_ds_name)

        pf_dist_path = os.path.join(
            cfg.output_dir,
            f"source_{cfg.reference_ds_name}-knn",
            f"target_{train_ds_name}",
        )
        is_not_exists_makedir(pf_dist_path)

        # breakpoint()

        # Output 3: the pushforward embedding datasets
        # Create two empty tensors to store the pushforward features and labels
        file_pf_feat, file_pf_label = [], []
        nsamples = 0
        nfiles = 0
        for _ in range(ref_epoch):
            for batch_features, batch_labels in reference_dl:
                if nsamples + batch_features.shape[0] >= cfg.max_dump_size:
                    # Dump to file, reset counters
                    cat_and_dump_hdf5(
                        file_pf_feat,
                        file_pf_label,
                        pf_dist_path,
                        f"train-pf-seed{cfg.seed}",
                        nfile=nfiles,
                    )
                    file_pf_feat, file_pf_label = [], []
                    nsamples = 0
                    nfiles += 1

                # source batch is following a fixed order
                source_ds = PyTorchDataset([batch_features, batch_labels])
                # target batch is a random sub-sampling
                perm = torch.randperm(len(train_ds)).tolist()
                target_ds = torch.utils.data.Subset(train_ds, perm[: cfg.batch_size])
                # * the second place using DatasetDistance
                print(
                    f"Computing barycentric projection with datasets of size: {len(source_ds)}, {len(target_ds)}"
                )
                # TODO: label_distances currently have 0 for self distances. Check that barycentri_project accounts for this.
                pf_feat, pf_label = barycentric_projection(
                    source_ds,
                    target_ds,
                    batch_features.shape,
                    num_target_class,
                    cfg.device,
                    λ_y=1.0,
                    precomputed_label_dist=label_distances,
                )
                # Can you use subset dataset in OTDD??  Yes, you can, but only with one nested Subset.
                pf_feat = pf_feat.detach().cpu()
                pf_label = pf_label.detach().cpu()
                nsamples += pf_feat.shape[0]
                file_pf_feat.append(pf_feat)
                file_pf_label.append(pf_label)

        cat_and_dump_hdf5(
            file_pf_feat,
            file_pf_label,
            pf_dist_path,
            f"train-pf-seed{cfg.seed}",
            nfile=nfiles,
        )

    for idx_i, train_ds1_name in enumerate(cfg.train_datasets_name):
        for idx_j, train_ds2_name in enumerate(cfg.train_datasets_name):
            if idx_i != idx_j:
                train_ds_1 = train_ds_list[idx_i]
                train_ds_2 = train_ds_list[idx_j]
                print(f"Solving label distance  {train_ds1_name}<->{train_ds2_name}")
                # breakpoint()
                # * the third place using DatasetDistance
                solve_label_distance(
                    train_ds_1,
                    train_ds_2,
                    train_ds1_name,
                    train_ds2_name,
                    cfg.device,
                    cfg.otdd_dir,
                    ds1_type="full",
                )


if __name__ == "__main__":
    main()
