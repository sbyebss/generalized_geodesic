"""
This file calculates the transport metric between any two datasets.
Based on them, it also finds the best interpolation parameter.

------ Inputs ------
the OTDD label distances --> otdd_dir
== otdd_dir from bary_projection.py
reference MAE embedding data --> vtab_data_dir
== vtab_data_dir in bary_projection.py
the pushforward MAE embedding datasets --> pushforward_dataset_dir
== output_dir from bary_projection.py
------ Outputs ------
the output statistics (best interpolation parameter) --> output_dir
the folder to images of the projection dataset --> projection_dataset_dir
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datamodules.datasets.hdf5_dataset import H5Dataset, cat_and_dump_hdf5
from src.models.loss_zoo import label_cost, mse_loss
from src.models.non_mask_mae import prepare_model
from src.scripts.vtab_bary_projection import is_not_exists_makedir
from src.transfer_learning.gen_geodesic import get_best_interp_param, get_emb_mae_data
from src.transfer_learning.mix_transformation import ConcatDataset, gen_geodesic_mix_no_pf
from src.transfer_learning.train_nist_classifier import get_num_label
from src.utils import lht_utils
from src.viz.img import save_seperate_imgs

log = lht_utils.get_logger(__name__)

# pylint: disable=too-many-locals,line-too-long, too-many-statements,unused-variable, undefined-loop-variable, too-many-branches


def discrete_transport_metric(
    dataloader1,
    dataloader2,
    label_dist_matrix,
    device,
    coeff_feat=1.0,
    coeff_label=1.0,
):
    total_distance = 0.0
    label_dist_matrix = label_dist_matrix.to(device)
    for (feat1, labels1), (feat2, labels2) in zip(dataloader1, dataloader2):
        feat1 = feat1.to(device)
        labels1 = labels1.to(device)
        batch_size = feat1.shape[0]
        feat2 = feat2.to(device)[:batch_size]
        labels2 = labels2.to(device)[:batch_size]
        try:
            if len(labels1.shape) == 1:
                # dataloader1 can be the reference dataset, and it can be
                # with hard labels
                batch_label_cost = label_cost(label_dist_matrix, labels1, labels2)
            elif len(labels1.shape) == 2:
                batch_label_cost = (
                    ((labels1 @ label_dist_matrix) * labels2).sum(axis=1).mean()
                )
        except:
            breakpoint()
        total_distance += (
            coeff_feat * mse_loss(feat1, feat2) + coeff_label * batch_label_cost
        ) * feat1.shape[0]
    return total_distance.item() / len(dataloader1)


def main():
    parser = argparse.ArgumentParser(
        description="Given pushforward datsets, find the best interpolation parameter"
    )
    parser.add_argument(
        "--reference_ds_name",
        type=str,
        # default="Retinopathy",
        default="ImageNet",
        metavar="D",
        help="reference dataset or the test dataset to use (default: Retinopathy)",
    )
    parser.add_argument(
        "--reference_ds_fold",
        type=str,
        default="train_knn",  # train, train800val200, knntrain
        help="fold of reference dataset to use (default: train800val200)",
    )
    parser.add_argument(
        "--train_datasets_name",
        nargs="+",
        default=["sNORB-Azim", "DMLab", "Camelyon"],
        metavar="D",
        help="a list of train datasets to use",
    )
    parser.add_argument(
        "--otdd_dir",
        type=str,
        default="data/otdd/vtab",
        help="path to the OTDD label distances",
    )
    parser.add_argument(
        "--vtab_data_dir",
        type=str,
        default="/home/jfan97/dpdata/datasets/masked_autoencoder/dropbox_file/shuffled_emb",
        help="path to the MAE embedding data",
    )
    parser.add_argument(
        "--pushforward_dataset_dir",
        type=str,
        default="data/pushforward_datasets/vtab",
        help="path to save the pushforward MAE embedding datasets by bary_projection.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/otdd_transport_metric/vtab",
        help="path to the output statistics",
    )
    parser.add_argument(
        "--projection_dataset_dir",
        type=str,
        default="data/projection_dataset/vtab",
        help="path to save the images of the projection dataset",
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        default="uniform",
        help="Use uniform weight or optimal weight",
    )
    parser.add_argument("--batch_size", type=int, default=128, metavar="N")
    parser.add_argument(
        "--max_dump_size",
        type=int,
        default=10000,
        metavar="N",
        help="maximum number of data in each dumped hdf5 file",
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

    # ----------------- MAE arguments ----------------- #
    parser.add_argument(
        "--imagenet_mean",
        type=np.ndarray,
        default=np.array([0.485, 0.456, 0.406]),
        help="mean of images in ImageNet dataset",
    )
    parser.add_argument(
        "--imagenet_std",
        type=np.ndarray,
        default=np.array([0.229, 0.224, 0.225]),
        help="mean of images in ImageNet dataset",
    )
    parser.add_argument(
        "--mae_chkpt_dir",
        type=str,
        default="data/ckpts/mae_visualize_vit_large_ganloss.pth",
    )

    cfg = parser.parse_args()
    cfg.device = (
        f"cuda:{cfg.device}"
        if (torch.cuda.is_available() and not cfg.no_cuda)
        else "cpu"
    )
    assert (
        cfg.max_dump_size > cfg.batch_size
    ), "we need this when dumping projection embeddings"

    # Load the "mae_vit_large_patch16" model
    model_mae_gan = prepare_model(cfg.mae_chkpt_dir)
    model_mae_gan.to(cfg.device)
    model_mae_gan.eval()

    target_alias = "".join(ds + "_" for ds in cfg.train_datasets_name)
    num_train_dataset = len(cfg.train_datasets_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    otdd_stat_path = os.path.join(
        cfg.output_dir,
        f"from_{cfg.reference_ds_name}-knn_2_{target_alias}seed{cfg.seed}.pth",
    )
    # it should contain all distances (C_n^2) and the best interpolation parameter

    # load the reference and the train datasets
    # and create the dataloaders without shuffling
    print(cfg.vtab_data_dir, cfg.reference_ds_name, cfg.reference_ds_fold)
    reference_ds, reference_dl = get_emb_mae_data(
        cfg.vtab_data_dir,
        cfg.reference_ds_name,
        fold=cfg.reference_ds_fold,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    n_label_reference = get_num_label(cfg.reference_ds_name)

    train_ds_list = []
    train_dl_list = []
    n_label_train = []
    for train_ds_name in cfg.train_datasets_name:
        train_ds = H5Dataset(
            os.path.join(
                cfg.pushforward_dataset_dir,
                f"source_{cfg.reference_ds_name}-knn",
                f"target_{train_ds_name}",
            ),
            pattern=f"train-pf-seed{cfg.seed}-mae",
        )
        n_label_train.append(get_num_label(train_ds_name))
        # assert len(train_ds) == len(
        #     reference_ds
        # ), "When calculating the (2,Q)-dataset distance, the two datasets should have the same length"
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
        train_ds_list.append(train_ds)
        train_dl_list.append(train_dl)
    del train_ds, train_dl, train_ds_name

    if cfg.weight_type == "optimal":
        # --------------- Calculate the (2,\mu) dataset distance -----------------
        if os.path.exists(otdd_stat_path):
            otdd_stat = torch.load(otdd_stat_path)
            # The i entry in w2_between_internal_external is Wasserstein distance between (\nu,\mu_i)
            # The (i,j) entry in w2_matrix_internal is Wasserstein distance between (\mu_i,\mu_j)
            w2_between_internal_external, w2_matrix_internal, best_interpo_params = (
                otdd_stat["W(nu,mu_i)"],
                otdd_stat["W(mu_i,mu_j)"],
                otdd_stat["best_interpo_params"],
            )
        else:
            w2_matrix_internal = np.zeros([num_train_dataset, num_train_dataset])
            w2_between_internal_external = np.zeros(num_train_dataset)

            for idx_i in range(num_train_dataset):
                for idx_j in range(num_train_dataset):
                    if idx_i > idx_j:
                        continue
                    if idx_i == idx_j:
                        sorted_names = sorted(
                            [
                                cfg.reference_ds_name + "-knn",
                                cfg.train_datasets_name[idx_i] + "-full",
                            ]
                        )
                        label_dist_matrix = torch.load(
                            os.path.join(
                                cfg.otdd_dir,
                                f"{sorted_names[0]}_{sorted_names[1]}.pt",
                            )
                        )[cfg.reference_ds_name]["w2_matrix"][
                            :n_label_reference, -n_label_train[idx_i] :
                        ]
                        assert label_dist_matrix.shape == (
                            n_label_reference,
                            n_label_train[idx_i],
                        ), "The shape of the label distance matrix is not correct"

                        distance = discrete_transport_metric(
                            reference_dl,
                            train_dl_list[idx_i],
                            label_dist_matrix,
                            cfg.device,
                        )
                        w2_between_internal_external[idx_i] = distance
                        log.info(
                            f"transport metric between source and {idx_i}th train datasets = {distance}"
                        )
                    else:
                        sorted_names = sorted(
                            [
                                cfg.train_datasets_name[idx_i] + "-full",
                                cfg.train_datasets_name[idx_j] + "-full",
                            ]
                        )
                        label_dist_path = os.path.join(
                            cfg.otdd_dir,
                            f"{sorted_names[0]}_{sorted_names[1]}.pt",
                        )
                        label_dist_matrix = torch.load(label_dist_path)[
                            cfg.train_datasets_name[idx_i]
                        ]["w2_matrix"][: n_label_train[idx_i], -n_label_train[idx_j] :]
                        assert label_dist_matrix.shape == (
                            n_label_train[idx_i],
                            n_label_train[idx_j],
                        ), "The shape of the label distance matrix is not correct"

                        distance = discrete_transport_metric(
                            train_dl_list[idx_i],
                            train_dl_list[idx_j],
                            label_dist_matrix,
                            cfg.device,
                        )
                        w2_matrix_internal[idx_i, idx_j] = w2_matrix_internal[
                            idx_j, idx_i
                        ] = distance
                        log.info(
                            f"transport metric between {idx_i}th and {idx_j}th train datasets = {distance}"
                        )

            # --------- Solve the best parameter by quadratic programming ----------
            best_interpo_params = get_best_interp_param(
                w2_between_internal_external, w2_matrix_internal
            )
            log.info(
                f"Best interpolation parameter for seed{cfg.seed} is {list(best_interpo_params)}"
            )
            torch.save(
                {
                    "W(nu,mu_i)": w2_between_internal_external,
                    "W(mu_i,mu_j)": w2_matrix_internal,
                    "best_interpo_params": best_interpo_params,
                },
                otdd_stat_path,
            )
        interpo_param = best_interpo_params
    elif cfg.weight_type == "uniform":
        interpo_param = torch.ones(num_train_dataset) / num_train_dataset

    # ------ Generate synthetic dataset and decode to images -----
    ds_subdir = f"from_{cfg.reference_ds_name}-knn_2_{target_alias}seed{cfg.seed}"

    projection_dir = os.path.join(
        cfg.projection_dataset_dir, cfg.weight_type + "_weight", ds_subdir
    )
    projection_img_dir = os.path.join(projection_dir, "images")
    is_not_exists_makedir(projection_img_dir)

    pf_dl = torch.utils.data.DataLoader(
        ConcatDataset(train_ds_list),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    mix_ds_labels = torch.tensor([])
    cnt = 0
    file_mix_feat, file_mix_label = [], []
    nsamples = 0
    nfiles = 0
    pattern = f"train-proj-seed{cfg.seed}"
    for batch in pf_dl:
        if nsamples + batch[0][0].shape[0] >= cfg.max_dump_size:
            # Dump to file, reset counters
            cat_and_dump_hdf5(
                file_mix_feat,
                file_mix_label,
                projection_dir,
                pattern,
                nfile=nfiles,
            )
            file_mix_feat, file_mix_label = [], []
            nsamples = 0
            nfiles += 1

        mix_feat, mix_label = gen_geodesic_mix_no_pf(
            batch, interpo_param, n_label_train, cfg.device
        )
        file_mix_feat.append(mix_feat.detach().cpu())
        file_mix_label.append(mix_label.detach().cpu())
        mix_feat = mix_feat.reshape(-1, 197, 1024)
        nsamples += mix_feat.shape[0]

        with torch.no_grad():
            pred = model_mae_gan.forward_decoder(
                mix_feat
            )  # mix_feat should be [batch_size, 197, 1024]
        pred = model_mae_gan.unpatchify(pred).detach().cpu()

        save_seperate_imgs(
            pred * cfg.imagenet_std.reshape(1, -1, 1, 1)
            + cfg.imagenet_mean.reshape(1, -1, 1, 1),
            projection_img_dir,
            cnt,
        )
        cnt += pred.shape[0]
        mix_ds_labels = torch.concat([mix_ds_labels, mix_label.detach().cpu()])

    cat_and_dump_hdf5(
        file_mix_feat,
        file_mix_label,
        projection_dir,
        pattern,
        nfile=nfiles,
    )

    mix_ds_labels = mix_ds_labels.numpy()
    ds_labels_df = pd.DataFrame(mix_ds_labels)
    ds_labels_df.to_csv(os.path.join(projection_dir, "labels.csv"))


if __name__ == "__main__":
    main()
