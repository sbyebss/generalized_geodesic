# pylint: disable=line-too-long, invalid-name,abstract-method,too-many-arguments, too-many-ancestors, arguments-differ,too-many-instance-attributes,too-many-locals,unused-import
"""
https://github.com/Stevellen/ResNet-Lightning/fork
python src/scripts/fulltrain_vtab.py --reference_ds_name "ImageNet" --train_ds_name "DMLab" "Camelyon" "sNORB-Azim" "ImageNet" --finetune_ds_path "/home/jfan97/dpdata/datasets/imagenet/train" --test_ds_path "/home/jfan97/dpdata/datasets/imagenet/val" --transfer_learning 0 -es 7 -et 3 --seed 1
python src/scripts/fulltrain_vtab.py --reference_ds_name "Retinopathy" --train_ds_name "DMLab" --finetune_ds_path "/data/VTAB/diabetic-retinopathy_btgraham-300_pytorch_imfolder/train" --test_ds_path "/data/VTAB/diabetic-retinopathy_btgraham-300_pytorch_imfolder/test" --transfer_learning 0 -es 7 -et 3 --seed 1 --limit_train_batches 2  --limit_test_batches 2

"""
import csv
import inspect
import os
import re
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl

# torch and lightning imports
import torch
import torch.utils.tensorboard as tb
import torchvision
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torchvision.datasets import ImageFolder

from src.datamodules.datasets.hdf5_dataset import H5Dataset
from src.datamodules.datasets.soft_label_imagefolder import SoftLabelImageFolder
from src.transfer_learning.train_nist_classifier import get_num_label


def is_not_exists_makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


NAME_MAP = {
    "Caltech101": "caltech101",
    "CIFAR100": "cifar100",
    "Camelyon": "patch_camelyon",
    "Clevr-Count": "clevr_count-all",
    "Clevr-Dist": "clevr_closest-object-detection",
    "DMLab": "dmlab",
    "dSpr-Ori": "dsprites_label-orientation",
    "dSpr-Loc": "dsprites_label-x-position",
    "DTD": "dtd",
    "EuroSAT": "eurosat",
    "Flowers102": "oxford_flowers102",
    "ImageNet": "imagenet1k",
    "KITTI": "kitti_closest-vehicle-distance",
    "OxfordIIITPet": "oxford_iiit_pet",
    "Resisc45": "resisc45",
    "Retinopathy": "diabetic-retinopathy_btgraham-300",
    "sNORB-Azim": "smallnorb_label-azimuth",
    "sNORB-Elev": "smallnorb_label-elevation",
    "SUN397": "sun397",
    "SVHN": "svhn_cropped",
}

warnings.filterwarnings("ignore")


def append_to_file(tsvfile, d):
    print(tsvfile)
    if not os.path.isdir(os.path.dirname(tsvfile)):
        os.makedirs(os.path.dirname(tsvfile), exist_ok=True)
    fields = d.keys()
    vals = ["None" if v is None else v for v in d.values()]
    header = not os.path.exists(tsvfile)
    print(header)
    with open(tsvfile, "a") as f:
        writer = csv.writer(f, delimiter="\t")
        if header:
            writer.writerow(fields)
        writer.writerow(vals)


# A light-weight version of the get_torchvviion_dataset in embed_mae
def get_torchvision_dataset(
    dataname,
    root="/data",
    split="train",
    transforms=None,
    download=True,
):
    assert hasattr(
        torchvision.datasets, dataname
    ), f"torchvision.datasets.{dataname} not found"
    DSET = getattr(torchvision.datasets, dataname)

    if transforms is None:
        transforms = torchvision.models.ResNet18_Weights.DEFAULT.transforms()

    if dataname.lower() == "imagenet":
        _root = os.path.join(root, "imagenet_pytorch")
        dset = {
            "train": DSET(root=_root, split="train", transform=transforms),
        }
        try:
            dset["test"] = DSET(root=_root, split="val", transform=transforms)
        except:
            dset["test"] = None
    else:  # downloadable datasets
        _root = os.path.join(root, "torchvision")
        if "train" in inspect.getfullargspec(DSET).args:
            dset = {
                "train": DSET(
                    root=_root, train=True, download=download, transform=transforms
                ),
                "test": DSET(
                    root=_root, train=False, download=download, transform=transforms
                ),
            }
        elif "split" in inspect.getfullargspec(DSET).args:
            dset = {}
            for _split in ["train", "trainval", "val", "test"]:
                try:
                    dset[_split] = DSET(
                        root=_root,
                        split=_split,
                        download=download,
                        transform=transforms,
                    )
                except:
                    print(f"Split {_split} not found in dataset {dataname}")
                    dset[_split] = None
                    continue
        else:
            print(
                "Warning: does not accept 'train' nor 'split' args. Will use only split as train"
            )
            dset = {
                "train": DSET(root=_root, download=download, transform=transforms),
                "val": None,
                "test": None,
            }

    if dset["train"] is None and dset["trainval"] is not None:
        print(
            f"WARNING: {dataname} does not have train split. Will use trainval instead."
        )
        dset["train"] = dset["trainval"]

    if dset[split] is None:
        print(f"Warning split {split} not found in {dset.keys()}")
    # else:
    #     print(split,len(dset[split]))
    return dset[
        split
    ]  # TODO: not very efficient, now that we take split arg just get the required split


class ResNetClassifier(pl.LightningModule):
    def __init__(
        self,
        num_train_class,
        num_test_class,
        resnet_version,
        train_path,
        torchvision_dataset=False,
        dataset_name=None,
        train_subset=None,
        val_path=None,
        test_path=None,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        pretrained=True,
        tune_fc_only=False,
        train_hard_label=False,
        train_dataset_type="ImageFolder",
        num_workers=20,
        **kwargs,
    ):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        self.num_train_class = num_train_class
        self.num_test_class = num_test_class
        self.train_path = train_path
        self.torchvision_dataset = torchvision_dataset
        self.dataset_name = dataset_name
        self.val_path = val_path
        self.test_path = test_path
        self.lr = lr
        self.batch_size = batch_size
        self.train_hard_label = train_hard_label
        self.train_dataset_type = train_dataset_type
        self.num_workers = num_workers
        self.pretrained = pretrained
        self.train_subset = train_subset

        optimizers = {"adam": Adam, "sgd": SGD}
        self._optname = optimizer
        self.optimizer = optimizers[optimizer]
        # instantiate loss criterion
        self.criterion = nn.CrossEntropyLoss()
        # Using a pretrained ResNet backbone
        weights = "IMAGENET1K_V1" if self.pretrained else None
        self.resnet_model = resnets[resnet_version](weights=weights)
        # Use standard transforms (they are identical for all ResNet models)
        self.transforms = models.ResNet50_Weights.DEFAULT.transforms()

        # Replace old FC layer with Identity so we can train our own
        self.linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(self.linear_size, self.num_train_class)

        if tune_fc_only:  # option to only tune the fully-connected layers
            self.freeze_trunk()

    def replace_fc_layer(self):
        self.resnet_model.fc = nn.Linear(self.linear_size, self.num_test_class)

    def freeze_trunk(self):
        for child in list(self.resnet_model.children())[:-1]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        print(f"Configure optimizers: {self.lr}")
        if self._optname == "SGD":
            opt = self.optimizer(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            opt = self.optimizer(self.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self):
        if self.train_hard_label and self.torchvision_dataset:
            img_train = get_torchvision_dataset(
                self.dataset_name,
                root="/data/",
                split="train",
                transforms=self.transforms,
            )
        elif self.train_hard_label and self.train_dataset_type.lower() == "imagefolder":
            img_train = ImageFolder(self.train_path, transform=self.transforms)
        elif self.train_hard_label and self.train_dataset_type.lower() == "hdf5":
            img_train = H5Dataset(self.train_path, pattern="train")
        elif (
            self.train_hard_label and self.train_dataset_type.lower() == "tensordataset"
        ):
            feat_data = torch.load(self.train_path + "/X.pt")
            label_data = torch.load(self.train_path + "/Y.pt")
            img_train = torch.utils.data.TensorDataset(feat_data, label_data)
        else:
            img_train = SoftLabelImageFolder(self.train_path, transform=self.transforms)
        if self.train_subset is not None:
            img_train = Subset(img_train, torch.load(self.train_subset))
        return DataLoader(
            img_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def training_step(self, batch, _):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        # Accuracy for soft label doesn't make sense
        # So we don't calculate training accuracy
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def val_dataloader(self):
        if self.trainer.limit_val_batches > 0 and self.torchvision_dataset:
            img_val = get_torchvision_dataset(
                self.dataset_name,
                root="/data/",
                split="val",
                transforms=self.transforms,
            )
        else:
            img_val = (
                ImageFolder(self.val_path, transform=self.transforms)
                if self.val_path is not None
                else None
            )
        if img_val is None:
            self.trainer.limit_val_batches = 0  # no validation
            return None
        else:
            return DataLoader(
                img_val, batch_size=self.batch_size, shuffle=False, num_workers=16
            )

    def test_dataloader(self):
        if self.torchvision_dataset:
            img_test = get_torchvision_dataset(
                self.dataset_name,
                root=self.test_path,
                split="test",
                transforms=self.transforms,
            )
        else:
            img_test = (
                ImageFolder(self.test_path, transform=self.transforms)
                if self.test_path is not None
                else None
            )
        if img_test is None:
            self.trainer.limit_test_batches = 0  # no test
            return None
        else:
            return DataLoader(
                img_test, batch_size=self.batch_size, shuffle=False, num_workers=16
            )

    def validation_step(self, batch, _):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = (y == torch.argmax(preds, 1)).type(torch.FloatTensor).mean()
        # perform logging
        self.log(
            "val_loss", loss, on_step=True, prog_bar=False, logger=True, sync_dist=True
        )
        self.log(
            "val_acc", acc, on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        return acc * x.shape[0]

    def test_step(self, batch, _):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = (y == torch.argmax(preds, 1)).type(torch.FloatTensor).mean()
        # perform logging
        self.log(
            "test_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            "test_acc", acc, on_step=True, prog_bar=True, logger=True, sync_dist=True
        )
        return acc * x.shape[0]

    def test_epoch_end(self, outputs):
        avg_acc = torch.stack(outputs).sum() / len(self.test_dataloader().dataset)
        self.log("avg_test_acc", avg_acc, logger=True)
        return avg_acc


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument(
        "--model_version",
        type=int,
        default=18,
        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
    )

    parser.add_argument(
        "--interpolation_method",
        type=str,
        default="OTDD_map",
        help="dataset interpolation method, only relevant if more than one training dataset (default: OTDD_map)",
    )
    parser.add_argument(
        "--reference_ds_name",
        type=str,
        default="ImageNet",
        metavar="D",
        help="reference dataset or the test dataset to use (default: Retinopathy)",
    )
    parser.add_argument(
        "--reference_ds_indices",
        type=str,
        default=None,
        help="(optional) If provided, training data from reference set will be subsampled with these indices",
    )

    parser.add_argument(
        "--train_ds_name",
        nargs="+",
        default=["sNORB-Azim", "DMLab", "Camelyon", "ImageNet"],
        metavar="D",
        help="a list of train datasets to use",
    )
    parser.add_argument(
        "--train_ds_path",
        type=Path,
        default="data/projection_dataset/vtab",
        help="Path to training data folder. This should be the interpolated dataset path",
    )
    parser.add_argument(
        "--finetune_ds_path",
        type=Path,
        default="/home/jfan97/dpdata/datasets/imagenet/train",
        help="Fine tine set path.",
    )
    parser.add_argument(
        "--val_ds_path",
        default="/home/jfan97/dpdata/datasets/imagenet/val",
        help="test set path.",
        type=Path,
    )
    parser.add_argument(
        "--test_ds_path",
        default="/data/",
        help="test set path.",
        type=Path,
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        default="optimal",
        help="Use uniform weight or optimal weight",
    )
    # Optional arguments
    parser.add_argument(
        "-es",
        "--num_epochs_source",
        help="Number of Epochs to Run on Source.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-et",
        "--num_epochs_target",
        help="Number of Epochs to Run on Target.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--limit_train_batches",
        help="Number of steps to Run on Source.",
        type=float,
        default=1.0,  # correct
    )
    parser.add_argument(
        "--limit_val_batches",
        help="Number of steps to Run on Source.",
        type=float,
        default=1.0,  # correct
    )
    parser.add_argument(
        "--limit_test_batches",
        help="Number of steps to Run on Target.",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "-o",
        "--optimizer",
        help="PyTorch optimizer to use. Defaults to adam.",
        default="adam",
    )
    parser.add_argument(
        "-tlr",
        "--train_lr",
        help="Adjust learning rate of optimizer for training.",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "-ftlr",
        "--finetune_lr",
        help="Adjust learning rate of optimizer for fine-tuning.",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        help="Manually determine batch size. Defaults to 64.",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--from_scratch",
        help="Determine whether to initialize net from scratch insteaf of pretrained.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-to",
        "--tune_fc_only",
        default=False,
        action="store_true",
        help="Tune only the final, fully connected layers.",
    )
    parser.add_argument(
        "-tl",
        "--transfer_learning",
        default=False,
        action="store_true",
        help="Tune only the final, fully connected layers.",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        help="Path to save version trained version checkpoint.",
        default="logs/reproduce/vtab",
    )
    parser.add_argument(
        "-g", "--gpus", help="Enables GPU acceleration.", type=int, default=0
    )
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    pprint(vars(args))

    # TODO: add seed_everything here

    n_test_class = get_num_label(args.reference_ds_name)

    if len(args.train_ds_name) > 1:
        # Interpolation regime
        num_train_classes = [get_num_label(ds) for ds in args.train_ds_name]
        n_train_class = sum(num_train_classes)
        target_alias = "".join(ds + "_" for ds in args.train_ds_name)
        sub_dir = f"from_{args.reference_ds_name}-knn_2_{target_alias}seed{args.seed}"

        if args.interpolation_method == "OTDD_map":
            initial_train_ds_path = os.path.join(
                args.train_ds_path, args.weight_type + "_weight", sub_dir
            )
            hard_labels = False
            dataset_type = "None"

        elif args.interpolation_ds_type == "knn":
            initial_train_ds_path = os.path.join(args.train_ds_path, sub_dir)
            hard_labels = True
            dataset_type = "TensorDataset"
            args.weight_type = None

        torchvision_dataset = False
        initial_train_name = None

    elif args.train_ds_name[0] == "NONE":
        # No pretraining baseline
        # assert args.transfer_learning == False
        args.transfer_learning = False
        args.pretrain_epochs = 0
        initial_train_name = None
        args.interpolation_method = None
        args.weight_type = None
        hard_labels = False
        initial_train_ds_path = None
        dataset_type = None
        torchvision_dataset = False
        n_train_class = 2  # Dummy
        target_alias = args.train_ds_name[0]
        sub_dir = f"from_{args.reference_ds_name}-knn_2_{target_alias}seed{args.seed}"
    else:
        # No interpolation regime
        args.interpolation_method = None
        args.weight_type = None
        initial_train_ds_path = None
        n_train_class = get_num_label(args.train_ds_name[0])
        # DEBUGL
        # #### If we want to train directly on embeddings:
        # assert os.path.realpath(args.train_ds_path) == os.path.realpath('/data/VTAB-mae-embeddings')
        # sub_dir = NAME_MAP[args.train_ds_name[0]] #+ '_pytorch_imfolder/train'
        # initial_train_ds_path = os.path.join(args.train_ds_path, sub_dir)
        # hard_labels = True
        # dataset_type = 'hdf5'
        # If we want to train on original image datasets in ImageFolder format:
        # assert os.path.realpath(args.train_ds_path) == os.path.realpath("/data/VTAB/")
        # initial_train_ds_path = os.path.join(
        #     args.train_ds_path,
        #     NAME_MAP[args.train_ds_name[0]] + "_pytorch_imfolder/train",
        # )
        # target_alias = args.train_ds_name[0]
        # sub_dir = f"from_{args.reference_ds_name}-knn_2_{target_alias}seed{args.seed}"
        # hard_labels = True
        # dataset_type = "ImageFolder"
        # If we want to train on original image datasets in torchvision native format:
        # assert os.path.realpath(args.train_ds_path) == os.path.realpath(
        #     "/data/torchvision/"
        # )
        target_alias = args.train_ds_name[0]
        sub_dir = f"from_{args.reference_ds_name}-knn_2_{target_alias}seed{args.seed}"
        hard_labels = True
        dataset_type = "torchvision"
        torchvision_dataset = True
        initial_train_name = args.train_ds_name[0]

    # if args.transfer_learning is not True:
    #     n_train_class = 1000
    # # Instantiate Model
    model = ResNetClassifier(
        num_train_class=n_train_class,
        num_test_class=n_test_class,
        resnet_version=args.model_version,
        dataset_name=initial_train_name,
        train_path=initial_train_ds_path,
        torchvision_dataset=torchvision_dataset,
        val_path=None,  # No valid during pretraining on source
        test_path=args.test_ds_path,
        optimizer=args.optimizer,
        lr=args.train_lr,
        batch_size=args.batch_size,
        pretrained=not args.from_scratch,
        train_hard_label=hard_labels,
        train_dataset_type=dataset_type,
    )
    trainer_args = {
        "accelerator": "gpu",
        "devices": args.gpus,
        "max_epochs": args.num_epochs_source,
        "reload_dataloaders_every_epoch": True,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "limit_test_batches": args.limit_test_batches,
    }

    if args.transfer_learning:
        # Initial training on source dataset
        trainer = pl.Trainer(**trainer_args)
        trainer.fit(model)
        if args.interpolation_method is None:
            save_path = os.path.join(
                args.save_path,
                sub_dir,
                "transfer_learn",
                f"no_interpolation_{args.train_ds_name[0]}",
            )
        else:
            save_path = os.path.join(
                args.save_path,
                sub_dir,
                args.weight_type + "_weight",
                "transfer_learn",
                args.interpolation_method,
            )
        print(save_path)
        trainer_args["max_epochs"] = args.num_epochs_target
    else:
        save_path = os.path.join(args.save_path, sub_dir, "non_transfer_learn")
        trainer_args["max_epochs"] = args.num_epochs_target  # + args.num_epochs_source

    # Fine-tuning on target dataset

    # Reset model to fine-tune on target dataset
    model.replace_fc_layer()
    if args.tune_fc_only:
        model.freeze_trunk()
    assert model.resnet_model.fc.out_features == n_test_class
    model.torchvision_dataset = True
    model.dataset_name = args.reference_ds_name
    model.train_subset = args.reference_ds_indices

    model.train_path = None  # args.finetune_ds_path
    model.val_path = None  # args.val_ds_path
    # model.test_path  = None
    model.train_hard_label = True
    # model.train_dataset_type = "torchvision"
    model.lr = args.finetune_lr

    trainer_args["limit_train_batches"] = 1.0  # NEver limit fine-tuning train
    trainer_args["devices"] = 1
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model)
    trainer.test(model)

    test_accuracy = trainer.logged_metrics["avg_test_acc"]
    is_not_exists_makedir(save_path)
    torch.save(test_accuracy, os.path.join(save_path, "accuracy.pt"))

    # Save trained model
    trainer.save_checkpoint(os.path.join(save_path, "trained_model.ckpt"))

    tsvfile = os.path.join(args.save_path, "combined.tsv")

    if args.interpolation_method is not None and args.weight_type == "optimal":
        weights_file = os.path.join("logs/otdd_transport_metric/vtab", sub_dir + ".pth")
        interpo_weights = dict(
            zip(args.train_ds_name, torch.load(weights_file)["best_interpo_params"])
        )
        interpo_weights = ",".join([f"{d}:{w:.4f}" for d, w in interpo_weights.items()])
    else:
        interpo_weights = None

    results = {
        "reference": args.reference_ds_name,
        "training": ",".join(args.train_ds_name),
        "seed": args.seed,
        "weight_type": args.weight_type,
        "interpolation_method": args.interpolation_method,
        "from_scratch": args.from_scratch,
        "pretrain_epochs": args.num_epochs_source,
        "finetune_epochs": args.num_epochs_target,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "pretrain_lr": args.train_lr,
        "finetune_lr": args.finetune_lr,
        "test_accuracy": test_accuracy,
        "interpolation_weights": interpo_weights,
        # 'train_loss': trainer.logged_metrics['train_loss'],
    }

    append_to_file(tsvfile, results)
