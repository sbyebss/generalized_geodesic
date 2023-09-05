# pylint: disable=line-too-long, invalid-name,abstract-method,too-many-arguments, too-many-ancestors, arguments-differ,too-many-instance-attributes,unused-import
"""
In this mixup script, we don't touch anything about MAE, everything is about imagefolder or torchvision.

"""
import os
import warnings
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl

# torch and lightning imports
import torch
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader, Subset

from src.scripts.fulltrain_vtab import ResNetClassifier, append_to_file, get_torchvision_dataset
from src.transfer_learning.mix_transformation import mixup
from src.transfer_learning.train_nist_classifier import get_num_label

warnings.filterwarnings("ignore")

def is_not_exists_makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class MixupClassifier(ResNetClassifier):
    def __init__(self, *argss, **kwargs):
        super().__init__(*argss, **kwargs)
        self.mixup_func = kwargs["mixup_func"]

    def train_dataloader(self):
        if isinstance(self.dataset_name, str):
            dataset_list = [self.dataset_name]
        elif isinstance(self.dataset_name, list):
            dataset_list = self.dataset_name

        img_train_list = [
            get_torchvision_dataset(
                train_ds_name,
                root=self.train_path,
                split="train",
                transforms=self.transforms,
            )
            for train_ds_name in dataset_list
        ]
        if len(img_train_list) == 1 and self.train_subset is not None:
            img_train_list = [Subset(img_train_list[0], torch.load(self.train_subset))]
        dl_list = [
            DataLoader(
                img_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )
            for img_train in img_train_list
        ]
        return dl_list

    def training_step(self, batch, _):
        if len(batch) == 1:
            x, y = batch[0]
        else:
            x, y = self.mixup_func(batch, device=self.device)
        preds = self(x)
        loss = self.criterion(preds, y)
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss


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
        default="Mixup",
        help="dataset interpolation method, only relevant if more than one training dataset (default: Mixup)",
    )
    parser.add_argument(
        "--reference_ds_name",
        type=str,
        default="OxfordIIITPet",
        metavar="D",
        help="reference dataset or the test dataset to use (default: Retinopathy)",
    )
    parser.add_argument(
        "--reference_ds_indices",
        type=str,
        default="/home/jfan97/Study_hard/general_monge_map/otdd_map/tests/test_indices.pt",
        help="(optional) If provided, training data from reference set will be subsampled with these indices",
    )
    parser.add_argument(
        "--train_ds_name",
        nargs="+",
        # default=["Flowers102"],
        default=["Caltech101", "DTD", "Flowers102"],
        metavar="D",
        help="a list of train datasets to use",
    )
    parser.add_argument(
        "--train_ds_path",
        type=Path,
        default="/home/jfan97/dpdata/datasets/torchvision",
        help="train set path. This is the path to vtab original images data",
    )
    parser.add_argument(
        "--val_ds_path",
        default="/home/jfan97/dpdata/datasets/imagenet/val",
        help="val set path.",
        type=Path,
    )
    parser.add_argument(
        "--test_ds_path",
        default="/home/jfan97/dpdata/datasets/torchvision",
        help="test set path.",
        type=Path,
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        default="pooling",
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
        default=3,
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
        default=0,
    )
    parser.add_argument(
        "--limit_test_batches",
        help="Number of steps to Run on Target.",
        type=float,
        default=1.0,  # correct
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
        default=True,
    )
    parser.add_argument(
        "-to",
        "--tune_fc_only",
        default=False,
        action="store_true",
        help="Tune only the final, fully connected layers.",
    )
    parser.add_argument(
        "--otdd_stat_path",
        type=str,
        default="logs/otdd_transport_metric/vtab",
        help="path to the best interpolation parameter statistics",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        help="Path to save version trained version checkpoint.",
        default="logs/reproduce/vtab",
    )
    parser.add_argument(
        "-g", "--gpus", help="Enables GPU acceleration.", type=int, default=1
    )
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    pprint(vars(args))

    n_test_class = get_num_label(args.reference_ds_name)

    # Interpolation regime
    num_train_classes = [get_num_label(ds) for ds in args.train_ds_name]
    n_train_class = sum(num_train_classes)
    target_alias = "".join(ds + "_" for ds in args.train_ds_name)
    sub_dir = f"from_{args.reference_ds_name}-knn_2_{target_alias}seed{args.seed}"
    if args.weight_type == "uniform":
        interpo_params = torch.ones(len(num_train_classes)) / len(num_train_classes)
    elif args.weight_type == "optimal":
        otdd_stat_path = os.path.join(
            args.otdd_stat_path,
            sub_dir + ".pth",
        )
        interpo_params = torch.load(otdd_stat_path)["best_interpo_params"]
    elif args.weight_type == "pooling":
        # We will randomly generate one-hot parameters later
        interpo_params = [None] * len(num_train_classes)
    else:
        interpo_params = torch.nn.functional.one_hot(
            torch.tensor([args.train_ds_name.index("Flowers102")]),
            len(num_train_classes),
        ).view(-1)

    mixup_func = partial(
        mixup,
        weights=interpo_params,
        num_target_classes=num_train_classes,
    )

    # # Instantiate Model
    model = MixupClassifier(
        num_train_class=n_train_class,
        num_test_class=n_test_class,
        resnet_version=args.model_version,
        dataset_name=args.train_ds_name,
        torchvision_dataset=True,
        train_path=args.train_ds_path,
        val_path=None,  # No valid during pretraining on source
        test_path=args.test_ds_path,
        optimizer=args.optimizer,
        lr=args.train_lr,
        batch_size=args.batch_size,
        pretrained=not args.from_scratch,
        train_hard_label=False,
        # mixup
        mixup_func=mixup_func,
    )
    trainer_args = {
        "gpus": args.gpus,
        "max_epochs": args.num_epochs_source,
        "reload_dataloaders_every_epoch": True,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "limit_test_batches": args.limit_test_batches,
    }

    ### Initial training on source dataset
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model)

    ### Fine-tuning on target dataset
    # Reset model to fine-tune on target dataset
    model.replace_fc_layer()
    if args.tune_fc_only:
        model.freeze_trunk()
    assert model.resnet_model.fc.out_features == n_test_class
    model.dataset_name = args.reference_ds_name
    model.train_subset = args.reference_ds_indices
    model.val_path = None
    model.train_hard_label = True
    model.lr = args.finetune_lr

    trainer_args = {
        "gpus": args.gpus,
        "max_epochs": args.num_epochs_target,
        "reload_dataloaders_every_epoch": True,
        "limit_train_batches": 1.0,
        "limit_val_batches": 0.0,
        "limit_test_batches": args.limit_test_batches,
    }

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model)
    trainer.test(model)

    test_accuracy = trainer.logged_metrics["avg_test_acc"]
    print(f"Test accuracy: {test_accuracy:.4f}")

    save_path = os.path.join(
        args.save_path,
        args.weight_type + "_weight",
        sub_dir,
        "transfer_learn",
        args.interpolation_method,
    )
    print(save_path)
    is_not_exists_makedir(save_path)
    torch.save(test_accuracy, os.path.join(save_path, "accuracy.pt"))

    # Save trained model
    save_path = os.path.join(save_path, "trained_model.ckpt")
    trainer.save_checkpoint(save_path)

    interpo_weights = dict(zip(args.train_ds_name, interpo_params))
    interpo_weights = ",".join([f"{d}:{w:.4f}" for d, w in interpo_weights.items()])

    tsvfile = os.path.join(args.save_path, "combined.tsv")
    results = {
        "reference": args.reference_ds_name,
        "training": ",".join(args.train_ds_name),
        "seed": args.seed,
        "weight_type": args.weight_type,
        "interpolation_method": "mixup",
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
