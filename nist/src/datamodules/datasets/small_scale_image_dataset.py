import os

import torch
from jammy.utils.git import git_rootdir
from torchvision import transforms
from torchvision.datasets import CIFAR10, EMNIST, KMNIST, MNIST, USPS, FashionMNIST

from .mnist_m import MNISTM

__all__ = ["get_img_dataset"]


def nist_dataset(dataset, data_path, img_size=32):
    if dataset == "MNIST":
        torch_dataset = MNIST
    elif dataset == "USPS":
        torch_dataset = USPS
    elif dataset == "FMNIST":
        torch_dataset = FashionMNIST
    elif dataset == "KMNIST":
        torch_dataset = KMNIST
    elif dataset == "MNISTM":
        torch_dataset = MNISTM
    elif dataset == "EMNIST":
        torch_dataset = EMNIST
        train = torch_dataset(
            data_path,
            split="letters",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(img_size),
                    lambda img: transforms.functional.rotate(img, -90),
                    transforms.functional.hflip,
                    transforms.ToTensor(),
                ]
            ),
        )
        test = torch_dataset(
            data_path,
            split="letters",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(img_size),
                    lambda img: transforms.functional.rotate(img, -90),
                    transforms.functional.hflip,
                    transforms.ToTensor(),
                ]
            ),
        )
        return train, test
    train = torch_dataset(
        data_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        ),
    )
    test = torch_dataset(
        data_path,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        ),
    )
    return train, test


def init_data_config(config):
    if "path" not in config:
        config.path = git_rootdir("data")


def get_img_dataset(config):  # pylint: disable=too-many-branches
    init_data_config(config)
    if config.dataset in ["MNIST", "USPS", "FMNIST", "KMNIST", "EMNIST", "MNISTM"]:
        return nist_dataset(config.dataset, config.path, config.image_size)
    if config.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.image_size), transforms.ToTensor()]
        )

    if config.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(config.path, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(config.path, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )
    return dataset, test_dataset


class PyTorchDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None, target_transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.data, self.targets = tensors
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)

        y = self.targets[index]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return self.data.size(0)
