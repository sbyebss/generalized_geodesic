import torch
from torchvision import transforms
from torchvision.datasets import EMNIST, KMNIST, MNIST, USPS, FashionMNIST

# from .mnist_m import MNISTM


def nist_dataset(dataset, data_path, img_size=32):
    if dataset == "MNIST":
        torch_dataset = MNIST
    elif dataset == "USPS":
        torch_dataset = USPS
    elif dataset == "FMNIST":
        torch_dataset = FashionMNIST
    elif dataset == "KMNIST":
        torch_dataset = KMNIST
    # elif dataset == "MNISTM":
    #     torch_dataset = MNISTM
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
