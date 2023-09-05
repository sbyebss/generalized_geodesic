import os

import pandas
import torch
from PIL import Image
from torch.utils import data

# pylint: disable=no-self-use


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transform") and self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transforms: ")
        if hasattr(self, "target_transform") and self.target_transform is not None:
            body += self._format_transform_repr(
                self.target_transform, "Target transforms: "
            )
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return ["{}{}".format(head, lines[0])] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def extra_repr(self):
        return ""


class SoftLabelImageFolder(VisionDataset):
    """Soft labels with image folder Dataset,
    It only supports train split now.

    Args:
        root (string): Root directory where images are located.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
    ):
        super().__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        soft_label = pandas.read_csv(os.path.join(root, "labels.csv"))
        self.soft_label = torch.tensor(soft_label.values)[:, 1:].float()

    def __getitem__(self, index):
        image_data = Image.open(
            os.path.join(
                self.root,
                "images",
                f"{index:07d}.jpg",
            )
        )
        target = self.soft_label[index]
        target /= target.sum()

        if self.transform is not None:
            image_data = self.transform(image_data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image_data, target

    def __len__(self):
        return len(self.soft_label)
