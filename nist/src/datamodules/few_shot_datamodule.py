import torch

from src.callbacks.w2_callbacks import few_shot_data, transform2torch
from src.datamodules.datasets.small_scale_image_dataset import nist_dataset

# pylint: disable=too-few-public-methods


class FewShotNIST:
    def __init__(
        self,
        data_path,
        fine_tune_dataset="MNIST",
        num_shot=5,
        batch_size=64,
        img_size=32,
        seed=1,
    ):
        super().__init__()

        fine_tune_dataset, test_dataset = nist_dataset(
            fine_tune_dataset, data_path, img_size
        )

        test_dataset = transform2torch(test_dataset)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )

        # Similar to few-shot datasets. Each class may only have 5~20 images.
        fine_tune_dataset = few_shot_data(fine_tune_dataset, num_shot, seed=seed)
        self.fine_tune_loader = torch.utils.data.DataLoader(
            fine_tune_dataset, batch_size=batch_size, shuffle=True
        )

    @property
    def num_data(self):
        return len(self.fine_tune_loader.dataset)
