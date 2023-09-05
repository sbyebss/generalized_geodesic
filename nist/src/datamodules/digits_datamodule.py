from typing import Optional

import torch
from einops import rearrange
from jamtorch.data import get_batch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, Dataset

from src.callbacks.w2_callbacks import extract_class, knn_data, transform2torch
from src.datamodules.data_transform import data_transform, inverse_data_transform
from src.datamodules.datasets.small_scale_image_dataset import get_img_dataset
from src.otdd.pytorch.datasets import CustomTensorDataset
from src.transfer_learning.train_nist_classifier import get_nist_num_label

# pylint: disable=W0223


class NIST(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = OmegaConf.create(kwargs)
        self.cfg = cfg
        self.source_train_data: Optional[Dataset] = None
        self.target_train_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.sorted_test_data = []
        self.source_class = get_nist_num_label(self.cfg.source.dataset)

    def prepare_data(self):
        get_img_dataset(self.cfg.source)
        get_img_dataset(self.cfg.target)

    def setup(self, stage: Optional[str] = None):
        self.source_train_data, self.test_data = get_img_dataset(self.cfg.source)
        self.target_train_data, _ = get_img_dataset(self.cfg.target)
        self.preprocess_label()
        self.sort_test_data()

    def preprocess_label(self):
        if self.cfg.source.dataset == "EMNIST":
            self.source_train_data = transform2torch(self.source_train_data)
            self.test_data = transform2torch(self.test_data)
        if self.cfg.target.dataset == "EMNIST":
            self.target_train_data = transform2torch(self.target_train_data)
        self.cluster_source_data()

    def cluster_source_data(self):
        pass

    def sort_test_data(self):
        for cls_idx in range(self.source_class):
            self.sorted_test_data.append(extract_class(self.test_data, cls_idx))

    def train_dataloader(self):
        return [
            DataLoader(
                dataset=self.source_train_data,
                batch_size=self.cfg.dl.batch_size,
                num_workers=self.cfg.dl.num_workers,
                pin_memory=self.cfg.dl.pin_memory,
                shuffle=True,
                drop_last=True,
            ),
            DataLoader(
                dataset=self.target_train_data,
                batch_size=self.cfg.dl.batch_size,
                num_workers=self.cfg.dl.num_workers,
                pin_memory=self.cfg.dl.pin_memory,
                shuffle=True,
                drop_last=True,
            ),
        ]

    def val_dataloader(self):
        loaders = {
            "source": DataLoader(
                dataset=self.source_train_data,
                batch_size=self.cfg.dl.batch_size,
                num_workers=self.cfg.dl.num_workers,
                pin_memory=self.cfg.dl.pin_memory,
                shuffle=True,
                drop_last=True,
            ),
            "target": DataLoader(
                dataset=self.target_train_data,
                batch_size=self.cfg.dl.batch_size,
                num_workers=self.cfg.dl.num_workers,
                pin_memory=self.cfg.dl.pin_memory,
                shuffle=True,
                drop_last=True,
            ),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def test_dataloader(self):
        num_test_sample = self.source_class * 5
        test_data = self.get_test_samples(num_test_sample)
        test_ds = CustomTensorDataset(test_data)
        return DataLoader(
            dataset=test_ds,
            batch_size=num_test_sample,
            num_workers=self.cfg.dl.num_workers,
            pin_memory=self.cfg.dl.pin_memory,
        )

    def get_test_samples(self, batch_size=100, shuffle=False):
        num_class = self.source_class
        num_per_class = int(batch_size / num_class)
        test_feat = torch.zeros(
            [
                num_per_class,
                num_class,
                self.cfg.source.channel,
                self.cfg.source.image_size,
                self.cfg.source.image_size,
            ]
        )
        test_label = torch.zeros([num_per_class, num_class]).to(torch.int64)
        for cls_idx in range(num_class):
            dataloader = DataLoader(
                dataset=self.sorted_test_data[cls_idx],
                batch_size=num_per_class,
                shuffle=shuffle,
            )
            test_feat[:, cls_idx], test_label[:, cls_idx] = get_batch(dataloader)
        test_label = test_label.view(-1)
        test_feat = rearrange(test_feat, "b n c h w -> (b n) c h w")
        return [test_feat, test_label]

    def data_transform(self, feed_dict):
        if isinstance(feed_dict, list):
            feed_dict = feed_dict[0]
        feed_dict = feed_dict.float()
        return data_transform(self.cfg, feed_dict)

    def inverse_data_transform(self, x):
        return inverse_data_transform(self.cfg, x)


class FewShotKnnNIST(NIST):
    def cluster_source_data(self):
        self.source_train_data = knn_data(
            self.source_train_data, self.cfg.knn_data_path
        )
