from typing import Optional

import numpy as np
import torch
from einops import repeat
from jamtorch.data import get_batch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from src.viz.points import grid_nn_2_generator

# pylint: disable=abstract-method, attribute-defined-outside-init, no-member


def feat_label_concat(feature, label):
    # feature, label: np.ndarray
    # concatenate feature and label into one vector
    feat_label = np.concatenate([feature, label.reshape(-1, 1)], axis=1)
    feat_label = torch.from_numpy(feat_label).float()
    return feat_label[torch.randperm(feat_label.shape[0])]


class BaseDataModule(LightningDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        self.source_data = self.source_sample(self.train_size)
        self.target_data = self.target_sample(self.train_size)

    def train_dataloader(self):
        self.setup()
        loader_source = DataLoader(
            self.source_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
        )
        loader_target = DataLoader(
            self.target_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
        )
        return [loader_source, loader_target]

    def get_test_samples(self, batch_size):
        self.setup()
        test_s = DataLoader(
            self.source_data[:batch_size], batch_size=batch_size, shuffle=True
        )
        test_t = DataLoader(
            self.target_data[:batch_size], batch_size=batch_size, shuffle=True
        )
        return get_batch(test_s), get_batch(test_t)


class ChessboardDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = OmegaConf.create(kwargs)
        self.cfg = cfg
        self.train_size = self.cfg.n_train_samples
        num_gmm_component = self.cfg.num_gmm_source
        num_dots = int(np.sqrt(num_gmm_component))
        means_ = grid_nn_2_generator(num_dots, -num_dots, num_dots)
        self.data_gmm = GaussianMixture(n_components=num_gmm_component)
        self.data_gmm.weights_ = np.ones(num_gmm_component) / num_gmm_component
        self.data_gmm.means_ = means_
        self.data_gmm.covariances_ = [
            np.eye(cfg.dims) * 0.08 for _ in range(num_gmm_component)
        ]

        self.label_permute_tab = np.random.permutation(num_gmm_component)

    def source_sample(self, n: int):
        feature, label = self.data_gmm.sample(n)
        return feat_label_concat(feature, label)

    def target_sample(self, n: int):
        feature, label = self.data_gmm.sample(n)
        label = self.label_permute_tab[label]
        return feat_label_concat(feature, label)


class GMMDataModule(BaseDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = OmegaConf.create(kwargs)
        self.cfg = cfg
        self.train_size = self.cfg.n_train_samples
        num_gmm_source = self.cfg.num_gmm_source
        num_gmm_target = self.cfg.num_gmm_target
        self.source_gmm = GaussianMixture(n_components=num_gmm_source)
        self.source_gmm.weights_ = np.ones(num_gmm_source) / num_gmm_source
        self.source_gmm.means_ = np.array([[0, 1], [0, -1]])
        self.source_gmm.covariances_ = [
            np.eye(cfg.dims) * 0.06 for _ in range(num_gmm_source)
        ]
        self.target_gmm = GaussianMixture(n_components=num_gmm_target)
        self.target_gmm.weights_ = np.ones(num_gmm_target) / num_gmm_target
        self.target_gmm.means_ = np.array([[-2, 0], [0, 0], [2, 0]])
        self.target_gmm.covariances_ = [
            np.eye(cfg.dims) * 0.06 for _ in range(num_gmm_target)
        ]

    def source_sample(self, n: int):
        feature, label = self.source_gmm.sample(self.cfg.raw_n_source_samples)
        self.raw_source_data = feat_label_concat(feature, label)
        return repeat(
            self.raw_source_data,
            "b d -> (tile b) d",
            tile=int(n / self.cfg.raw_n_source_samples),
        )

    def target_sample(self, n: int):
        feature, label = self.target_gmm.sample(self.cfg.raw_n_target_samples)
        self.raw_target_data = feat_label_concat(feature, label)
        return repeat(
            self.raw_target_data,
            "b d -> (tile b) d",
            tile=int(n / self.cfg.raw_n_target_samples),
        )
