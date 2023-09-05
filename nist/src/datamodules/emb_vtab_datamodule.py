from functools import partial
from typing import Optional

from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.transfer_learning.gen_geodesic import get_knn_or_full_emb_vtab_data
from src.transfer_learning.train_nist_classifier import get_num_label

# pylint: disable=W0223


class FewShotEmbVTAB(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = OmegaConf.create(kwargs)
        self.cfg = cfg
        self.source_train_data: Optional[Dataset] = None
        self.target_train_data: Optional[Dataset] = None
        self.source_class = get_num_label(self.cfg.source)

    def setup(self, stage: Optional[str] = None):
        get_few_shot_dataset = partial(
            get_knn_or_full_emb_vtab_data,
            vtab_data_path=self.cfg.vtab_data_path,
            batch_size=self.cfg.batch_size,
        )
        self.source_train_data, _ = get_few_shot_dataset(
            ds_name=self.cfg.source, few_shot=True, knn_data_path=self.cfg.knn_data_path
        )
        self.target_train_data, _ = get_few_shot_dataset(
            ds_name=self.cfg.target, few_shot=False
        )

    def train_dataloader(self):
        return [
            DataLoader(
                dataset=self.source_train_data,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                drop_last=True,
            ),
            DataLoader(
                dataset=self.target_train_data,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                drop_last=True,
            ),
        ]
