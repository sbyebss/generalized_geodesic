import os

import torch

from src.models.base_model import BaseModule, turn_off_grad
from src.utils import lht_utils

log = lht_utils.get_logger(__name__)

# pylint: disable=no-self-use,abstract-method,too-many-ancestors


class EmbVtabModule(BaseModule):
    def get_real_data(self, batch):
        source_data, target_data = batch
        return source_data, target_data

    def load_classifier(self):
        classifier_save_path_full = self.cfg.classifier_save_path  # dataset_name.pt
        path_item = classifier_save_path_full.split("/")
        dir_path = "/".join(path_item[:-1])
        file_name = path_item[-1]
        pt_name = file_name.split(".")[0]
        is_found = False
        for file in os.listdir(dir_path):
            if pt_name.lower() in file.lower():
                classifier_save_path_full = os.path.join(dir_path, file)
                is_found = True
                break
        assert is_found, f"Can't find classifier {pt_name} in {dir_path}"

        self.classifier.load_state_dict(torch.load(classifier_save_path_full))
        log.info(
            f"Successfully load the pretrained classifier from <{classifier_save_path_full}>"
        )
        self.pretrain_clsf = False
        turn_off_grad(self.classifier)
