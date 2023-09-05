import os

import hydra
import omegaconf
import torch
from jammy.cli.gpu_sc import get_gpu_by_utils
from torch import optim

from src.callbacks.w2_callbacks import few_shot_data, transform2torch
from src.transfer_learning.data_utils import get_train_test_dataset

# pylint: disable=unused-import
from src.transfer_learning.train_nist_classifier import (
    LoaderSampler,
    get_num_label,
    simple_transformation,
    test_lenet,
    train_classifier_on_sampler,
)

# pylint: disable=no-value-for-parameter


def data_transform(samples):
    data, target = samples
    return simple_transformation(data, target)


@hydra.main(config_path="../../configs/scripts", config_name="pretrain_classifier")
def pretrain_classifier(cfg: omegaconf.DictConfig):
    if cfg.auto_gpu:
        best_id = get_gpu_by_utils(1, sleep_sec=10)
        cfg.device = f"cuda:{best_id[0]}"

    if not os.path.exists(cfg.classifier_path):
        os.makedirs(cfg.classifier_path)
    if not os.path.exists(cfg.few_shot_classifier_path):
        os.makedirs(cfg.few_shot_classifier_path)
    for train_iter, data_type in zip(cfg.train_iters, cfg.all_datasets):
        train_dataset, test_dataset = get_train_test_dataset(cfg, data_type=data_type)

        if cfg.few_shot:
            train_dataset = few_shot_data(
                train_dataset, n_shot=cfg.num_shot, seed=cfg.seed
            )
            total_data = len(train_dataset)
            assert len(train_dataset) == cfg.num_shot * get_num_label(data_type)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=total_data, shuffle=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                drop_last=True,
            )
        train_sampler = LoaderSampler(train_loader)

        classifier_net = hydra.utils.instantiate(
            cfg.classifier_net, num_class=get_num_label(data_type)
        )
        classifier_net = classifier_net.to(cfg.device)
        optimizer = optim.Adam(classifier_net.parameters(), lr=cfg.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
        train_classifier_on_sampler(
            classifier_net,
            optimizer,
            train_sampler,
            data_transform,
            train_iter,
            device=cfg.device,
            scheduler=scheduler,
        )

        if cfg.few_shot:
            save_path = os.path.join(
                cfg.few_shot_classifier_path,
                f"{data_type}_{cfg.num_shot}_shot_{cfg.net_name}_seed{cfg.seed}.pt",
            )
        else:
            save_path = os.path.join(
                cfg.classifier_path, f"{data_type}_{cfg.net_name}.pt"
            )
        if test_dataset is not None:
            test_dataset = transform2torch(test_dataset)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=cfg.batch_size, shuffle=True
            )
            accuracy = test_lenet(classifier_net, test_loader, cfg.device)
            torch.save(
                {"model_state_dict": classifier_net.state_dict(), "accuracy": accuracy},
                save_path,
            )
            print(f"Done! Accuracy = {accuracy}")
        else:
            torch.save(classifier_net.state_dict(), save_path)


if __name__ == "__main__":
    pretrain_classifier()
