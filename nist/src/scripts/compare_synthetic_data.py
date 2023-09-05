"""
Compare the effect of synthetic datasets through a transfer learning task
"""
# pylint: disable=line-too-long,not-callable,redefined-outer-name,too-many-locals,no-value-for-parameter,too-many-statements,wildcard-import
import os
from collections import defaultdict
from typing import Dict, List

from src.networks.classifier import SimpleMLP, SpinalNet
from src.scripts import *
from src.transfer_learning.data_utils import get_fine_tune_test_dl
from src.transfer_learning.gen_geodesic import tuple_dataloader
from src.transfer_learning.mix_transformation import get_geodesic_mix_by_method
from src.transfer_learning.train_nist_classifier import TupleSampler, add_layer

log = lht_utils.get_logger(__name__)


def get_network(cfg, total_label):
    if cfg.fine_tune_dataset in cfg.nist_datasets:
        cls_model = SpinalNet(num_class=total_label)
    else:
        cls_model = SimpleMLP(cfg.vtab_emb_dim, num_class=total_label)
    return cls_model


@hydra.main(config_path="../../configs/scripts", config_name="compare_methods")
def compare_methods(cfg: omegaconf.DictConfig):
    if cfg.auto_gpu:
        best_id = get_gpu_by_utils(1, sleep_sec=10)
        cfg.device = f"cuda:{best_id[0]}"

    if cfg.full_dataset:
        # We use the interpolation of full datasets for training.
        cfg.train_datasets = cfg.all_datasets
        if cfg.fine_tune_dataset in cfg.train_datasets:
            cfg.train_datasets.remove(cfg.fine_tune_dataset)

    num_target_classes = [get_num_label(ds) for ds in cfg.train_datasets]
    total_label = sum(num_target_classes)

    if cfg.ds_type == "NIST":
        target_alias = "".join(ds[0] for ds in cfg.train_datasets)
    elif cfg.ds_type == "VTAB":
        target_alias = "".join(ds + "_" for ds in cfg.train_datasets)

    num_source_label = get_num_label(cfg.fine_tune_dataset)
    num_seed = len(cfg.seeds)

    accuracy_save_path = (
        f"train_on_{target_alias}_epoch{cfg.fine_tune_epoch}_repeat{num_seed}.pt"
    )
    #! This means we have to finish all seeds in one run.
    # And we need to load the last several accuracies when reading accuracy.
    accuracies: Dict[str, List[float]] = (
        torch.load(accuracy_save_path)
        if os.path.exists(accuracy_save_path)
        else defaultdict(list)
    )

    best_itp_params: Dict[str, np.array] = torch.load(
        os.path.join(
            cfg.work_dir,
            f"logs/otdd_ternary_transport_metric/{cfg.num_shot}_shot",
            f"external_{cfg.fine_tune_dataset}/from_{cfg.fine_tune_dataset}2{target_alias}_repeat{num_seed}.pth",
        )
    )["best_interpo_params"]
    log.info(f"The best interpolation parameter is {best_itp_params}")

    for seed in cfg.seeds:
        # This param is normalized to sum to 1.
        simplex_vector = best_itp_params[f"seed{seed}"]
        for method in cfg.methods:
            dl_path = f"{method}_dl_interpl_{target_alias}_seed{seed}.pt"
            log.info(f"The path of dataloder is {dl_path}")
            tuple_dl = tuple_dataloader(
                method,
                cfg,
                seed=seed,
                dl_path=dl_path,
            )
            pf_sampler = TupleSampler(tuple_dl)

            classifier_path = f"{method}_classifier_{target_alias}_seed{seed}.pt"

            if not os.path.exists(classifier_path) or cfg.retrain:
                cls_model = get_network(cfg, total_label)
                cls_model = cls_model.to(cfg.device)
                optimizer = optim.Adam(cls_model.parameters(), lr=cfg.lr)
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=3000, gamma=0.5
                )
                data_transformer = get_geodesic_mix_by_method(
                    method, simplex_vector, num_target_classes, cfg.device
                )
                feat, labels = train_classifier_on_sampler(
                    cls_model,
                    optimizer,
                    pf_sampler,
                    data_transformer,
                    cfg.train_iteration,
                    device=cfg.device,
                    scheduler=scheduler,
                )
                save_tensor_imgs(
                    inverse_data_transform(feat),
                    8,
                    0,
                    f"train_on_{target_alias}_method_{method}_seed{seed}",
                )
                log.info(f"sanity check labels: {torch.argmax(labels, dim=-1)}")
                log.info(
                    f"Finish training on one interpolation, the interpolation method is {method}"
                )
                torch.save(
                    {"model_state_dict": cls_model.state_dict()},
                    classifier_path,
                )
            else:
                log.info(
                    f"Load the pretrained classifier, the interpolation method is {method}"
                )

        # Fine-tune dataloder
        test_loader, fine_tune_loader = get_fine_tune_test_dl(cfg, seed=seed)
        # sanity check
        assert len(fine_tune_loader.dataset) == cfg.num_shot * num_source_label

        # Fine-tuning, and testing for different interpolation time
        for method in cfg.methods:
            ft_model_path = (
                f"fine_tune_{method}_classifier_{target_alias}_seed{seed}.pt"
            )
            cls_model = get_network(cfg, total_label)

            if not os.path.exists(ft_model_path) or cfg.refine_tune:
                # Now we choose to fine-tune all the layers
                # instead of only the last one
                cls_model.load_state_dict(
                    torch.load(f"{method}_classifier_{target_alias}_seed{seed}.pt")[
                        "model_state_dict"
                    ]
                )
                cls_model = fine_tune_lenet(
                    cls_model,
                    fine_tune_loader,
                    cfg.fine_tune_epoch,
                    num_source_label,
                    cfg.device,
                    f"fine_tune_{cfg.fine_tune_dataset}_seed{seed}",
                    lr=cfg.lr,
                )
                torch.save(cls_model.state_dict(), ft_model_path)
                log.info(
                    f"Finish fine tuning on one interpolation, method = {method}, seed = {seed}"
                )
            else:
                cls_model = add_layer(cls_model, num_source_label)
                cls_model.load_state_dict(torch.load(ft_model_path))
                log.info(f"Load the fine tuned model, method = {method}, seed = {seed}")

            accuracy = test_lenet(cls_model, test_loader, cfg.device).item()
            accuracies[method].append(accuracy)

            log.info(f"accuracy = {accuracy}")

        torch.save(
            accuracies,
            f"train_on_{target_alias}_epoch{cfg.fine_tune_epoch}_repeat{seed}.pt",
        )
    torch.save(accuracies, accuracy_save_path)


if __name__ == "__main__":
    compare_methods()
