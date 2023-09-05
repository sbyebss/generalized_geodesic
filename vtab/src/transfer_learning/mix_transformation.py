import os
from copy import deepcopy
from functools import partial

import torch
import torch.nn.functional as F
from torchvision.transforms import Grayscale

from ..networks.classifier import SimpleMLP, SpinalNet
from ..networks.mlp import ResFeatureGenerator
from ..networks.unet_simple import Conditional_UNet
from ..otdd.pytorch.datasets import CustomTensorDataset
from ..otdd.pytorch.distance import DatasetDistance
from ..transfer_learning.train_nist_classifier import get_num_label, simple_transformation

# pylint: disable=too-many-arguments,too-many-locals,line-too-long, non-ascii-name, invalid-name
vtab_emb_dim = 768
nist_list = ["MNIST", "USPS", "FMNIST", "KMNIST", "EMNIST", "MNISTM"]


def map_classifier_list_loader(
    load_epochs,
    fine_tune_dataset,
    train_datasets,
    device,
    pretrained_classifier_path,
    otdd_map_dir,
    seed=1,
    num_shot=5,
):
    map_list = []
    classifier_list = []
    num_source_label = get_num_label(fine_tune_dataset)
    num_target_labels = [get_num_label(ds) for ds in train_datasets]
    for num_target_label, load_epoch, target_dataset in zip(
        num_target_labels, load_epochs, train_datasets
    ):
        if fine_tune_dataset in nist_list:
            map_g = Conditional_UNet(num_classes=num_source_label)
            prefix = "C" if fine_tune_dataset == "MNISTM" else fine_tune_dataset[0]
            prefix = prefix + "2X_few_shot"
        else:
            # TODO: this is very ugly.
            map_g = ResFeatureGenerator(
                feat_dim=vtab_emb_dim, num_classes=num_source_label, num_layer=4
            )
            prefix = "VTAB_few_shot"
        map_g.load_state_dict(
            torch.load(
                os.path.join(
                    otdd_map_dir,
                    f"{prefix}/{fine_tune_dataset}/{target_dataset}/Exact_origin/{num_shot}/seed{seed}/map_{load_epoch}_ema.pth",
                )
            )
        )
        if fine_tune_dataset in nist_list:
            target_domain_classifier = SpinalNet(num_target_label)
            target_domain_classifier.load_state_dict(
                torch.load(
                    os.path.join(
                        pretrained_classifier_path,
                        f"{target_dataset}_spinalnet_long.pt",
                    )
                )["model_state_dict"]
            )
        else:
            target_domain_classifier = SimpleMLP(
                vtab_emb_dim, num_class=num_target_label
            )
            target_domain_classifier.load_state_dict(
                torch.load(
                    os.path.join(pretrained_classifier_path, f"{target_dataset}.pt")
                )
            )
        map_g.eval()
        target_domain_classifier.eval()
        map_g = map_g.to(device)
        target_domain_classifier = target_domain_classifier.to(device)

        map_list.append(map_g)
        classifier_list.append(target_domain_classifier)
    return map_list, classifier_list


def get_geodesic_mix_by_method(method, simplex_vector, num_target_classes, device):
    assert method in ["otdd_map", "mixup", "barycenteric_map"], "method not supported"
    if method == "otdd_map":
        return partial(
            gen_geodesic_mix_no_pf,
            weights=simplex_vector,
            num_target_classes=num_target_classes,
            device=device,
            data_transform=None,
        )
    elif method == "mixup":
        return partial(
            mixup,
            weights=simplex_vector,
            num_target_classes=num_target_classes,
            device=device,
            data_transform=simple_transformation,
        )
    elif method == "barycenteric_map":
        return partial(
            gen_geodesic_mix_no_pf,
            weights=simplex_vector,
            num_target_classes=num_target_classes,
            device=device,
            data_transform=simple_transformation,
        )
    return "Not implemented"


def get_random_one_hot_weight(length):
    one_hot_vector = torch.nn.functional.one_hot(
        torch.randint(0, length, (1,)), length
    ).view(-1)
    return one_hot_vector


# mixup data transformation directly from tuple data
def mixup(
    tuple_data: torch.Tensor,
    weights: torch.Tensor,
    num_target_classes: list,
    device: str,
    data_transform=None,
):
    # tuple_data: [(feat1,label1),(feat2,label2),...]
    # feat: size = (b,1,h,w)
    # label: size = (b,)
    # weights: np.array e.g. [0.1, 0.1, 0.8]
    if weights[0] is None:
        weights = get_random_one_hot_weight(len(weights))
    assert (sum(weights) - 1) < 1e-3
    total_label = sum(num_target_classes)
    feat1, label1 = tuple_data[0]
    if data_transform is not None:
        feat1, _ = data_transform(feat1, label1)
    mix_feat = torch.zeros_like(feat1).to(device)
    mix_probs = torch.zeros([feat1.shape[0], total_label]).to(device)
    begin_index = 0
    for index, (n_class, (feat, hard_label)) in enumerate(
        zip(num_target_classes, tuple_data)
    ):
        if weights[index] == 0:
            begin_index += n_class
            continue
        # This is additional for mixup because we didn't transform data
        # during ConcatenateDataset.
        if data_transform is not None:
            feat, hard_label = data_transform(feat, hard_label)
        feat = feat.to(device)
        soft_label = F.one_hot(hard_label, n_class).to(device)
        mix_feat += feat * weights[index]

        mix_probs[:, begin_index : (begin_index + n_class)] += (
            soft_label * weights[index]
        )
        begin_index += n_class
    assert abs(mix_probs.sum(dim=1).mean() - 1) < 1e-3
    return mix_feat, mix_probs


# barycenteric mapping directly from tuple data
def gen_geodesic_mix_no_pf(
    tuple_data: torch.Tensor,
    weights: torch.Tensor,
    num_target_classes: list,
    device: str,
    data_transform=None,
):
    # tuple_data: [(feat1,label1),(feat2,label2),...]
    # feat: size = (b,*)
    # label: size = (b,c)
    # weights: torch.Tensor e.g. [0.1, 0.1, 0.8]
    assert (sum(weights) - 1) < 1e-3
    total_label = sum(num_target_classes)
    feat1, labels1 = tuple_data[0]
    if data_transform is not None:
        feat1, _ = data_transform(feat1, labels1)
    mix_feat = torch.zeros_like(feat1).to(device)
    mix_labels = torch.zeros([feat1.shape[0], total_label]).to(device)
    begin_index = 0

    for index, (n_class, (feat, soft_label)) in enumerate(
        zip(num_target_classes, tuple_data)
    ):
        if weights[index] == 0:
            begin_index += n_class
            continue
        if data_transform is not None:
            feat, soft_label = data_transform(feat, soft_label)
        feat = feat.to(device)
        soft_label = soft_label.to(device)
        mix_feat += feat * weights[index]

        mix_labels[:, begin_index : (begin_index + n_class)] += (
            soft_label * weights[index]
        )
        begin_index += n_class
    # EMNIST -> FMNIST barycentric mapping,
    # some pushforward has all labels zeros, haven't figured out why.
    assert abs(mix_labels.sum(dim=1).mean() - 1) < 1e-2
    return mix_feat, mix_labels


def transform_ds_feat(dataset, dataloader, batch_size, tf_func=simple_transformation):
    for idx, (feat, labels) in enumerate(dataloader):
        feat, _ = tf_func(feat, labels)
        dataset.tensors[0][idx * batch_size : (idx + 1) * batch_size] = feat
    return dataset


def transform_ds_feat_label(dataset, dataloader, tf_func=simple_transformation):
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for idx, (feat, labels) in enumerate(dataloader):
            feat, labels = tf_func(feat, labels)
            cur_size = dataset.tensors[0][
                idx * batch_size : (idx + 1) * batch_size
            ].shape[0]
            dataset.tensors[0][idx * batch_size : (idx + 1) * batch_size] = feat[
                :cur_size
            ].cpu()
            dataset.tensors[1][idx * batch_size : (idx + 1) * batch_size] = labels[
                :cur_size
            ].cpu()
    return dataset


def otdd_map_transform_func(feat, label, feat_map, classifier, device):
    feat, label = simple_transformation(feat, label)
    feat = feat.to(device)
    label = label.to(device)
    pf_feat = feat_map(feat, label)
    output_logits = classifier(pf_feat, None)
    pf_probs = F.softmax(output_logits, dim=1)
    return pf_feat, pf_probs


def barycentric_projection(
    source_ds,
    target_ds,
    feat_shape,
    num_target_class,
    device,
    λ_y=10.0,
    precomputed_label_dist=None,
):
    """
    OTDD projection between labeled datasets
    """
    # 1. bias: inner debiased should be True to cause less error
    # debiased_loss should be False because we want label_distance to be simple.
    # but it doesn't matter, because here we don't care about the loss, we only care about the coupling.
    # 2. batchified: We don't add anything about batchify because we already do batch manually.
    dist = DatasetDistance(
        source_ds,
        target_ds,
        debiased_loss=False,
        batchified=None,
        inner_ot_method="exact",
        inner_ot_debiased=True,
        inner_ot_entreg=1e-2,  # 1e-2 could be burry
        # maxbatch=30000,
        p=2,
        entreg=1e-2,
        min_labelcount=1,
        device=device,
        nworkers_dists=64,
        nworkers_stats=32,
        λ_x=1.0,
        λ_y=λ_y,
    )

    if precomputed_label_dist is not None:
        dist.label_distances = precomputed_label_dist

    # This coupling is calculated based on the subsampling of #[maxsamples] samples.
    # [old returns] _, coupling, target_feat, target_hard_label
    _, dist_log = dist.distance(
        maxsamples=10000, return_log=True, compute_coupling=True
    )
    coupling = dist_log["coupling"]
    target_feat, target_hard_label = dist_log["target_samples"]

    coupling1 = (
        torch.nan_to_num(coupling, nan=100.0, posinf=100.0, neginf=1e-10).abs() + 1e-4
    )
    coupling2 = coupling1 / coupling1.sum(axis=1, keepdims=True)

    cond = coupling1.sum(axis=1) <= 1e-3
    indices = cond.nonzero()
    if len(indices) > 0:
        num_target_data = target_feat.shape[0]
        fake_labels = torch.randint(0, num_target_data, (len(source_ds),))
        one_hot_fake_labels = torch.nn.functional.one_hot(fake_labels, num_target_data)
        coupling2[indices] = one_hot_fake_labels[indices].float().to(device)

    assert abs(coupling2.sum(axis=1).min() - 1.0) < 1e-3
    coupling = coupling2

    pf_feat = coupling @ target_feat
    # OTDD solver reshapes features to a flat vector
    pf_feat = pf_feat.reshape(feat_shape)
    # OTDD solver is shifting target_hard_labels
    # FIXME: This is dangerous. We should instead subtract len(d1.classes) or similar
    target_hard_label -= target_hard_label.min()
    target_soft_labels = F.one_hot(target_hard_label, num_target_class).float()
    pf_probs = coupling @ target_soft_labels
    assert abs(target_soft_labels.sum(axis=-1).min() - 1.0) < 1e-3

    return pf_feat, pf_probs


def bary_map_transform_func(feat, label, target_ds, num_target_class, device):
    # feat: size = (b,c,h,w)
    # label: size = (b,) is hard label
    # target_ds: nist_dataset (target_ds.data, target_ds.targets)
    # also contains hard labels.
    target_ds = deepcopy(target_ds)
    indices = torch.randperm(len(target_ds))[: feat.shape[0]]
    target_ds.data = target_ds.data[indices]
    target_ds.targets = target_ds.targets[indices]
    source_ds = CustomTensorDataset(
        [
            feat,
            label.to(torch.int64),
        ]
    )
    if feat.shape[1] == 3 and len(target_ds.data.shape) == 3:
        target_ds.transform.transforms = [Grayscale(3)] + target_ds.transform.transforms

    # Here to calculate the coupling, we use the (b,1,32,32) data
    # because we hope to save memory for more data.
    # We use coefficients as both 1.0
    return barycentric_projection(
        source_ds,
        target_ds,
        feat.shape,
        num_target_class,
        device,
    )


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
