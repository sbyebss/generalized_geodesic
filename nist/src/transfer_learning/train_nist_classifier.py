import torch
from torch import nn, optim
from torchmetrics.classification.accuracy import Accuracy

from src.networks.classifier import LeNet, SimpleMLP, SpinalNet

# pylint: disable=import-error,inconsistent-return-statements,invalid-name
from src.viz.img import save_tensor_imgs

train_acc = Accuracy()
loss_fn = torch.nn.CrossEntropyLoss()

# pylint: disable=unused-argument


def simple_transformation(feat, labels):
    device = feat.device
    if len(feat.shape) > 2:  # if it's an image
        feat = 2 * feat - 1.0
        # assert feat.min() >= -1.0 and feat.max() <= 1.0
        if feat.shape[1] == 1:
            feat = feat.expand(feat.shape[0], 3, *feat.shape[2:])
    if len(labels.shape) == 1 or labels.shape[1] == 1:
        labels = labels.to(torch.int64).to(device)
    return feat, labels


def inverse_data_transform(data):
    data = (data + 1.0) / 2
    return data


def train_classifier_on_sampler(
    model,
    optimizer,
    sampler,
    data_transform,
    n_iteration,
    device="cuda:1",
    scheduler=None,
):
    model.train()
    for _ in range(n_iteration):
        optimizer.zero_grad()
        samples = sampler.sample()
        feat, labels = data_transform(samples)
        feat = feat.to(device)
        labels = labels.to(device)
        label_logits = model(feat, None)
        # print(feat.max(), feat.min())
        loss = loss_fn(label_logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        print("loss = ", loss.item())
    return feat, labels


def train_lenet_on_dl(model, optimizer, train_loader, data_transform, device="cuda:1"):
    model.train()
    for idx, (feat, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        feat, labels = data_transform(feat, labels)
        feat = feat.to(device)
        labels = labels.to(device)
        if idx == 0:
            record_feat = feat
            record_labels = labels
        label_logits = model(feat, None)
        # print(label_logits.shape)
        loss = loss_fn(label_logits, labels)
        loss.backward()
        optimizer.step()
        print("loss = ", loss.item())
    return record_feat, record_labels


def test_lenet(model, test_loader, device):
    correct = 0
    model.eval()
    model = model.to(device)
    for _, (data, target) in enumerate(test_loader):
        target = target.to(device)
        data, target = simple_transformation(data, target)
        label_logits = model(data.to(device), None)
        pred = torch.argmax(label_logits, dim=1)
        acc = train_acc(pred.detach().cpu(), target.cpu())
        # print(acc.item())
        correct += data.shape[0] * acc
        train_acc.reset()
    return correct / len(test_loader.dataset)


def get_nist_num_label(dataset_name: str):
    if dataset_name == "EMNIST":
        return 26
    else:
        return 10


def get_num_label(dataset_name: str):
    DATASET_NCLASSES = {
        # MNIST and Friends
        "MNIST": 10,
        "MNISTM": 10,
        "FMNIST": 10,
        "FashionMNIST": 10,
        "EMNIST": 26,
        "KMNIST": 10,
        "USPS": 10,
        # VTAB and Other Torchvision Datasets
        "Caltech101": 101,  # VTAB & Torchvision
        "Caltech256": 256,  # Torchvision
        "Camelyon": 2,  # VTAB version
        "PCAM": 2,  # Torchvision version
        "CIFAR10": 10,  # Torchvision
        "CIFAR100": 100,  # VTAB & Torchvision
        "Clevr-Count": 8,  # VTAB
        "Clevr-Dist": 6,  # VTAB
        "DMLAB": 6,  # VTAB
        "DMLab": 6,  # VTAB
        "dSpr-Ori": 16,  # VTAB
        "dSpr-Loc": 16,  # VTAB
        "DTD": 47,  # VTAB & Torchvision
        "EuroSAT": 10,  # VTAB & Torchvision
        "Flowers102": 102,  # VTAB & Torchvision
        "Food101": 101,  # Torchvision
        "tiny-ImageNet": 200,  # None?
        "ImageNet": 1000,  # VTAB & Torchvision
        "KITTI": 4,  # VTAB & Torchvision
        "LSUN": 10,  # Torchvision
        "OxfordIIITPet": 37,  # VTAB & Torchvision
        "Resisc45": 45,  # VTAB
        "Retinopathy": 5,  # VTAB
        "sNORB-Azim": 18,  # VTAB
        "sNORB-Elev": 9,  # VTAB
        "STL10": 10,  # Torchvision
        "Sun397": 397,  # VTAB and Torchvision
        "SVHN": 10,  # VTAB and Torchvision
    }
    return DATASET_NCLASSES[dataset_name]


def add_layer(network, num_test_class):
    if isinstance(network, LeNet):
        network.fc3 = nn.Linear(84, num_test_class)
    elif isinstance(network, SpinalNet):
        # TODO: this can be a problem, hard code layer_width
        network.fc_out[-1] = nn.Linear(128 * 4, num_test_class)
    elif isinstance(network, SimpleMLP):
        network.net[-1] = nn.Linear(100, num_test_class)
    return network


def fine_tune_lenet(
    lenet,
    fine_tune_loader,
    fine_tune_epoch,
    num_test_class,
    device,
    save_img_path,
    lr=1e-3,
):
    lenet = add_layer(lenet, num_test_class)
    lenet = lenet.to(device)
    optimizer = optim.Adam(lenet.parameters(), lr=lr)

    for _ in range(1, fine_tune_epoch + 1):
        feat, _ = train_lenet_on_dl(
            lenet,
            optimizer,
            fine_tune_loader,
            simple_transformation,
            device=device,
        )
    if "feat" in locals():
        save_tensor_imgs(
            inverse_data_transform(feat),
            8,
            0,
            save_img_path,
        )
    return lenet


# pylint: disable=invalid-name,too-many-function-args,too-few-public-methods


class LoaderSampler:
    def __init__(self, loader):
        super().__init__()
        self.loader = loader
        self.it = iter(self.loader)

    def check_batch_size(self, batch):
        if len(batch[0]) < self.loader.batch_size:
            return self.sample()

    def sample(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample()

        self.check_batch_size(batch)

        return batch


class TupleSampler(LoaderSampler):
    def check_batch_size(self, batch):
        if len(batch[0][0]) < self.loader.batch_size:
            return self.sample()
