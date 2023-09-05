# pylint: skip-file
import os
import sys
import argparse
import inspect

import matplotlib.pyplot as plt
import numpy as np
import glob
import requests
import torch
from PIL import Image
import torchvision
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import h5py
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# from src.viz.img import save_seperate_imgs
# sys.path.append(".")
# import models_mae
from src.models import non_mask_mae as models_mae
# from src.datamodules.datasets.hdf5_dataset import H5Dataset


# GLobals
# DATASETS_PATH = '/data'
MAX_DUMP_SIZE = 10000

# Map from short dataset names to those in  embedding dumps
# When the dataset exists in torchvision, I default that spelling
NAME_MAP = {
    'Caltech101': 'caltech101',
    'CIFAR100': 'cifar100',
    'Camelyon': 'patch_camelyon',
    'Clevr-Count': 'clevr_count-all',
    'Clevr-Dist': 'clevr_closest-object-detection',
    'DMLab': 'dmlab',
    'dSpr-Ori': 'dsprites_label-orientation',
    'dSpr-Loc': 'dsprites_label-x-position',
    'DTD': 'dtd',
    'EuroSAT': 'eurosat',
    'Flowers102': 'oxford_flowers102',
    'ImageNet': 'imagenet1k',
    'KITTI': 'kitti_closest-vehicle-distance',
    'OxfordIIITPet': 'oxford_iiit_pet',
    'Resisc45': 'resisc45',
    'Retinopathy': 'diabetic-retinopathy_btgraham-300',
    'sNORB-Azim': 'smallnorb_label-azimuth',
    'sNORB-Elev': 'smallnorb_label-elevation',
    'SUN397': 'sun397',
    'SVHN': 'svhn_cropped',
}

# Map from short dataset names to those in VTAB dataset directory
VTAB_DATASETS = {
    'Caltech101': 'caltech101',
    'Cifar100': 'cifar100',
    'Clevr-Dist': 'clevr_closest-object-detection',  # 'clevr(task="closest_object_distance")',
    'Clevr-Count': 'clevr_count-all',  # 'clevr(task="count_all")',
    # 'diabetic_retinopathy(config="btgraham-300")',
    'Retinopathy': 'diabetic-retinopathy_btgraham-300',
    'DMLab': 'dmlab',  # requires adding missing dataset_builder.download_and_prepare() in baseclass
    # 'dsprites(predicted_attribute="label_x_position",num_classes=16)',
    'dSpr-Loc': 'dsprites_label-x-position',
    # 'dsprites(predicted_attribute="label_orientation",num_classes=16)',
    'dSpr-Ori': 'dsprites_label-orientation',
    'DTD': 'dtd',  # removed in in the v2 benchmark because of colission but still accessible here
    'EuroSAT': 'eurosat',
    # kitti(task="closest_vehicle_distance")',  # requires updating the version in code to 3.2.0
    'KITTI': 'kitti_closest-vehicle-distance',
    'Flowers102': 'oxford_flowers102',
    'OxfordIIITPet': 'oxford_iiit_pet',
    'Camelyon': 'patch_camelyon',
    'Resisc45': 'resisc45',
    'sNORB-Azim': 'smallnorb_label-azimuth',  # 'smallnorb(predicted_attribute="label_azimuth")',
    # 'smallnorb(predicted_attribute="label_elevation")',
    'sNORB-Elev': 'smallnorb_label-elevation',
    # messed up https://github.com/tensorflow/datasets/issues/2889, downloaded version has 1 fewer image
    'Sun397': 'sun397',
    'SVHN': 'svhn_cropped',
}

DATASET_NCLASSES = {
    # MNIST and Friends
    'MNIST': 10,
    'FashionMNIST': 10,
    'EMNIST': 26,
    'KMNIST': 10,
    'USPS': 10,
    # VTAB and Other Torchvision Datasets
    'Caltech101': 101,      # VTAB & Torchvision
    'Caltech256': 256,      # Torchvision
    'Camelyon': 2,          # VTAB version
    'PCAM': 2,              # Torchvision version
    'CIFAR10': 10,          # Torchvision
    'CIFAR100': 100,        # VTAB & Torchvision
    'Clevr-Count': 8,       # VTAB
    'Clevr-Dist': 6,        # VTAB
    'DMLab': 6,             # VTAB
    'dSpr-Ori': 16,         # VTAB
    'dSpr-Loc': 16,         # VTAB
    'DTD': 47,              # VTAB & Torchvision
    'EuroSAT': 10,          # VTAB & Torchvision
    'Flowers102': 102,      # VTAB & Torchvision
    'Food101': 101,         # Torchvision
    'tiny-ImageNet': 200,   # None?
    'ImageNet': 1000,       # VTAB & Torchvision
    'KITTI': 4,             # VTAB & Torchvision
    'LSUN': 10,             # Torchvision
    'OxfordIIITPet': 37,    # VTAB & Torchvision
    'Resisc45': 45,         # VTAB
    'Retinopathy': 5,       # VTAB
    'sNORB-Azim': 18,       # VTAB
    'sNORB-Elev': 9,        # VTAB
    'STL10': 10,            # Torchvision
    'Sun397': 397,          # VTAB and Torchvision
    'SVHN': 10,             # VTAB and Torchvision
}

DATASET_NORMALIZATION = {
    'MNIST': ((0.1307,), (0.3081,)),
    'USPS': ((0.1307,), (0.3081,)),
    'FashionMNIST': ((0.1307,), (0.3081,)),
    'QMNIST': ((0.1307,), (0.3081,)),
    'EMNIST': ((0.1307,), (0.3081,)),
    'KMNIST': ((0.1307,), (0.3081,)),
    'ImageNet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'Retinopathy': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'tiny-ImageNet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'CIFAR10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'CIFAR100': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'STL10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0, int(1e8), size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)


# def get_datapath(dname, fold='train', group='VTAB'):
#     if 'imagenet' in dname.lower():
#         srcpath = os.path.join(DATASETS_PATH, 'imagenet', fold)
#     else:
#         dpath = VTAB_DATASETS[dname]
#         srcpath = os.path.join(DATASETS_PATH, group, dpath + '_pytorch_imfolder', fold)
#     return srcpath


def get_torchvision_dataset(dataname, root='/data', transforms=None, download=True, resize=None, rotation=None, **kwargs):
    if 'rot' in dataname.lower():
        dataname, rotation = dataname.split('-')
        rotation = float(rotation.replace('Rot', ''))
    assert hasattr(torchvision.datasets, dataname), f"torchvision.datasets.{dataname} not found"
    DSET = getattr(torchvision.datasets, dataname)

    if transforms is None:
        if dataname.lower() == 'imagenet':
            transforms = torchvision.models.ResNet18_Weights.DEFAULT.transforms()
        else:
            # Rotation should be done before normalization, because it fills with 0
            transforms = [torchvision.transforms.RandomRotation(
                (rotation, rotation))] if rotation is not None else []
            transforms += [torchvision.transforms.Resize(resize)] if resize is not None else []
            transforms += [torchvision.transforms.ToTensor()]
            transforms += [torchvision.transforms.Normalize(*DATASET_NORMALIZATION[dataname])]
            transforms = torchvision.transforms.Compose(transforms)

    if dataname.lower() == 'imagenet':
        _root = os.path.join(root, 'imagenet_pytorch')
        dset = {
            'train': DSET(root=_root, split='train', transform=transforms),
        }
        try:
            dset['test'] = DSET(root=_root, split='val', transform=transforms)
        except:
            dset['test'] = None

    else:  # downloadable datasets
        _root = os.path.join(root, 'torchvision')
        if 'train' in inspect.getfullargspec(DSET).args:
            dset = {
                'train': DSET(root=_root, train=True, download=download, transform=transforms),
                'test': DSET(root=_root, train=False, download=download, transform=transforms)
            }
        elif 'split' in inspect.getfullargspec(DSET).args:
            dset = {}
            for split in ['train', 'trainval', 'val', 'test']:
                try:
                    dset[split] = DSET(root=_root, split=split,
                                       download=download, transform=transforms)
                except:
                    continue
        else:
            print("Warning: does not accept 'train' nor 'split' args. Will use only split as train")
            dset = {
                'train': DSET(root=_root, download=download, transform=transforms)}
    if not 'train' in dset and 'trainval' in dset:
        print(f'WARNING: {dataname} does not have train split. Will use trainval instead.')
        dset['train'] = dset['trainval']

    # Make sure all splits have targets attribute
    for s, d in dset.items():
        if not hasattr(d, 'targets'):
            if hasattr(d, 'y'):
                d.targets = d.y
            elif hasattr(d, '_labels'):
                d.targets = d._labels
            elif hasattr(d, 'labels'):
                d.targets = d.labels
    assert hasattr(d, 'targets'), 'Failed to find targets for dataset'
    return dset


def transform_factory(embedding=None, library='torchvision', resize=None):
    if embedding in [None, 'None', 'none', 'euclidean']:
        transform_list = [
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.Resize(resize),
            # torchvision.transforms.CenterCrop(resize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*DATASET_NORMALIZATION['ImageNet']),
        ]
    elif library == 'torchvision':
        # No resize when using embedding
        if resize:
            print("Ignoring resize arg in transform_factory since embedding is provided")
        # Transform from  https://pytorch.org/vision/stable/models.html
        transform_list = [
            # torchvision.transforms.Resize(256),
            # torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*DATASET_NORMALIZATION['ImageNet']),
        ]
    elif library == 'timm':
        if resize:
            print("Ignoring resize arg in transform_factory since embedding is provided")
        # Take a look at:
        # https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/data/loader.py#L184
        # https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/data/transforms_factory.py#L167
        breakpoint()
    else:
        raise ValueError('Model library not recognized')

    transform = torchvision.transforms.Compose(transform_list)
    return transform


# def show_image(image, title=""):
#     # image is [H, W, 3]
#     assert image.shape[2] == 3
#     plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
#     plt.title(title, fontsize=16)
#     plt.axis("off")
#     return


def prepare_model(chkpt_dir, arch="mae_vit_large_patch16"):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    return model


def cat_and_dump_hdf5(latents, Ys, out_path, fold, nfile=0):
    latents = torch.cat(latents, dim=0)
    Ys = torch.cat(Ys, dim=0)
    print(latents.shape, Ys.shape)
    with h5py.File(os.path.join(out_path, f'{fold}-mae-{nfile}.hdf5'), "w") as f:
        #data = f.create_dataset('/tmp/data', shape=latents.shape, dtype=np.float32, fillvalue=0)
        f.create_dataset('X', data=latents, dtype='float32')
        f.create_dataset('y', data=Ys, dtype='float32')


def main(ds_name, args):
    assert ds_name in NAME_MAP.keys(), f"Dataset {ds_name} not found"
    # Check if already embedded
    if args.subsample:
        out_path = os.path.join(
            args.outpath, NAME_MAP[ds_name], f'train{args.subsample}_seed{args.seed}')
        pattern = f'train{args.subsample}_seed{args.seed}'
    else:
        out_path = os.path.join(args.outpath, NAME_MAP[ds_name], args.fold)
        pattern = args.fold
    if os.path.exists(out_path) and len(os.listdir(out_path)) > 0:
        print(f"Found {out_path}. Skipping.")
        return
    print(f"Embedding {ds_name} to {out_path}")
    os.makedirs(out_path, exist_ok=True)

    model = args.embedding_model
    device = f'cuda:{args.device}' if torch.cuda.is_available() and args.device > -1 else 'cpu'
    input_dim = 512 if model == 'resnet-18' else 768

    #transform = transform_factory(embedding=None, library='torchvision', resize=224)
    transforms = models.ResNet50_Weights.DEFAULT.transforms()

    # Older approach via ImageFolder
    ###dset = ImageFolder(get_datapath(ds_name,fold=args.fold,group='VTAB'), transform=transforms)
    # Newer approach via torch-native VTAB datasets - DMLab is not here!
    dset = get_torchvision_dataset(
        ds_name, root=args.data_path, transforms=transforms, download=True, resize=None, rotation=None)
    dset = dset['train']

    # TODO: pool together train val for small VTAB datasets when selecting 1k ?
    if args.subsample is not None:
        X = np.arange(len(dset))
        y = np.array(dset.targets)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=args.subsample, test_size=len(
            dset) - args.subsample, random_state=args.seed)
        for _, (train_idxs, _) in enumerate(sss.split(X, y)):
            train_idxs = sorted(train_idxs)
        #cts = orch.unique(torch.tensor([dset.targets[i] for i in train_idxs]), return_counts=True)[1]
        dset = torch.utils.data.Subset(dset, train_idxs)
        torch.save(train_idxs, os.path.join(out_path, 'indices.pt'))

    # if args.subsample is None:
    #     dldr_args = {'batch_size': args.batch_size,'shuffle': args.shuffle, 'num_workers':args.num_workers}
    # else:
    #     dldr_args = {'shuffle': args.shuffle, 'num_workers':args.num_workers,
    #                 'batch_sampler': StratifiedBatchSampler(np.array(dset.targets), batch_size=args.batch_size)}
    # dldr = DataLoader(dset, **dldr_args)

    dldr = DataLoader(dset, batch_size=args.batch_size,
                      shuffle=args.shuffle, num_workers=args.num_workers)

    chkpt_dir = "data/ckpts/mae_visualize_vit_large_ganloss.pth"
    model_mae_gan = prepare_model(chkpt_dir, "mae_vit_large_patch16").to(device)
    model_mae_gan.eval()
    print("Model loaded.")

    latents, Ys = [], []
    nsamples = 0
    nfiles = 0
    breakpoint()
    for i, (X, Y) in enumerate(tqdm(dldr)):
        with torch.no_grad():
            X = X.to(device)
            if nsamples + X.shape[0] > MAX_DUMP_SIZE:
                # Dump to file, reset counters
                cat_and_dump_hdf5(latents, Ys, out_path, pattern, nfile=nfiles)
                latents, Ys = [], []
                nsamples = 0
                nfiles += 1

            nsamples += X.shape[0]
            latent = model_mae_gan.forward_encoder(X.float())
            latents.append(latent.detach().flatten(start_dim=1).cpu())
            Ys.append(Y)

    # Dump the leftovers
    cat_and_dump_hdf5(latents, Ys, out_path, pattern, nfile=nfiles)
    print(f"Done with {ds_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main scropt for embedding data for GeodOT',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--outpath', type=str, default='/data/VTAB-mae-embeddings/')
    parser.add_argument('--datapath', type=str, default='/data/')
    parser.add_argument('--fold', type=str, default='train')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--arch', type=str, default='MLP')
    parser.add_argument('--datasets', nargs="+",
                        default=['ImageNet'], metavar="D", help="datasets to embed")
    parser.add_argument('--batch_size', type=int, default=200 if torch.cuda.is_available() else 64)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--embedding_model', type=str, default='beit_base')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--subsample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    for ds_name in args.datasets:
        main(ds_name, args)

    print("Done.")
