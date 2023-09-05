from torch.utils.data import DataLoader

from src.callbacks.w2_callbacks import few_shot_data, transform2torch
from src.datamodules.datasets.hdf5_dataset import H5Dataset
from src.datamodules.datasets.small_scale_image_dataset import PyTorchDataset, nist_dataset

NAME_MAP = {
    "Caltech101": "caltech101",
    "Camelyon": "patch_camelyon",
    "Clevr-Count": "clevr_count-all",
    "Clevr-Dist": "clevr_closest-object-detection",
    "DMLab": "dmlab",
    "dSpr-Ori": "dsprites_label-orientation",
    "dSpr-Loc": "dsprites_label-x-position",
    "DTD": "dtd",
    "EuroSAT": "eurosat",
    "Flowers102": "oxford_flowers102",
    "ImageNet": "imagenet1k",
    "KITTI": "kitti_closest-vehicle-distance",
    "OxfordIIITPet": "oxford_iiit_pet",
    "Resisc45": "resisc45",
    "Retinopathy": "diabetic-retinopathy_btgraham-300",
    "sNORB-Azim": "smallnorb_label-azimuth",
    "sNORB-Elev": "smallnorb_label-elevation",
    "Sun397": "sun397",
    "SVHN": "svhn_cropped",
}


def get_vtab_pattern_from_name(ds_name):
    folder = NAME_MAP[ds_name]
    pattern = folder + "/trainval-beit_base"
    return pattern


def get_vtab_train_dataset(vtab_data_path, ds_name):
    pattern = get_vtab_pattern_from_name(ds_name)
    train_dataset = H5Dataset(vtab_data_path, pattern=pattern)
    return train_dataset


def get_vtab_test_dataset(vtab_data_path, ds_name):
    folder = NAME_MAP[ds_name]
    pattern = folder + "/test-beit_base"
    train_dataset = H5Dataset(vtab_data_path, pattern=pattern)
    return train_dataset


def get_train_test_dataset(cfg, data_type=None):
    if data_type is None:
        data_type = cfg.dataset
    if data_type in cfg.nist_datasets:
        train_dataset, test_dataset = nist_dataset(
            data_type, cfg.nist_data_path, cfg.img_size
        )
    else:  # we assume it's a hdf5 dataset
        train_dataset = get_vtab_train_dataset(cfg.vtab_data_path, data_type)
        test_dataset = get_vtab_test_dataset(cfg.vtab_data_path, data_type)
    train_dataset = transform2torch(train_dataset)
    test_dataset = transform2torch(test_dataset)
    test_dataset = PyTorchDataset([test_dataset.data, test_dataset.targets])
    return train_dataset, test_dataset


def get_train_dataset(cfg):
    train_dataset, _ = get_train_test_dataset(cfg)
    return train_dataset


def get_fine_tune_test_dl(cfg, seed):
    train_dataset, test_dataset = get_train_test_dataset(
        cfg, data_type=cfg.fine_tune_dataset
    )
    if cfg.ds_type == "VTAB":
        test_dataset = PyTorchDataset([test_dataset.data, test_dataset.targets])
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)
    # Similar to few-shot datasets. Each class may only have 5~20 images.
    fine_tune_dataset = few_shot_data(train_dataset, cfg.num_shot, seed=seed)
    fine_tune_loader = DataLoader(
        fine_tune_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    return test_loader, fine_tune_loader
