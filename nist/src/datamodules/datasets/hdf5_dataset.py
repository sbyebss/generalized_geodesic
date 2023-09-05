"""
Modified from https://vict0rs.ch/2021/06/15/pytorch-h5/
"""
# pylint: skip-file
import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class H5Dataset(Dataset):
    def __init__(
        self,
        h5_path,
        transform=None,
        pattern=None,
        limit=-1,
        target_name="y",
        window=1,
        datatype="vector",
        memory_cache=False,
    ):
        self.limit = limit
        self.h5_path = h5_path
        self.pattern = pattern
        self._archives = self.open_archives()
        self.data = torch.from_numpy(self._archives[0]["X"][:])
        self.targets = torch.from_numpy(self._archives[0][target_name][:])
        self.indices = {}
        if datatype == "image":
            self.transform = [transforms.ToPILImage()]
        else:
            self.transform = []

        if transform is not None:
            try:
                self.transform.extend(
                    transform.transforms
                )  # assuming we get a composed transform, so we need to access the individual elements
            except TypeError:
                self.transform.extend(
                    transform.transforms.transforms
                )  # we are likely getting a weird ClassificationPresetTrain from pytorch-pretrained-models, so need to go deeper

        idx = 0
        self.transform = transforms.Compose(self.transform)
        for a, archive in enumerate(self.archives):
            for idx in range(len(archive["X"])):
                self.indices[idx] = (a, idx)
                idx += 1

        # print(self.archives, self.pattern)
        ### For one-hot encoding?
        # self.classes = np.arange(0, self.archives[0][target_name].shape[0]) # np.unique(self.archives[0]['y'][:])
        ### For normal encoding:
        self.classes = np.unique(self.archives[0][target_name][:])
        # print(self.classes)
        self.window = window
        self.target_name = target_name
        self.memory_cache = memory_cache

        if self.memory_cache:
            self.cache = []
            for a, archive in enumerate(self.archives):
                self.cache.append({})

                self.cache[a]["X"] = archive["X"][:]
                self.cache[a][self.target_name] = archive[self.target_name][:]

        self._archives = None

    def open_archives(self):
        return [
            h5py.File(h5_file, "r")
            for h5_file in glob.glob(os.path.join(self.h5_path, f"{self.pattern}.hdf5"))
        ]

    @property
    def archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = self.open_archives()
        return self._archives

    def __getitem__(self, index):
        a, i = self.indices[index]
        archive = self.archives[a]
        if self.memory_cache:
            X = torch.from_numpy(self.cache[a]["X"][i : i + self.window])
            y = torch.from_numpy(self.cache[a][self.target_name][i : i + self.window])
        else:
            X = torch.from_numpy(archive["X"][i : i + self.window])
            y = archive[self.target_name][i : i + self.window]

        # to preserve pytorch dataset compatibility
        if self.window == 1:
            X = X[0]
            y = y[0]

        # if it's in channel last format, we need to swap the channels in the 3d tensor
        if X.shape[-1] == 3:
            if self.window > 1:
                X = X.permute(0, 3, 1, 2)
            else:
                X = X.permute(2, 0, 1)
        
        if X.ndim >= 3 and X.shape[-2] == 1 and X.shape[-1] == 1:
            ## DAM. Patch. Had to add a squeeze here because some of the embedded datasets have (N,D,1,1) dims.
            X = X.squeeze()

        if self.transform is not None:
            X = self.transform(X)

        # if self.target_name == 'y_raw':
        #     # cast to double
        #     y = torch.from_numpy(y).double()

        return X, y  # {"data": X, "labels": y}

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)


if __name__ == "__main__":
    from torch.utils.data import DistributedSampler

    dset = H5Dataset(
        "/mnt/nerds5/AutoML_datasets/VTAB-h5/synth3/",
        pattern="test",
        transform=transforms.Compose([transforms.PILToTensor()]),
        memory_cache=True,
    )
    sampler = DistributedSampler(dset, num_replicas=4, rank=0, shuffle=True)
    loader = torch.utils.data.DataLoader(
        dset, num_workers=32, batch_size=32, shuffle=True
    )
    # loader = torch.utils.data.DataLoader(H5Dataset('/mnt/nerds5/AutoML_datasets/VTAB-h5/synth3/', pattern='train', window=256), num_workers=32, batch_size=None)
    # import pdb
    # pdb.set_trace()
    # measure number of samples per second
    import time

    start = time.time()

    epochs = 10
    samples = 0
    for e in range(epochs):  # simulated epochs
        for i, data in enumerate(loader):

            if samples % 100 == 0:
                print(f"{samples}/{len(loader)*epochs*len(data[0])}")
                print(f"{(samples+1)/(time.time() - start)} samples per second")
                print(f"{(samples+1)/((time.time() - start)/60)} samples per minute")
                print(f"{(samples+1)/((time.time() - start)/3600)} samples per hour")

            samples += len(data[0])
