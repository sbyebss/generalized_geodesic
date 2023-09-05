import os
import re 
import glob
from tabnanny import check
import src.datamodules.datasets.hdf5_dataset
from src.datamodules.datasets.hdf5_dataset import H5Dataset
import argparse
import pandas as pd
import seaborn as sn
import torch
from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchmetrics
from torchmetrics import Accuracy
from torchvision import transforms
import torchvision
from torchvision.datasets import MNIST

from src.callbacks.w2_callbacks import few_shot_data


# Map from short dataset names to those in  embedding dumps
# When the dataset exists in torchvision, I default that spelling
NAME_MAP = {
    'Caltech101': 'caltech101',
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


class LTMLP(LightningModule):
    def __init__(self,input_dim, dropout, nclass, lr=1e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(500, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, nclass),
        )
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()        
        self.test_acc  = torchmetrics.Accuracy()       
        self.lr = lr 

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y = y.long()        
        loss = F.cross_entropy(preds, y)
        self.train_acc(preds, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y = y.long()
        loss = F.cross_entropy(preds, y)
        self.valid_acc(preds, y)
        self.log("val_loss", loss)        
        self.log("val_acc", self.valid_acc, on_step=True, on_epoch=True,  prog_bar=True)     

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y = y.long()
        loss = F.cross_entropy(preds, y)
        self.test_acc(preds, y)
        self.log("test_loss", loss)        
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True,  prog_bar=True)   

    def configure_optimizers(self):
        print('Initiating optimizer with lr = ', self.lr)
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def make_data(dataname, val_split=0.2, few_shot=None, return_datasets=False, args={}):
    #dpath = os.path.join(VTAB_emb_path, NAME_MAP[dataname], f"trainval-{model}.hdf5")
    loaders  = {'train': None, 'valid': None, 'test': None}
    datasets = {'train': None, 'valid': None, 'test': None}
    dset = H5Dataset(
        os.path.join(args.VTAB_emb_path, NAME_MAP[dataname]),
        memory_cache=True,
        # window=5000,
        pattern = f"trainval-{args.embedding_model}", 
        #transform = transforms.Compose([transforms.ToTensor()]),
        datatype='embedding'
    )
    if val_split > 0.0:
        valid_set_size = int(len(dset) * val_split)
        train_set_size = len(dset) - valid_set_size
        train_dset, valid_dset = random_split( dset, [train_set_size, valid_set_size], 
                generator=torch.Generator().manual_seed(args.seed))
        datasets['valid'] = valid_dset
    else:
        train_dset = dset
    if few_shot:
        # few_shot_data needs dataset to have data and targets attributes
        train_dset.data = dset.data[train_dset.indices]
        train_dset.targets = dset.targets[train_dset.indices]
        train_dset = few_shot_data(train_dset, n_shot=few_shot, seed=args.seed, nist_ds=False) 
    datasets['train'] = train_dset

    # Check if we have a test set
    try:
        test_dset = H5Dataset(os.path.join(args.VTAB_emb_path, NAME_MAP[dataname]), 
                             memory_cache=True,
                             pattern = f"test-{args.embedding_model}",
                             #transform = transforms.ToTensor(),
                             datatype='embedding')    
    except:
        print('Could not find test set')

    loaders['train'] = DataLoader(train_dset, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=True)
    if datasets['valid']:
        loaders['valid'] = DataLoader(valid_dset, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    if datasets['test']:
        loaders['test'] = DataLoader(test_dset, batch_size=512, num_workers = args.num_workers, shuffle=False)
    if return_datasets:
        return loaders, datasets
    else:
        return loaders

def train_wrapper(dataname, model, train, valid, test=None, dirpath=None, args={}):
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(args.model_save_path, dataname),
        save_weights_only=True,
        filename=args.embedding_model + '-{epoch}-{val_acc:.2f}',
        monitor="val_acc", mode="max"
    )
    # Initialize a trainer
    trainer = Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=args.epochs,
        callbacks=[
                #TQDMProgressBar(refresh_rate=20), 
                RichProgressBar(refresh_rate=1),
                #PrintTableMetricsCallback(),
                EarlyStopping(monitor='val_acc', patience=3, mode='max'),
                checkpoint_cb]
    )

    # Train the model ⚡
    trainer.fit(model, train, valid)
    print(checkpoint_cb.best_model_path)
    model.load_state_dict(torch.load(checkpoint_cb.best_model_path)['state_dict'])
    #model.save(f'models/{dataname}_acc_{checkpoint_cb.best_model_score:.2f}.pt')
    if test is not None:
        acc = trainer.test(model, test)[0]['test_acc_epoch']
    else:
        acc = checkpoint_cb.best_model_score
        print('Sure you want to test on the validation set?')
        breakpoint()
    torch.save(model.state_dict(), os.path.join(args.model_save_path, f'{dataname}_{args.embedding_model}_acc_{acc:.4f}.pt'))
    return model
        


if __name__ == "__main__":
    # class args:
    #     VTAB_emb_path='/data/VTAB-embeddings/'
    #     model_save_path = 'models'
    #     arch = 'MLP'
    #     dataname = 'DMLab'
    #     batch_size = 256 if torch.cuda.is_available() else 64
    #     num_workers = 64
    #     epochs = 3[0
    #     embedding_model = 'beit_base'
    #     device = 2
    #     transfer_to = 'Camelyon'
    #     transfer_shots = 5
    #     seed = 42

    parser = argparse.ArgumentParser(description='Main scropt for batchOT',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--VTAB_emb_path', type=str, default='/data/VTAB-embeddings/')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--arch', type=str, default='MLP')
    parser.add_argument('--dataname', type=str, default='ImageNet')
    parser.add_argument('--batch_size', type=int, default=256 if torch.cuda.is_available() else 64)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--embedding_model', type=str, default='beit_base')
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--transfer_to', type=str, default='Camelyon')
    parser.add_argument('--transfer_shots', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model = args.embedding_model
    #model = 'vit_large'
    dataname = args.dataname
    assert dataname in NAME_MAP.keys(), f"Dataset {dataname} not found"

    input_dim = 512 if model == 'resnet-18' else 768

    # Init our model
    model = LTMLP(input_dim=input_dim, dropout=0.2, nclass=DATASET_NCLASSES[dataname])

    # Check if pretrained model exists
    model_paths = glob.glob(os.path.join(args.model_save_path, f"{dataname}_{args.embedding_model}_acc*.pt"))

    loaders = make_data(dataname, args=args)

    if model_paths:
        model_accs = {k: float(re.search(r'.*acc_(.*?).pt', k).group(1)) for k in model_paths}
        best_model_path, best_model_acc = list(sorted(model_accs.items(), key=lambda item: float(item[1]), reverse=True))[0]
        # Ask user for confirmation to overwrite
        print(f"Pretrained model for {dataname} (acc={best_model_acc}) already exists.")
        overwrite = input("Train from scratch (+ potentially overwrite)? (y/n): ")

    if not model_paths or overwrite == 'y':
        print(f"Will pre-train {dataname} model from scratch...")
        model = train_wrapper(dataname, model, **loaders, args=args)
    else:
        print(f"Loading model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path))
        tester = Trainer(accelerator="auto", auto_select_gpus=True,
                        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        )       
        if loaders['test_dl']:
            acc = tester.test(model, loaders['test_dl'])[0]['test_acc_epoch']
            print(f"Actual test accuracy: {acc} (vs claimed={best_model_acc})")
        else:
            print("No test set found. Skipping test accuracy evaluation.")
            breakpoint()

    if args.transfer_to:   
        print(f"Will fine-tune model on {args.transfer_to} using {args.transfer_shots}-shot data...")
        #dataname = 'RandomInit'
        if dataname == 'RandomInit':
            model = LTMLP(input_dim=input_dim, dropout=0.2, nclass=DATASET_NCLASSES[args.transfer_to])
        else:
            ### Method 1: use already loaded model and replace last layer
            transferred_layers = [a for a in model.net.children()][:-1]
            transferred_layers.append(torch.nn.Linear(100, DATASET_NCLASSES[args.transfer_to]))
            model.net = torch.nn.Sequential(*transferred_layers)
        model.lr = 1e-4
        ### Method 2: create new model 
        #model = LTMLP(input_dim=768, dropout=0.2, nclass=DATASET_NCLASSES[dataname])

        #breakpoint()
        loaders = make_data(args.transfer_to, few_shot=args.transfer_shots,  args=args)
        model = train_wrapper(dataname + '_to_' + args.transfer_to, model, **loaders, args=args)
    
    breakpoint()


