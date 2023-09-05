# KNN pseudo-label generation
python src/scripts/knn_dataset.py -m dataset=MNIST,USPS,FMNIST,KMNIST,EMNIST,MNISTM seed=1,2,3,4,5 num_shot=5

# Classifier training on mapped domain
python src/scripts/pretrain_classifier.py all_datasets="["MNIST", "USPS", "FMNIST", "KMNIST", "EMNIST", "MNISTM"]" train_iters="[10000, 10000, 20000, 20000, 20000, 20000]"

# Classifier training on few-shot dataset directly
python src/scripts/pretrain_classifier.py -m few_shot=true all_datasets="["MNIST", "USPS", "FMNIST", "KMNIST", "EMNIST"]" train_iters="[2000, 2000, 2000, 2000, 5000]" seed=1,2,3,4,5 num_shot=5

# Calculate the label distance
python src/scripts/calculate_otdd.py -m source=MNIST,USPS,FMNIST,KMNIST,EMNIST,MNISTM target=MNIST,USPS,FMNIST,KMNIST,EMNIST,MNISTM seed=1,2,3,4,5 source_few_shot=true,false target_few_shot=true,false num_shot=5

# Neural net map training
python run.py -m name=E2X_few_shot mode=paper experiment=nist datamodule.source.dataset=EMNIST datamodule.target.dataset=FMNIST,USPS,MNIST,KMNIST seed=1,2,3,4,5 logger=wandb

python run.py -m name=M2X_few_shot mode=paper experiment=nist datamodule.source.dataset=MNIST datamodule.target.dataset=FMNIST,USPS,EMNIST,KMNIST seed=1,2,3,4,5 logger=wandb

python run.py -m name=K2X_few_shot mode=paper experiment=nist datamodule.source.dataset=KMNIST datamodule.target.dataset=FMNIST,USPS,EMNIST,MNIST seed=1,2,3,4,5 logger=wandb

python run.py -m name=U2X_few_shot mode=paper experiment=nist datamodule.source.dataset=USPS datamodule.target.dataset=FMNIST,KMNIST,EMNIST,MNIST seed=1,2,3,4,5 logger=wandb

python run.py -m name=F2X_few_shot mode=paper experiment=nist datamodule.source.dataset=FMNIST datamodule.target.dataset=USPS,KMNIST,EMNIST,MNIST seed=1,2,3,4,5 logger=wandb

python run.py -m name=C2X_few_shot mode=paper experiment=nist datamodule.source.dataset=MNISTM datamodule.target.dataset=USPS,KMNIST,EMNIST,MNIST,FMNIST seed=1,2,3,4,5 datamodule.source.channel=3 logger=wandb

# OTDD solving the best interpolation parameter
python src/scripts/otdd_ternary.py fine_tune_dataset=MNIST load_epochs="[50, 50, 35, 35]" &
python src/scripts/otdd_ternary.py fine_tune_dataset=EMNIST load_epochs="[50, 50, 50, 35]" &
python src/scripts/otdd_ternary.py fine_tune_dataset=KMNIST &
python src/scripts/otdd_ternary.py fine_tune_dataset=USPS &
python src/scripts/otdd_ternary.py fine_tune_dataset=FMNIST load_epochs="[50, 50, 40, 40]"&
python src/scripts/otdd_ternary.py fine_tune_dataset=MNISTM load_epochs="[50, 50, 40, 40, 30]"

### Compare with mixup and barycentric projection
python src/scripts/compare_synthetic_data.py fine_tune_dataset=MNIST load_epochs="[50, 50, 35, 35]" &
python src/scripts/compare_synthetic_data.py fine_tune_dataset=EMNIST load_epochs="[50, 50, 50, 35]" &
python src/scripts/compare_synthetic_data.py fine_tune_dataset=KMNIST &
python src/scripts/compare_synthetic_data.py fine_tune_dataset=USPS &
python src/scripts/compare_synthetic_data.py fine_tune_dataset=FMNIST load_epochs="[50, 50, 40, 40]" &
python src/scripts/compare_synthetic_data.py fine_tune_dataset=MNISTM load_epochs="[50, 50, 40, 40, 30]"
