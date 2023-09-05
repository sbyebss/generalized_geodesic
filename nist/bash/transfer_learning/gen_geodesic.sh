python src/scripts/gen_geodesic.py fine_tune_dataset=MNIST train_datasets="["EMNIST", "FMNIST", "USPS"]" load_epochs="[50, 200, 150]" &
python src/scripts/gen_geodesic.py fine_tune_dataset=EMNIST train_datasets="["MNIST", "FMNIST", "USPS"]" load_epochs="[90, 100, 70]" &
python src/scripts/gen_geodesic.py fine_tune_dataset=KMNIST train_datasets="["MNIST", "EMNIST", "USPS"]" load_epochs="[100, 100, 100]" &
python src/scripts/gen_geodesic.py fine_tune_dataset=USPS train_datasets="["MNIST", "EMNIST", "KMNIST"]" load_epochs="[100, 100, 100]" &
python src/scripts/gen_geodesic.py fine_tune_dataset=FMNIST train_datasets="["MNIST", "KMNIST", "USPS"]" load_epochs="[100, 80, 100]"

# other considerations:
# python src/scripts/gen_geodesic.py fine_tune_dataset=KMNIST train_datasets="["MNIST", "FMNIST", "USPS"]" load_epochs="[100, 100, 100]"

# python src/scripts/gen_geodesic.py fine_tune_dataset=USPS train_datasets="["MNIST", "EMNIST", "FMNIST"]" load_epochs="[100, 100, 100]"
