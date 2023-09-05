# real ternary
# python src/scripts/otdd_ternary.py fine_tune_dataset=MNIST train_datasets="["EMNIST", "FMNIST", "USPS"]" load_epochs="[50, 200, 150]" &
# python src/scripts/otdd_ternary.py fine_tune_dataset=EMNIST train_datasets="["MNIST", "FMNIST", "USPS"]" load_epochs="[90, 100, 70]" &
# python src/scripts/otdd_ternary.py fine_tune_dataset=KMNIST train_datasets="["MNIST", "EMNIST", "USPS"]" load_epochs="[100, 100, 100]" &
# python src/scripts/otdd_ternary.py fine_tune_dataset=USPS train_datasets="["MNIST", "EMNIST", "KMNIST"]" load_epochs="[100, 100, 100]" &
# python src/scripts/otdd_ternary.py fine_tune_dataset=FMNIST train_datasets="["MNIST", "KMNIST", "USPS"]" load_epochs="[100, 80, 100]"

# to get the best interpolation parameter for 20-shot.
# python src/scripts/otdd_ternary.py fine_tune_dataset=MNIST full_dataset=true load_epochs="[150, 200, 80, 50]" &
# python src/scripts/otdd_ternary.py fine_tune_dataset=EMNIST full_dataset=true load_epochs="[90, 70, 100, 60]" &
# python src/scripts/otdd_ternary.py fine_tune_dataset=KMNIST full_dataset=true load_epochs="[100, 100, 100, 100]" &
# python src/scripts/otdd_ternary.py fine_tune_dataset=USPS full_dataset=true load_epochs="[100, 100, 100, 100]" &
# python src/scripts/otdd_ternary.py fine_tune_dataset=FMNIST full_dataset=true load_epochs="[100, 100, 80, 35]"

# to get the best interpolation parameter for 5-shot.
python src/scripts/otdd_ternary.py fine_tune_dataset=MNIST full_dataset=true load_epochs="[50, 50, 35, 35]" &
python src/scripts/otdd_ternary.py fine_tune_dataset=EMNIST full_dataset=true load_epochs="[50, 50, 50, 35]" &
python src/scripts/otdd_ternary.py fine_tune_dataset=KMNIST full_dataset=true&
python src/scripts/otdd_ternary.py fine_tune_dataset=USPS full_dataset=true &
python src/scripts/otdd_ternary.py fine_tune_dataset=FMNIST full_dataset=true load_epochs="[50, 50, 40, 40]"&
python src/scripts/otdd_ternary.py fine_tune_dataset=MNISTM full_dataset=true load_epochs="[50, 50, 40, 40, 30]"

# ----- discarded visualization combinations -----
# python src/scripts/otdd_ternary.py fine_tune_dataset=KMNIST train_datasets="["MNIST", "FMNIST", "USPS"]" load_epochs="[100, 100, 100]" &
# python src/scripts/otdd_ternary.py fine_tune_dataset=USPS train_datasets="["MNIST", "EMNIST", "KMNIST"]" load_epochs="[100, 100, 100]"
