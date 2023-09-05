# 20-shot
# python src/scripts/compare_synthetic_data.py fine_tune_dataset=MNIST load_epochs="[150, 200, 50, 50]" &
# python src/scripts/compare_synthetic_data.py fine_tune_dataset=EMNIST load_epochs="[90, 70, 100, 60]" &
# python src/scripts/compare_synthetic_data.py fine_tune_dataset=KMNIST load_epochs="[100, 100, 100, 100]" &
# python src/scripts/compare_synthetic_data.py fine_tune_dataset=USPS load_epochs="[100, 100, 100, 100]" &
# python src/scripts/compare_synthetic_data.py fine_tune_dataset=FMNIST load_epochs="[100, 100, 80, 35]"

# 5-shot
python src/scripts/compare_synthetic_data.py fine_tune_dataset=MNIST load_epochs="[50, 50, 35, 35]" &
python src/scripts/compare_synthetic_data.py fine_tune_dataset=EMNIST load_epochs="[50, 50, 50, 35]" &
python src/scripts/compare_synthetic_data.py fine_tune_dataset=KMNIST &
python src/scripts/compare_synthetic_data.py fine_tune_dataset=USPS &
python src/scripts/compare_synthetic_data.py fine_tune_dataset=FMNIST load_epochs="[50, 50, 40, 40]" &
python src/scripts/compare_synthetic_data.py fine_tune_dataset=MNISTM load_epochs="[50, 50, 40, 40, 30]"

# debug
# python src/scripts/compare_synthetic_data.py fine_tune_dataset=USPS retrain=true
