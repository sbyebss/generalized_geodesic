# \*NIST datasets experiments

This repository contains experiments conducted on various \*NIST datasets. The primary focus is on improving few-shot classification test accuracy using a synthetic training dataset on the generalized geodesic.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Reproducing Results](#reproducing-results)
- [Pipeline Overview](#pipeline-overview)
- [Detailed Steps](#detailed-steps)
  - [Data Preparation](#data-preparation)
  - [Classifier Training](#classifier-training)
  - [Additional Evaluation](#additional-evaluations)

## Install

Install [poetry](https://python-poetry.org/docs/).

```bash
pip install -e .
pip install --no-deps -r requirements.txt
```

Install `pytorch >= 1.12`

## Configuration

Update the dataset paths in the following configuration files:

- `configs/scripts/local/default.yaml`
- `configs/otdd_map/local/default.yaml`

## Reproduce results

To reproduce the experiment results, execute:

```bash
bash bash/all_in_one_nist.sh
```

## Pipeline Overview

The primary objective is to enhance few-shot classification test accuracy using a synthetic dataset on the generalized geodesic. The pipeline consists of several stages, with the assumption that the test dataset contains only a few-shot labeled data.

### Data Preparation for Interpolation

1. **KNN pseudo-label generation**: For few-shot labeled dataset, use KNN to generate pseudo-labels. This is useful for the generalized geodesic.

2. **Classifier training**: Train the classifier on the the mapped domain.

3. **Calculate the label distance**: Calculate label distances in advance, which will be used in training OTDD map as well as plotting OTDD heatmap.

4. **OTDD neural map training**: Given two labeled datasets, solve the OTDD neural map from source to target dataset.

### Transfer Learning: Training and Fine-tuning

1. **Training classifier on interpolated data**

2. **Fine-tune the classifier on the interpolated data**

### Additional evaluation

1. **Plot the OTDD from external to existing datasets (Figure 6)**

2. **Compare OTDD neural map, OTDD barycentric projection, and Mixup methods (Table 1)**

## Detailed Steps

### Data Preparation

#### KNN pseudo-label generation

Random seed is involved in this part, it shows up in which few-shot samples to take. You can find the configuration in `configs/scripts/knn_dataset.yaml`.

```bash
python src/scripts/knn_dataset.py -m dataset=MNIST,USPS,FMNIST,KMNIST,EMNIST,MNISTM seed=1,2,3,4,5 num_shot=5
```

<!-- python src/scripts/knn_dataset.py -m dataset=MNISTM seed=1,2,3,4,5 num_shot=200 num_shot=5 -->

After you run the command above, you should be able to find the fitted KNN labels are saved under the path `data/knn_results`.

#### Classifier training on mapped domain

No random seed in this part. You can find the configuration in `configs/scripts/pretrain_classifier.yaml`.

```bash
python src/scripts/pretrain_classifier.py all_datasets="["MNIST", "USPS", "FMNIST", "KMNIST", "EMNIST", "MNISTM"]" train_iters="[10000, 10000, 20000, 20000, 20000, 20000]"
```

<!-- python src/scripts/pretrain_classifier.py all_datasets="["MNISTM"]" train_iters="[10000]" -->

Or if you want to train on the few-shot dataset

```bash
# few-shot dataset needs much less iterations to avoid over-fitting
python src/scripts/pretrain_classifier.py -m few_shot=true all_datasets="["MNIST", "USPS", "FMNIST", "KMNIST", "EMNIST"]" train_iters="[2000, 2000, 2000, 2000, 5000]" seed=1,2,3,4,5 num_shot=5,20
```

<!-- python src/scripts/pretrain_classifier.py -m few_shot=true all_datasets="["MNISTM"]" train_iters="[2000]" seed=1,2,3,4,5 num_shot=5 -->

After you run the command above, you should be able to find the pretrained classifier are saved under the path `data/pretrain_classifier`.

#### Calculate the label distance

```bash
python src/scripts/calculate_otdd.py -m source=MNIST,USPS,FMNIST,KMNIST,EMNIST,MNISTM target=MNIST,USPS,FMNIST,KMNIST,EMNIST,MNISTM seed=1,2,3,4,5 source_few_shot=true,false target_few_shot=true,false num_shot=5
```

You can find the statistics of OTDD are saved under the path `data/otdd`.

#### OTDD neural map training

Modify the `gpus` in `config/trainer/default.yaml`. It should list the available GPUs.

All the commands are in folder `bash`.

### Classifier training

<!-- There are two ways: 1) interpolation between two datasets: McCann's interpolation; 2) interpolation among multiple datasets: generalized geodesic. -->

<!-- ### McCann's interpolation

```bash
python src/scripts/suff2insuff.py
``` -->
<!--
### generalized geodesic -->

Run everything

```bash
bash bash/transfer_learning/gen_geodesic.sh
```

Or run a single set:

```bash
python src/scripts/gen_geodesic.py fine_tune_dataset=MNIST train_datasets="["EMNIST", "FMNIST", "USPS"]" load_epochs="[50, 200, 150]"
```

<!-- `python src/scripts/gen_geodesic.py fine_tune_epoch=1 train_iteration=3` -->

### Additional evaluations

#### OTDD ternary plot (Figure 6)

Random seeds is included in the script. You can find the configuration in `configs/scripts/otdd_ternary.yaml`.

Run everything

```bash
bash bash/transfer_learning/otdd_ternary.sh
```

Or run your ternary combination below.

```bash
python src/scripts/otdd_ternary.py fine_tune_dataset=USPS train_datasets="["MNIST", "EMNIST", "KMNIST"]" load_epochs="[200, 200, 200]"
```

<!-- python src/scripts/otdd_ternary.py fine_tune_dataset=MNISTM full_dataset=true load_epochs="[50, 50, 40, 40, 30]" -->

After you run the command above, you should be able to find the ternary plot of this set is saved under the path `logs/otdd_ternary_transport_metric/external_USPS`.

#### Compare with mixup and barycentric projection (Table 1)

```bash
bash bash/transfer_learning/compare.sh
```
