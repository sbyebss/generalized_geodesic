# VTAB Dataset Experiments

This repository contains experiments conducted on the VTAB datasets, focusing on the few-shot classification task. The goal is to achieve classification accuracy with limited labeled data in the test dataset domain.

## Install

1. Install [poetry](https://python-poetry.org/docs/).

2. Install the required packages:

```bash
pip install -e .
pip install --no-deps -r requirements.txt
```

3. Ensure you have `PyTorch >= 1.12` installed.

Update the `DATA_PATH` variable in the `bash/all_in_one_vtab.sh` file to point to your downloaded data location.

## Reproduce results

To reproduce the results, simply execute:

```bash
bash bash/all_in_one_vtab.sh
```

## Pipeline Overview

The pipeline consists of several stages, primarily focusing on data interpolation and classifier training. Note that we only use OTDD barycentric projection for solving OTDD map, but not OTDD neural map.

### Data Preparation

1. **Embedding with Masked Autoencoder**:

   - Convert VTAB images into the latent space of the [Masked Autoencoder](https://arxiv.org/abs/2111.06377).

2. **k-NN Pseudo-label Generation**:

   - For the few-shot labeled test dataset, use k-NN to generate pseudo-labels.
   - This creates a reference dataset, enriching the labeled data in the test domain.
   - This enriched data aids in solving the OTDD maps.

3. **Barycentric Projection**:

   - Compute the OTDD maps from the reference dataset to the training datasets.
   - These maps are crucial for generating synthetic data on the generalized geodesic.

4. **Optimal Interpolation Dataset Generation**:
   - Determine the best interpolation dataset for training.
   - This involves solving a quadratic programming problem to obtain the optimal interpolation parameter.

### Classifier Training

1. **Our Method - OTDD Map Interpolation**:

   - Choose between optimal or uniform interpolation parameters.

2. **Baseline - Mixup Interpolation**:

   - Options include optimal, uniform interpolation parameters, or pooling all images together.

3. **Baseline - Individual Dataset Training**:

   - Train the classifier on each training dataset individually, followed by fine-tuning on the test dataset.

4. **Baseline - Sub-pooling Dataset**:
   - Construct a sub-pooling dataset using k-NN and use it as the training dataset.
