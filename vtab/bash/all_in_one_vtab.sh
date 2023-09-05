#!/bin/bash

DATA_PATH="/path/to/save/data"

# VTAB datasets:
# Retinopathy: https://www.kaggle.com/c/diabetic-retinopathy-detection/data
# Camelyon: https://camelyon17.grand-challenge.org/Data/
# ImageNet: http://image-net.org/download-images
# DMLab:
# sNORB-Azim:
# sNORB-Elev:
# OxfordIIITPet: https://www.robots.ox.ac.uk/~vgg/data/pets/
# Flowers102: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
# Caltech101: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
# Caltech256: http://www.vision.caltech.edu/Image_Datasets/Caltech256/
# Resisc45: https://www.robots.ox.ac.uk/~vgg/data/resisc45/
# Sun397: https://vision.princeton.edu/projects/2010/SUN/
# KITTI: http://www.cvlibs.net/datasets/kitti/
# DTD: https://www.robots.ox.ac.uk/~vgg/data/dtd/
# EuroSAT:

REFDSET="OxfordIIITPet"
TRNDSETS=(
    "Caltech101"    # Natural
    # "CIFAR100"      # Natural
    "DTD"           # Natural
    "Flowers102"    # Natural
    #"OxfordIIITPet" # Natural
    # "SUN397"        # Natural
    # "SVHN"          # Natural
    #"ImageNet"      # Natural
    # "EuroSAT"       # Specialized
    # "Resisc45"      # Specialized
    # "Camelyon"      # Specialized
    # "Retinopathy"   # Specialized
    # "Clevr-Count"   # Structured
    # "Clevr-Dist"    # Structured
    # "dSpr-Loc"
    # "dSpr-Ori"
    # "sNORB-Azim"
    # "sNORB-Elev"
    # "DMLab"
    # "KITTI"
)


ALLDSETS=("${TRNDSETS[@]}" "${REFDSET[@]}")


while getopts "s:" opt; do
    case $opt in
        s) IFS=',' read -ra SKIPS <<< "$OPTARG"
        #...
    esac
done
shift $((OPTIND -1))


for i in "${SKIPS[@]}"
do
    echo "Will skip step $i!"
done


#### Step 0: Embed using Masked Autoencoder
if [[ ! " ${SKIPS[@]} " =~ " 0 " ]]; then
    echo "Running step 0"
    python src/scripts/embed_mae.py --datasets "${TRNDSETS[@]}" --fold "train" --shuffle --device 0 --datapath "$DATA_PATH/"
    for i in {0..4}
    do
        python src/scripts/embed_mae.py --datasets "$REFDSET" --fold "train" --subsample 1000 --shuffle --device 0 --seed $i --datapath "$DATA_PATH/"
    done
    python src/scripts/embed_mae.py --datasets "${ALLDSETS[@]}" --fold "train800val200" --shuffle --device 0 --datapath "$DATA_PATH/"
    echo "Done with Step 0"
else
    echo "Skipping step 0"
fi


#### Step 1: Generate pseudo-labeled target dataset via knn on feature-only train fold and labeled train800val200 fold
if [[ ! " ${SKIPS[@]} " =~ " 1 " ]]; then
    echo "Running step 1"
    for i in {0..4}
    do
        python src/scripts/knn_vtab.py --ds_name "$REFDSET" --vtab_data_dir "$DATA_PATH/VTAB-mae-embeddings/" --num_neighbor 1 --labeled_fold "train1000_seed$i"
    done
    echo "Done with Step 1"
else
    echo "Skipping step 1"
fi


#### Step 2: Solve the barycentric projection from reference to train datasets
if [[ ! " ${SKIPS[@]} " =~ " 2 " ]]; then
    echo "Running step 2"
    for i in {0..4}
    do
        python src/scripts/vtab_bary_projection.py  --reference_ds_name "$REFDSET" \
                                                    --reference_ds_fold "knn_train1000_seed$i" \
                                                    --train_datasets_name "${TRNDSETS[@]}" \
                                                    --vtab_data_dir "$DATA_PATH/VTAB-mae-embeddings" \
                                                    --pf_ds_size "max_train_ds_size" \
                                                    --output_dir "$DATA_PATH/pushforward_datasets/vtab" \
                                                    --seed $i --device $(($i%4))
    done
    echo "Done with Step 2"
else
    echo "Skipping step 2"
fi


#### Step 3: Solve the best interpolation dataset
if [[ ! " ${SKIPS[@]} " =~ " 3 " ]]; then
    echo "Running step 3"
    for i in {0..4}
    do
        for weight_type in "optimal" #"uniform"
        do
            python src/scripts/vtab_synthetic_ds.py --reference_ds_name "$REFDSET" \
                                                    --reference_ds_fold "knn_train1000_seed$i" \
                                                    --pushforward_dataset_dir "$DATA_PATH/pushforward_datasets/vtab" \
                                                    --projection_dataset_dir "$DATA_PATH/projection_datasets/vtab" \
                                                    --train_datasets_name "${TRNDSETS[@]}" \
                                                    --vtab_data_dir "$DATA_PATH/VTAB-mae-embeddings" \
                                                    --seed $i --device $(($i%4)) \
                                                    --weight_type "$weight_type"
        done
    done
    echo "Done with Step 3"
else
    echo "Skipping step 3"
fi

### Step 4: Train and fine-tune the classifiers
if [[ ! " ${SKIPS[@]} " =~ " 4 " ]]; then
    echo "Running step 4"
    for i in {5..9}
    do
        # # # otddmap interpolation
        for weight_type in "optimal" "uniform"
        do
            python src/scripts/fulltrain_vtab.py \
                --interpolation_method "OTDD_map" \
                --reference_ds_name "$REFDSET" \
                --reference_ds_indices "$DATA_PATH/VTAB-mae-embeddings/oxford_iiit_pet/train1000_seed$i/indices.pt" \
                --train_ds_name "${TRNDSETS[@]}" \
                --train_ds_path '$DATA_PATH/projection_datasets/vtab' \
                --test_ds_path '$DATA_PATH/' \
                --limit_train_batches 1.0 \
                --limit_test_batches 1.0 \
                --from_scratch \
                --weight_type "$weight_type" \
                --gpus 4 --seed $i -es 5 -et 10 --transfer_learning
        done

        # mixup interpolation baseline
        for weight_type in "pooling" "optimal" "uniform"
        do
            python src/scripts/mixup_fulltrain_vtab.py \
                --reference_ds_name "$REFDSET" \
                --reference_ds_indices "$DATA_PATH/VTAB-mae-embeddings/oxford_iiit_pet/train1000_seed$i/indices.pt" \
                --train_ds_name "${TRNDSETS[@]}" \
                --train_ds_path '$DATA_PATH/' \
                --test_ds_path '$DATA_PATH/' \
                --limit_train_batches 1.0 \
                --limit_test_batches 1.0 \
                --from_scratch \
                --weight_type "$weight_type" \
                --gpus 1 --seed $i -es 5 -et 10
        done

        # baseline: train on each dataset individually
        BASELINES=("NONE" "${TRNDSETS[@]}")
        for TRNDSET in "${BASELINES[@]}"
        do
            python src/scripts/fulltrain_vtab.py \
                --reference_ds_name "$REFDSET" \
                --reference_ds_indices "$DATA_PATH/VTAB-mae-embeddings/oxford_iiit_pet/train1000_seed$i/indices.pt" \
                --train_ds_name "$TRNDSET" \
                --train_ds_path "$DATA_PATH/torchvision/" \
                --test_ds_path '$DATA_PATH/' \
                --limit_train_batches 1.0 \
                --limit_test_batches 1.0 \
                --from_scratch \
                --gpus 4 --seed $i -es 5 -et 10 --transfer_learning
        done

        # KNN interpolation
        python src/scripts/fulltrain_vtab.py \
            --interpolation_method "knn" \
            --reference_ds_name "$REFDSET" \
            --reference_ds_indices "$DATA_PATH/VTAB-mae-embeddings/oxford_iiit_pet/train1000_seed$i/indices.pt" \
            --train_ds_name "${TRNDSETS[@]}" \
            --train_ds_path '$DATA_PATH/baseline_knn_dataset/vtab' \
            --test_ds_path '$DATA_PATH/' \
            --limit_train_batches 1.0 \
            --limit_test_batches 1.0 \
            --from_scratch \
            --weight_type "None" \
            --gpus 4 --seed $i -es 5 -et 10 --transfer_learning

    done
    echo "Done with Step 4"
else
    echo "Skipping step 4"
fi

exit 0
