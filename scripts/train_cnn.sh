#!/bin/bash

# Variables
SCRIPTS_DIR=scripts 
MODE=train
TYPE=cnn

declare -a datasets=("mnist" "cifar10")

declare -a script_types=(
    "train" 
    "train_BIM_ep4_iter1" "train_BIM_ep8_iter1" "train_BIM_ep16_iter1"
    "train_BIM_ep1_iter4" "train_BIM_ep1_iter8" "train_BIM_ep1_iter16"
    "train_ILLCM_ep1_iter4" "train_ILLCM_ep1_iter8" "train_ILLCM_ep1_iter16")
declare -a script_all_types=(
    "train" 
    "train_BIM_ep2_iter1" "train_BIM_ep4_iter1" "train_BIM_ep8_iter1" "train_BIM_ep16_iter1"
    "train_BIM_ep1_iter2" "train_BIM_ep1_iter4" "train_BIM_ep1_iter8" "train_BIM_ep1_iter16"
    "train_ILLCM_ep1_iter2" "train_ILLCM_ep1_iter4" "train_ILLCM_ep1_iter8" "train_ILLCM_ep1_iter16")

for script_type in "${script_types[@]}"
do 
    for dataset in "${datasets[@]}"
    do
        sbatch $SCRIPTS_DIR/$TYPE/$dataset/$MODE/$script_type.sh
        echo "$SCRIPTS_DIR/$TYPE/$dataset/$MODE/o_$script_type.out"
    done
    echo ""
done