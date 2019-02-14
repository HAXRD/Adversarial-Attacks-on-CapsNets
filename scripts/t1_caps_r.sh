#!/bin/bash

# Variables
SCRIPTS_DIR=scripts 
MODE=test-1-adv_in_f_clean_mdls
TYPE=caps_r

declare -a datasets=("mnist" "cifar10")

declare -a script_types=(
    "t1_BIM_ep4_iter1" "t1_BIM_ep8_iter1" "t1_BIM_ep16_iter1"
    "t1_BIM_ep1_iter4" "t1_BIM_ep1_iter8" "t1_BIM_ep1_iter16"
    "t1_ILLCM_ep1_iter4" "t1_ILLCM_ep1_iter8" "t1_ILLCM_ep1_iter16")
declare -a script_all_types=(
    "t1_BIM_ep2_iter1" "t1_BIM_ep4_iter1" "t1_BIM_ep8_iter1" "t1_BIM_ep16_iter1"
    "t1_BIM_ep1_iter2" "t1_BIM_ep1_iter4" "t1_BIM_ep1_iter8" "t1_BIM_ep1_iter16"
    "t1_ILLCM_ep1_iter2" "t1_ILLCM_ep1_iter4" "t1_ILLCM_ep1_iter8" "t1_ILLCM_ep1_iter16")

for script_type in "${script_types[@]}"
do 
    for dataset in "${datasets[@]}"
    do
        sbatch $SCRIPTS_DIR/$TYPE/$dataset/$MODE/$script_type.sh
        echo "$SCRIPTS_DIR/$TYPE/$dataset/$MODE/o_$script_type.out"
    done
    echo ""
done