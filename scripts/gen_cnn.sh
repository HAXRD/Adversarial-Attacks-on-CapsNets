#!/bin/bash

# Variables
SCRIPTS_DIR=scripts 
MODE=gen_adv
TYPE=cnn

declare -a datasets=("mnist" "cifar10")

declare -a script_types=(
    "gen_BIM_ep4_iter1" "gen_BIM_ep8_iter1" "gen_BIM_ep16_iter1"
    "gen_BIM_ep1_iter4" "gen_BIM_ep1_iter8" "gen_BIM_ep1_iter16"
    "gen_ILLCM_ep1_iter4" "gen_ILLCM_ep1_iter8" "gen_ILLCM_ep1_iter16")
declare -a script_all_types=(
    "gen_BIM_ep2_iter1" "gen_BIM_ep4_iter1" "gen_BIM_ep8_iter1" "gen_BIM_ep16_iter1"
    "gen_BIM_ep1_iter2" "gen_BIM_ep1_iter4" "gen_BIM_ep1_iter8" "gen_BIM_ep1_iter16"
    "gen_ILLCM_ep1_iter2" "gen_ILLCM_ep1_iter4" "gen_ILLCM_ep1_iter8" "gen_ILLCM_ep1_iter16")

for script_type in "${script_types[@]}"
do 
    for dataset in "${datasets[@]}"
    do
        sbatch $SCRIPTS_DIR/$TYPE/$dataset/$MODE/$script_type.sh
        echo "$SCRIPTS_DIR/$TYPE/$dataset/$MODE/o_$script_type.out"
    done
    echo ""
done