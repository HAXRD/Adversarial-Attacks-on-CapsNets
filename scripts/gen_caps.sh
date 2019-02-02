#!/bin/bash

# Variables
SCRIPTS_DIR=scripts 
MODE=gen_adv
TYPE=caps

declare -a datasets=("mnist" "fashion_mnist" "svhn" "cifar10")

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
        echo "$TYPE/$dataset/$MODE/$script_type.sh"
    done
    echo ""
done