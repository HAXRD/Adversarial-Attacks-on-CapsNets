#!/bin/bash

# Variables
SCRIPTS_DIR=scripts 
MODE=test-4-adv-transfer
TYPE=cnn

declare -a datasets=("mnist")

declare -a script_all_types=(
    "t4_BIM_ep2_iter1" "t4_BIM_ep4_iter1" "t4_BIM_ep8_iter1" "t4_BIM_ep16_iter1"
    "t4_BIM_ep1_iter2" "t4_BIM_ep1_iter4" "t4_BIM_ep1_iter8" "t4_BIM_ep1_iter16")

for script_type in "${script_all_types[@]}"
do 
    for dataset in "${datasets[@]}"
    do
        sbatch $SCRIPTS_DIR/$TYPE/$dataset/$MODE/$script_type.sh
        echo "$SCRIPTS_DIR/$TYPE/$dataset/$MODE/o_$script_type.out"
    done
    echo ""
done