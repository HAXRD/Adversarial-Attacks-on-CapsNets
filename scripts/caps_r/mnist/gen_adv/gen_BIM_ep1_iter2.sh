#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=15000M        # memory per node
#SBATCH --time=0-01:00      # time (DD-HH:MM)
#SBATCH --output=scripts/caps_r/mnist/gen_adv/o_gen_BIM_ep1_iter2.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

REPO_DIR=/home/xuc/Adversarial-Attack-on-CapsNets
SUMMARY_DIR=/home/xuc/scratch/xuc/summary/

MODEL=caps_r
DATASET=mnist

ADVERSARIAL_METHOD=BIM 
EPSILON=1
ITERATION_N=2

python $REPO_DIR/experiment.py --mode=gen_adv --data_dir=$REPO_DIR/data/$MODEL/$DATASET --dataset=$DATASET --adversarial_method=$ADVERSARIAL_METHOD --epsilon=$EPSILON --iteration_n=$ITERATION_N --summary_dir=$SUMMARY_DIR/$MODEL/$DATASET/Default/
