#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M        # memory per node
#SBATCH --time=0-00:10      # time (DD-HH:MM)
#SBATCH --output=scripts/caps_r/cifar10/test-1-adv_in_f_clean_mdls/o_t1_ILLCM_ep1_iter8.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

REPO_DIR=/home/xuc/Adversarial-Attack-on-CapsNets
SUMMARY_DIR=/home/xuc/scratch/xuc/summary/

MODEL=caps_r
DATASET=cifar10

ADVERSARIAL_METHOD=ILLCM
EPSILON=1
ITERATION_N=8

TEST_FILE=test_$ADVERSARIAL_METHOD\_eps$EPSILON\_iter$ITERATION_N.npz

python $REPO_DIR/experiment.py --mode=test --summary_dir=$SUMMARY_DIR/$MODEL/$DATASET/Default/ --load_test_path=$REPO_DIR/data/$MODEL/$DATASET/$TEST_FILE
