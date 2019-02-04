#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M        # memory per node
#SBATCH --time=0-00:10      # time (DD-HH:MM)
#SBATCH --output=scripts/cnn/cifar10/test-2-clean_in_f_adv_mdls/o_t2_ILLCM_ep1_iter4.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

REPO_DIR=/home/xuc/Adversarial-Attack-on-CapsNets
SUMMARY_DIR=/home/xuc/scratch/xuc/summary/

MODEL=cnn
DATASET=cifar10

ADVERSARIAL_METHOD=ILLCM 
EPSILON=1
ITERATION_N=4

TEST_FILE=test.npz

python $REPO_DIR/experiment.py --mode=test --adversarial_method=$ADVERSARIAL_METHOD --epsilon=$EPSILON --iteration_n=$ITERATION_N --summary_dir=$SUMMARY_DIR/$MODEL/$DATASET/$ADVERSARIAL_METHOD --load_test_path=$REPO_DIR/data/$MODEL/$DATASET/$TEST_FILE