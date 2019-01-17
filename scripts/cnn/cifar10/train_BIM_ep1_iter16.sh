#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=30000M        # memory per node
#SBATCH --time=0-24:00      # time (DD-HH:MM)
#SBATCH --output=scripts/cnn/cifar10/o_train_BIM_ep1_iter16.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

python ~/Adversarial-Attack-on-CapsNets/experiment.py --data_dir=/home/xuc/Adversarial-Attack-on-CapsNets/data/cnn/cifar10/ --dataset=cifar10 --adversarial_method=BIM --epsilon=1 --iteration_n=16 --summary_dir=/home/xuc/projects/def-sageev/xuc/AD/summary/ --model=cnn 