
#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10000M        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=train/outs/basic-%N-%j.out  # %N for node name, %j for jobID

source ~/tfp363/bin/activate

python ~/Adversarial-Attack-on-CapsNets/experiment.py --total_batch_size=200 --mode=train --model=cnn --data_dir=/home/xuc/Adversarial-Attack-on-CapsNets/dataset/mnist/ --dataset=mnist --max_epochs=1000 --summary_dir=/home/xuc/projects/def-sageev/xuc/final/cnn/mnist

# debug
# python ~/Adversarial-Attack-on-CapsNets/experiment.py --total_batch_size=200 --mode=train --model=cnn --data_dir=/home/xuc/Adversarial-Attack-on-CapsNets/data/mnist/ --dataset=mnist --max_epochs=6 --save_epochs=2 --summary_dir=/home/xuc/projects/def-sageev/xuc/debug/cnn/mnist

#######################################
mkdir -p scripts/caps/mnist
mkdir -p scripts/caps/fashion_mnist
mkdir -p scripts/caps/svhn
mkdir -p scripts/caps/cifar10

mkdir -p scripts/cnn/mnist
mkdir -p scripts/cnn/fashion_mnist
mkdir -p scripts/cnn/svhn
mkdir -p scripts/cnn/cifar10

mkdir -p scripts/caps_r/mnist
mkdir -p scripts/caps_r/fashion_mnist
mkdir -p scripts/caps_r/svhn
mkdir -p scripts/caps_r/cifar10
#######################################

######## train the model ########
# mnist train
sbatch scripts/caps/mnist/train.sh
sbatch scripts/caps_r/mnist/train.sh
sbatch scripts/cnn/mnist/train.sh
# fashion_mnist train
sbatch scripts/caps/fashion_mnist/train.sh
sbatch scripts/caps_r/fashion_mnist/train.sh
sbatch scripts/cnn/fashion_mnist/train.sh
# svhn
sbatch scripts/caps/svhn/train.sh
sbatch scripts/caps_r/svhn/train.sh
sbatch scripts/cnn/svhn/train.sh
# cifar10
sbatch scripts/caps/cifar10/train.sh
sbatch scripts/caps_r/cifar10/train.sh
sbatch scripts/cnn/cifar10/train.sh


python experiment.py --mode=gen_adv --data_dir=data/caps/mnist/ --dataset=mnist --summary_dir=summary/caps/mnist/Default/ --adversarial_method=BIM --epsilon=2 --iteration_n=1