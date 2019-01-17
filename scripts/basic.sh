
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

######## clean train the model ######## 10989467 - 10989500
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


######## FGSM ep2 train the model ######## 10989656 - 10989671
# mnist
sbatch scripts/caps/mnist/train_BIM_ep2_iter1.sh
sbatch scripts/caps_r/mnist/train_BIM_ep2_iter1.sh
sbatch scripts/cnn/mnist/train_BIM_ep2_iter1.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_BIM_ep2_iter1.sh
sbatch scripts/caps_r/fashion_mnist/train_BIM_ep2_iter1.sh
sbatch scripts/cnn/fashion_mnist/train_BIM_ep2_iter1.sh
# svhn
sbatch scripts/caps/svhn/train_BIM_ep2_iter1.sh
sbatch scripts/caps_r/svhn/train_BIM_ep2_iter1.sh
sbatch scripts/cnn/svhn/train_BIM_ep2_iter1.sh
# cifar10
sbatch scripts/caps/cifar10/train_BIM_ep2_iter1.sh
sbatch scripts/caps_r/cifar10/train_BIM_ep2_iter1.sh
sbatch scripts/cnn/cifar10/train_BIM_ep2_iter1.sh

######## FGSM ep4 train the model ######## 10989690 - 10989703
# mnist
sbatch scripts/caps/mnist/train_BIM_ep4_iter1.sh
sbatch scripts/caps_r/mnist/train_BIM_ep4_iter1.sh
sbatch scripts/cnn/mnist/train_BIM_ep4_iter1.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_BIM_ep4_iter1.sh
sbatch scripts/caps_r/fashion_mnist/train_BIM_ep4_iter1.sh
sbatch scripts/cnn/fashion_mnist/train_BIM_ep4_iter1.sh
# svhn
sbatch scripts/caps/svhn/train_BIM_ep4_iter1.sh
sbatch scripts/caps_r/svhn/train_BIM_ep4_iter1.sh
sbatch scripts/cnn/svhn/train_BIM_ep4_iter1.sh
# cifar10
sbatch scripts/caps/cifar10/train_BIM_ep4_iter1.sh
sbatch scripts/caps_r/cifar10/train_BIM_ep4_iter1.sh
sbatch scripts/cnn/cifar10/train_BIM_ep4_iter1.sh

######## FGSM ep8 train the model ######## 10989705 - 10989716
# mnist
sbatch scripts/caps/mnist/train_BIM_ep8_iter1.sh
sbatch scripts/caps_r/mnist/train_BIM_ep8_iter1.sh
sbatch scripts/cnn/mnist/train_BIM_ep8_iter1.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_BIM_ep8_iter1.sh
sbatch scripts/caps_r/fashion_mnist/train_BIM_ep8_iter1.sh
sbatch scripts/cnn/fashion_mnist/train_BIM_ep8_iter1.sh
# svhn
sbatch scripts/caps/svhn/train_BIM_ep8_iter1.sh
sbatch scripts/caps_r/svhn/train_BIM_ep8_iter1.sh
sbatch scripts/cnn/svhn/train_BIM_ep8_iter1.sh
# cifar10
sbatch scripts/caps/cifar10/train_BIM_ep8_iter1.sh
sbatch scripts/caps_r/cifar10/train_BIM_ep8_iter1.sh
sbatch scripts/cnn/cifar10/train_BIM_ep8_iter1.sh

######## FGSM ep16 train the model ######## 10989717 - 10989728
# mnist
sbatch scripts/caps/mnist/train_BIM_ep16_iter1.sh
sbatch scripts/caps_r/mnist/train_BIM_ep16_iter1.sh
sbatch scripts/cnn/mnist/train_BIM_ep16_iter1.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_BIM_ep16_iter1.sh
sbatch scripts/caps_r/fashion_mnist/train_BIM_ep16_iter1.sh
sbatch scripts/cnn/fashion_mnist/train_BIM_ep16_iter1.sh
# svhn
sbatch scripts/caps/svhn/train_BIM_ep16_iter1.sh
sbatch scripts/caps_r/svhn/train_BIM_ep16_iter1.sh
sbatch scripts/cnn/svhn/train_BIM_ep16_iter1.sh
# cifar10
sbatch scripts/caps/cifar10/train_BIM_ep16_iter1.sh
sbatch scripts/caps_r/cifar10/train_BIM_ep16_iter1.sh
sbatch scripts/cnn/cifar10/train_BIM_ep16_iter1.sh

######## BIM ep1_iter2 train the model ######## 10989729 - 10989740
# mnist
sbatch scripts/caps/mnist/train_BIM_ep1_iter2.sh
sbatch scripts/caps_r/mnist/train_BIM_ep1_iter2.sh
sbatch scripts/cnn/mnist/train_BIM_ep1_iter2.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_BIM_ep1_iter2.sh
sbatch scripts/caps_r/fashion_mnist/train_BIM_ep1_iter2.sh
sbatch scripts/cnn/fashion_mnist/train_BIM_ep1_iter2.sh
# svhn
sbatch scripts/caps/svhn/train_BIM_ep1_iter2.sh
sbatch scripts/caps_r/svhn/train_BIM_ep1_iter2.sh
sbatch scripts/cnn/svhn/train_BIM_ep1_iter2.sh
# cifar10
sbatch scripts/caps/cifar10/train_BIM_ep1_iter2.sh
sbatch scripts/caps_r/cifar10/train_BIM_ep1_iter2.sh
sbatch scripts/cnn/cifar10/train_BIM_ep1_iter2.sh

######## BIM ep1_iter4 train the model ########  10989745  - 10989759
# mnist
sbatch scripts/caps/mnist/train_BIM_ep1_iter4.sh
sbatch scripts/caps_r/mnist/train_BIM_ep1_iter4.sh
sbatch scripts/cnn/mnist/train_BIM_ep1_iter4.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_BIM_ep1_iter4.sh
sbatch scripts/caps_r/fashion_mnist/train_BIM_ep1_iter4.sh
sbatch scripts/cnn/fashion_mnist/train_BIM_ep1_iter4.sh
# svhn
sbatch scripts/caps/svhn/train_BIM_ep1_iter4.sh
sbatch scripts/caps_r/svhn/train_BIM_ep1_iter4.sh
sbatch scripts/cnn/svhn/train_BIM_ep1_iter4.sh
# cifar10
sbatch scripts/caps/cifar10/train_BIM_ep1_iter4.sh
sbatch scripts/caps_r/cifar10/train_BIM_ep1_iter4.sh
sbatch scripts/cnn/cifar10/train_BIM_ep1_iter4.sh


######## BIM ep1_iter8 train the model ######## 10989822 - 10989857
# mnist
sbatch scripts/caps/mnist/train_BIM_ep1_iter8.sh
sbatch scripts/caps_r/mnist/train_BIM_ep1_iter8.sh
sbatch scripts/cnn/mnist/train_BIM_ep1_iter8.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_BIM_ep1_iter8.sh
sbatch scripts/caps_r/fashion_mnist/train_BIM_ep1_iter8.sh
sbatch scripts/cnn/fashion_mnist/train_BIM_ep1_iter8.sh
# svhn
sbatch scripts/caps/svhn/train_BIM_ep1_iter8.sh
sbatch scripts/caps_r/svhn/train_BIM_ep1_iter8.sh
sbatch scripts/cnn/svhn/train_BIM_ep1_iter8.sh
# cifar10
sbatch scripts/caps/cifar10/train_BIM_ep1_iter8.sh
sbatch scripts/caps_r/cifar10/train_BIM_ep1_iter8.sh
sbatch scripts/cnn/cifar10/train_BIM_ep1_iter8.sh

######## BIM ep1_iter16 train the model ######## 10989868 - 10989897
# mnist
sbatch scripts/caps/mnist/train_BIM_ep1_iter16.sh
sbatch scripts/caps_r/mnist/train_BIM_ep1_iter16.sh
sbatch scripts/cnn/mnist/train_BIM_ep1_iter16.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_BIM_ep1_iter16.sh
sbatch scripts/caps_r/fashion_mnist/train_BIM_ep1_iter16.sh
sbatch scripts/cnn/fashion_mnist/train_BIM_ep1_iter16.sh
# svhn
sbatch scripts/caps/svhn/train_BIM_ep1_iter16.sh
sbatch scripts/caps_r/svhn/train_BIM_ep1_iter16.sh
sbatch scripts/cnn/svhn/train_BIM_ep1_iter16.sh
# cifar10
sbatch scripts/caps/cifar10/train_BIM_ep1_iter16.sh
sbatch scripts/caps_r/cifar10/train_BIM_ep1_iter16.sh
sbatch scripts/cnn/cifar10/train_BIM_ep1_iter16.sh

######## ILLCM ep1_iter2 train the model ######## 10989909 - 10989920
# mnist
sbatch scripts/caps/mnist/train_ILLCM_ep1_iter2.sh
sbatch scripts/caps_r/mnist/train_ILLCM_ep1_iter2.sh
sbatch scripts/cnn/mnist/train_ILLCM_ep1_iter2.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_ILLCM_ep1_iter2.sh
sbatch scripts/caps_r/fashion_mnist/train_ILLCM_ep1_iter2.sh
sbatch scripts/cnn/fashion_mnist/train_ILLCM_ep1_iter2.sh
# svhn
sbatch scripts/caps/svhn/train_ILLCM_ep1_iter2.sh
sbatch scripts/caps_r/svhn/train_ILLCM_ep1_iter2.sh
sbatch scripts/cnn/svhn/train_ILLCM_ep1_iter2.sh
# cifar10
sbatch scripts/caps/cifar10/train_ILLCM_ep1_iter2.sh
sbatch scripts/caps_r/cifar10/train_ILLCM_ep1_iter2.sh
sbatch scripts/cnn/cifar10/train_ILLCM_ep1_iter2.sh

######## ILLCM ep1_iter4 train the model ######## 10989925 - 10989941
# mnist
sbatch scripts/caps/mnist/train_ILLCM_ep1_iter4.sh
sbatch scripts/caps_r/mnist/train_ILLCM_ep1_iter4.sh
sbatch scripts/cnn/mnist/train_ILLCM_ep1_iter4.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_ILLCM_ep1_iter4.sh
sbatch scripts/caps_r/fashion_mnist/train_ILLCM_ep1_iter4.sh
sbatch scripts/cnn/fashion_mnist/train_ILLCM_ep1_iter4.sh
# svhn
sbatch scripts/caps/svhn/train_ILLCM_ep1_iter4.sh
sbatch scripts/caps_r/svhn/train_ILLCM_ep1_iter4.sh
sbatch scripts/cnn/svhn/train_ILLCM_ep1_iter4.sh
# cifar10
sbatch scripts/caps/cifar10/train_ILLCM_ep1_iter4.sh
sbatch scripts/caps_r/cifar10/train_ILLCM_ep1_iter4.sh
sbatch scripts/cnn/cifar10/train_ILLCM_ep1_iter4.sh

######## ILLCM ep1_iter8 train the model ######## 10989943 - 10989954
# mnist
sbatch scripts/caps/mnist/train_ILLCM_ep1_iter8.sh
sbatch scripts/caps_r/mnist/train_ILLCM_ep1_iter8.sh
sbatch scripts/cnn/mnist/train_ILLCM_ep1_iter8.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_ILLCM_ep1_iter8.sh
sbatch scripts/caps_r/fashion_mnist/train_ILLCM_ep1_iter8.sh
sbatch scripts/cnn/fashion_mnist/train_ILLCM_ep1_iter8.sh
# svhn
sbatch scripts/caps/svhn/train_ILLCM_ep1_iter8.sh
sbatch scripts/caps_r/svhn/train_ILLCM_ep1_iter8.sh
sbatch scripts/cnn/svhn/train_ILLCM_ep1_iter8.sh
# cifar10
sbatch scripts/caps/cifar10/train_ILLCM_ep1_iter8.sh
sbatch scripts/caps_r/cifar10/train_ILLCM_ep1_iter8.sh
sbatch scripts/cnn/cifar10/train_ILLCM_ep1_iter8.sh


######## ILLCM ep1_iter16 train the model ######## 10989956 - 10989967
# mnist
sbatch scripts/caps/mnist/train_ILLCM_ep1_iter16.sh
sbatch scripts/caps_r/mnist/train_ILLCM_ep1_iter16.sh
sbatch scripts/cnn/mnist/train_ILLCM_ep1_iter16.sh
# fashion_mnist
sbatch scripts/caps/fashion_mnist/train_ILLCM_ep1_iter16.sh
sbatch scripts/caps_r/fashion_mnist/train_ILLCM_ep1_iter16.sh
sbatch scripts/cnn/fashion_mnist/train_ILLCM_ep1_iter16.sh
# svhn
sbatch scripts/caps/svhn/train_ILLCM_ep1_iter16.sh
sbatch scripts/caps_r/svhn/train_ILLCM_ep1_iter16.sh
sbatch scripts/cnn/svhn/train_ILLCM_ep1_iter16.sh
# cifar10
sbatch scripts/caps/cifar10/train_ILLCM_ep1_iter16.sh
sbatch scripts/caps_r/cifar10/train_ILLCM_ep1_iter16.sh
sbatch scripts/cnn/cifar10/train_ILLCM_ep1_iter16.sh

python experiment.py --mode=gen_adv --data_dir=data/caps/mnist/ --dataset=mnist --summary_dir=summary/caps/mnist/Default/ --adversarial_method=BIM --epsilon=2 --iteration_n=1