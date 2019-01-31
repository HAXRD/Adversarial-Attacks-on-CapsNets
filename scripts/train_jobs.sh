#!/bin/bash

# Varaiables
SCRIPTS_DIR=scripts
MODE=train
######## $MODE clean ########
# mnist 
sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train.sh 
sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train.sh 
sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train.sh 
# fashion_mnist
sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train.sh 
sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train.sh 
sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train.sh 
# svhn
sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train.sh 
sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train.sh 
sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train.sh 
# cifar10
sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train.sh 
sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train.sh 
sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train.sh 


######## $MODE FGSM ep2 ########
# # mnist 
# sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_BIM_ep2_iter1.sh 
# sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_BIM_ep2_iter1.sh 
# sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_BIM_ep2_iter1.sh 
# # fashion_mnist
# sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_BIM_ep2_iter1.sh 
# sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_BIM_ep2_iter1.sh 
# sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_BIM_ep2_iter1.sh 
# # svhn
# sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_BIM_ep2_iter1.sh 
# sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_BIM_ep2_iter1.sh 
# sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_BIM_ep2_iter1.sh 
# # cifar10
# sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_BIM_ep2_iter1.sh 
# sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_BIM_ep2_iter1.sh 
# sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_BIM_ep2_iter1.sh 

######## $MODE FGSM ep4 ########
# mnist 
sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_BIM_ep4_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_BIM_ep4_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_BIM_ep4_iter1.sh 
# fashion_mnist
sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_BIM_ep4_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_BIM_ep4_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_BIM_ep4_iter1.sh 
# svhn
sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_BIM_ep4_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_BIM_ep4_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_BIM_ep4_iter1.sh 
# cifar10
sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_BIM_ep4_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_BIM_ep4_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_BIM_ep4_iter1.sh 

######## $MODE FGSM ep8 ########
# mnist 
sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_BIM_ep8_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_BIM_ep8_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_BIM_ep8_iter1.sh 
# fashion_mnist
sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_BIM_ep8_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_BIM_ep8_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_BIM_ep8_iter1.sh 
# svhn
sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_BIM_ep8_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_BIM_ep8_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_BIM_ep8_iter1.sh 
# cifar10
sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_BIM_ep8_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_BIM_ep8_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_BIM_ep8_iter1.sh 

######## $MODE FGSM ep16 ########
# mnist 
sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_BIM_ep16_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_BIM_ep16_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_BIM_ep16_iter1.sh 
# fashion_mnist
sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_BIM_ep16_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_BIM_ep16_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_BIM_ep16_iter1.sh 
# svhn
sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_BIM_ep16_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_BIM_ep16_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_BIM_ep16_iter1.sh 
# cifar10
sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_BIM_ep16_iter1.sh 
sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_BIM_ep16_iter1.sh 
sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_BIM_ep16_iter1.sh 

######## $MODE BIM ep1_iter2 ########
# # mnist 
# sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_BIM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_BIM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_BIM_ep1_iter2.sh 
# # fashion_mnist
# sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_BIM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_BIM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_BIM_ep1_iter2.sh 
# # svhn
# sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_BIM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_BIM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_BIM_ep1_iter2.sh 
# # cifar10
# sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_BIM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_BIM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_BIM_ep1_iter2.sh 


######## $MODE BIM ep1_iter4 ########
# mnist 
sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_BIM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_BIM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_BIM_ep1_iter4.sh 
# fashion_mnist
sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_BIM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_BIM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_BIM_ep1_iter4.sh 
# svhn
sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_BIM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_BIM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_BIM_ep1_iter4.sh 
# cifar10
sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_BIM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_BIM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_BIM_ep1_iter4.sh 


######## $MODE BIM ep1_iter8 ########
# mnist 
sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_BIM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_BIM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_BIM_ep1_iter8.sh 
# fashion_mnist
sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_BIM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_BIM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_BIM_ep1_iter8.sh 
# svhn
sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_BIM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_BIM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_BIM_ep1_iter8.sh 
# cifar10
sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_BIM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_BIM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_BIM_ep1_iter8.sh 


######## $MODE BIM ep1_iter16 ########
# mnist 
sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_BIM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_BIM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_BIM_ep1_iter16.sh 
# fashion_mnist
sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_BIM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_BIM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_BIM_ep1_iter16.sh 
# svhn
sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_BIM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_BIM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_BIM_ep1_iter16.sh 
# cifar10
sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_BIM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_BIM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_BIM_ep1_iter16.sh 

######## $MODE ILLCM ep1_iter2 ########
# # mnist 
# sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_ILLCM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_ILLCM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_ILLCM_ep1_iter2.sh 
# # fashion_mnist
# sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_ILLCM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_ILLCM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_ILLCM_ep1_iter2.sh 
# # svhn
# sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_ILLCM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_ILLCM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_ILLCM_ep1_iter2.sh 
# # cifar10
# sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_ILLCM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_ILLCM_ep1_iter2.sh 
# sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_ILLCM_ep1_iter2.sh 


######## $MODE ILLCM ep1_iter4 ########
# mnist 
sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_ILLCM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_ILLCM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_ILLCM_ep1_iter4.sh 
# fashion_mnist
sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_ILLCM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_ILLCM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_ILLCM_ep1_iter4.sh 
# svhn
sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_ILLCM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_ILLCM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_ILLCM_ep1_iter4.sh 
# cifar10
sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_ILLCM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_ILLCM_ep1_iter4.sh 
sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_ILLCM_ep1_iter4.sh 


######## $MODE ILLCM ep1_iter8 ########
# mnist 
sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_ILLCM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_ILLCM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_ILLCM_ep1_iter8.sh 
# fashion_mnist
sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_ILLCM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_ILLCM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_ILLCM_ep1_iter8.sh 
# svhn
sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_ILLCM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_ILLCM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_ILLCM_ep1_iter8.sh 
# cifar10
sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_ILLCM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_ILLCM_ep1_iter8.sh 
sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_ILLCM_ep1_iter8.sh 


######## $MODE ILLCM ep1_iter16 ########
# mnist 
sbatch $SCRIPTS_DIR/caps/mnist/$MODE/train_ILLCM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/caps_r/mnist/$MODE/train_ILLCM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/cnn/mnist/$MODE/train_ILLCM_ep1_iter16.sh 
# fashion_mnist
sbatch $SCRIPTS_DIR/caps/fashion_mnist/$MODE/train_ILLCM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/caps_r/fashion_mnist/$MODE/train_ILLCM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/cnn/fashion_mnist/$MODE/train_ILLCM_ep1_iter16.sh 
# svhn
sbatch $SCRIPTS_DIR/caps/svhn/$MODE/train_ILLCM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/caps_r/svhn/$MODE/train_ILLCM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/cnn/svhn/$MODE/train_ILLCM_ep1_iter16.sh 
# cifar10
sbatch $SCRIPTS_DIR/caps/cifar10/$MODE/train_ILLCM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/caps_r/cifar10/$MODE/train_ILLCM_ep1_iter16.sh 
sbatch $SCRIPTS_DIR/cnn/cifar10/$MODE/train_ILLCM_ep1_iter16.sh 
