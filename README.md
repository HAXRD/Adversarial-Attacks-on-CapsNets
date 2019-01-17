
# How to run this code


## Download datasets
```
./download_data.sh
```
**Some datasets may not be accessable in countries with limited internet access.**

## Run the unit test code
```
python unitT_prepare_dataset.py
python unitT_mnist_input.py
python unitT_fashion_mnist_input.py
python unitT_svhn_input.py
python unitT_cifar10_input.py
python unitT_exp_train.py
python unitT_exp_gen_adv.py
python unitT_exp_test.py
```


## Prepare datasets
```
python prepare_dataset.py
```

## Train
```
python experiment.py --data_dir=data/caps/mnist/ --dataset=mnist --summary_dir=summary/ --save_epochs=1 --max_epochs=3 --model=caps
```

## Generate Adversarial Examples for Testing
```
python experiment.py --mode=gen_adv --num_gpus=2 --dataset=mnist --adversarial_method=FGSM --total_batch_size=2 --summary_dir=summary/caps/mnist/Default/ --data_dir=data/caps/mnist/
```

## Test
```
python experiment.py --summary_dir=summary/caps/mnist/Default/ --load_test_path=data/caps/mnist/test.npz --mode=test
```

## For debug
```
salloc --time=3:0:0 --mem=30000M --cpus-per-task=6 --gres=gpu:2
```
**train**
```
python experiment.py --data_dir=data/caps/cifar10/ --dataset=cifar10 --adversarial_method=Default --model=caps --save_epochs=1 --max_epochs=2 --total_batch_size=50 --summary_dir=summary/b50
```
**gen_adv**
```
python experiment.py --mode=gen_adv --data_dir=data/caps/cifar10/ --dataset=cifar10 --total_batch_size=50 --adversarial_method=BIM --epsilon=2 --iteration_n=1 --summary_dir=summary/b50/caps/cifar10/Default/
```

**test**
```
python experiment.py --mode=test --adversarial_method=Default --total_batch_size=50 --summary_dir=summary/b50/caps/cifar10/Default/ --load_test_path=data/caps/cifar10/test.npz
```