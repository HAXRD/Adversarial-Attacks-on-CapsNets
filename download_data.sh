#!/bin/bash

find . \( -name __pycache__ -o -name "*.pyc" \) -delete

rm -rf data 
rm -rf summary

rm -rf debug

rm -rf dataset/mnist/*.npz

rm -rf dataset/fashion_mnist/*.npz
rm -rf dataset/fashion_mnist/*.gz 

rm -rf dataset/svhn/*.mat
rm -rf dataset/svhn/*.npz 

rm -rf dataset/cifar10/*.gz 
rm -rf dataset/cifar10/*.mat 
rm -rf dataset/cifar10/*.html

wget -P dataset/mnist https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

wget -P dataset/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget -P dataset/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget -P dataset/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget -P dataset/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

wget -P dataset/svhn http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -P dataset/svhn http://ufldl.stanford.edu/housenumbers/test_32x32.mat

wget -P dataset/ https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
tar -C dataset/ -xvzf dataset/cifar-10-matlab.tar.gz 
cp -r dataset/cifar-10-batches-mat/* dataset/cifar10/
rm -rf dataset/cifar-10-batches-mat
rm -rf dataset/cifar-10-matlab*