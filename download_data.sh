#!/bin/bash

find . \( -name __pycache__ -o -name "*.pyc" \) -delete

rm -rf ../scratch/data 
rm -rf summary

rm -rf debug

rm -rf dataset/mnist/*.npz

rm -rf dataset/cifar10/*.gz 
rm -rf dataset/cifar10/*.mat 
rm -rf dataset/cifar10/*.html

wget -P dataset/mnist https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

wget -P dataset/ https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
tar -C dataset/ -xvzf dataset/cifar-10-matlab.tar.gz 
cp -r dataset/cifar-10-batches-mat/* dataset/cifar10/
rm -rf dataset/cifar-10-batches-mat
rm -rf dataset/cifar-10-matlab*