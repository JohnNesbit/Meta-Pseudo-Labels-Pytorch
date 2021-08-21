# Meta-Pseudo-Labels-Pytorch
This repo is a demonstration/implementation of Meta Pseudo Labels in pytorch using CIFAR 10
The Meta Pseudo Labels paper(https://arxiv.org/pdf/2003.10580.pdf) which is the basis for this repository is currently SOTA for its proccessing requirement on Imagenet. 
This repository aims to make it easy for anyone to write a script in pytorch utilizing this technique. The soft Pseudo labels script works well for smaller models but becomes 
inefficent as the model size grows. The example model used within the Soft Pseudo labels script is of a comprable size to AlexNet. The Pytorch backend handles the gradient 
backprop through the student model for the soft labels script, but a novel method of backprob is needed for the hard labels script(which is still in the works of now).
