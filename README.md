# Meta-Pseudo-Labels-Pytorch
## Overview

This repo is a demonstration/implementation of Meta Pseudo Labels in pytorch using CIFAR 10
The Meta Pseudo Labels paper(https://arxiv.org/pdf/2003.10580.pdf) which is the basis for this repository is currently SOTA for its proccessing requirement on Imagenet. 
This repository aims to make it easy for anyone to write a script in pytorch utilizing this technique. The soft Pseudo labels script works well for smaller models but becomes 
inefficent as the model size grows. The example model used within the Soft Pseudo labels script is of a comprable size to AlexNet. The Pytorch backend handles the gradient 
backprop through the student model for the soft labels script, but a novel method of backprob is needed for the hard labels script.

## Hard Label Script Specifics

The Hard label script uses two main equations to backtrace student loss to the teacher. The backtrace equations are simply more compuationally efficient derrivations of this basic
equation which breaks down the problem into a simple chain rule equation where the gradient of the teacher in relation to the student is simply the student's gradient times the
teacher's relation to the student's weights:

![image](https://user-images.githubusercontent.com/49009243/130339482-322280d5-8f42-4a29-ba45-c87f5d711469.png)

The teacher's relation to the student's weights is broken down using REINFORCE into two dependencies of the student and one dependency of the teacher, from this state it can be
encoded into python.


![image](https://user-images.githubusercontent.com/49009243/130339521-96dddb59-d7a6-4e92-891a-4ceacabd0ed5.png)


The gradient of the teacher weights is implied to be the left side of the equation. The paper includes an equation to integrate UDA loss into this training regime, but that 
equation is ignored in this repo as that is not at the core of the paper or the methodology.
