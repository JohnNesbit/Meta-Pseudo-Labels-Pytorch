import torch
import torch.nn as nn
import torchvision
import numpy as np
import gc
from tqdm import tqdm
import torchvision.transforms as transforms

#parameters
epochs = 40
batch_size_train = 256
batch_size_test = 256
device = "cuda"
lr = .0001

# seed
random_seed = 1
torch.manual_seed(random_seed)

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# datasets
trainset =   torchvision.datasets.CIFAR10('/files/', train=True, download=True,
                             transform=trans)

trainset_c = trainset
trainset.data = trainset.data[25000:]
trainset.targets = trainset.targets[25000:]
trainset.num_samples = 25000

data_size = trainset.num_samples

trainset_c.data = trainset_c.data[:25000]
trainset_c.targets = trainset_c.targets[:25000]
trainset_c.num_samples = 25000

train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size_train, shuffle=True)

# in a real application of this paper one would include any unsupervised data here, which highly boosts student performance
# here I am just sifoning a portion of the labeled data and stripping its labels
unsupervised_loader = torch.utils.data.DataLoader(trainset_c,batch_size=batch_size_train, shuffle=True)



test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('/files/', train=False, download=True, transform=trans), batch_size=batch_size_test, shuffle=True)

del trainset_c, trainset
gc.collect()
teacher = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=False).to(device)
teacher.classifier[1] = nn.Linear(9216,4096)
teacher.classifier[4] = nn.Linear(4096,1024)
teacher.classifier[6] = nn.Linear(1024,10)

teacher = teacher.to(device)

criterion = nn.CrossEntropyLoss()

toptimizer = torch.optim.Adam(teacher.parameters(), lr=.0001)

# use soft labels for now - uses more mem, comp exponentially with
# model size, no performace difference but hard harder to imp
loss_list = []
# teacher training
for i in range(20):
    t = tqdm(train_loader, position=0, leave=True)
    for (x, y) in t:
        pred = teacher(x.to(device))

        loss = criterion(pred, y.to(device))
        loss.backward()
        t.set_description_str(str(loss.cpu().detach().numpy()))

        toptimizer.step()
        toptimizer.zero_grad()
    acc = 0
    
    for _, (x, y) in enumerate(test_loader):
        pred = teacher(x.to(device))
        
        for l, i in enumerate(pred):
            if i.argmax(axis=0) == y[l].to(device):
                acc += 1

    print(acc/(_*batch_size_test))
    loss_list.append(acc/(_*batch_size_test))
    torch.cuda.empty_cache()
    
student = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=False).to(device)
student.classifier[1] = nn.Linear(9216,4096)
student.classifier[4] = nn.Linear(4096,1024)
student.classifier[6] = nn.Linear(1024,10)

soptimizer = torch.optim.Adam(student.parameters(), lr=.0001)
loss_list2 = []
student = student.to(device)
# student training
for i in range(epochs):
    t = tqdm(enumerate(zip(unsupervised_loader, train_loader)), position=0, leave=True)
    for c, ((x,y), (x2, y2)) in t: # loop through supervised and unsupervised data at same time
        
        # as paper proposes, alternate between unsupervised method and supervised freqently, we do this every cycle
        
        # unsupervised, teacher teaches student
        x = x.to(device)
        y_hat = teacher(x)
        pred = student(x)

        loss = criterion(pred, y_hat.detach().argmax(axis=1))
        loss.backward()
        
        # compute h - the teacher's feedback coefficient
        # reconstruct each theta's update from how it updates after calling a soptim.step()
        w_list = [] # theta timstep=t gradients on yhat preds
        for param in student.modules():
            #print(param)
            try:
                w_list.append(param.weight.grad.flatten())
            except:
                continue
        
        # update student based on teacher preds
        soptimizer.step()
        soptimizer.zero_grad()
        
        # predict on real data
        pred = student(x2.to(device))

        loss = criterion(pred, y2.to(device))
        loss.backward()
        t.set_description_str(str(loss.cpu().detach().numpy()))
        h = 0
        
        # update student weights on reals
        soptimizer.step()
        soptimizer.zero_grad()
        
        if c == 0:
            continue
        
        # sum up h, which is the product of the gradients, with the gradient of the real y being first adn transformed
        r_list = [] # theta timstep=t gradients on yhat preds
        for _, param in enumerate(student.modules()):
            #print(param)
            try:
                r_list.append(param.weight.grad.flatten())
            except:
                continue
        
                
        #print(len(r_list))
        for _ in range(len(r_list)):
            #print(r_list[_].shape, w_list[_].shape)
            h += lr*torch.matmul((r_list[_].reshape([r_list[_].size()[0], 1]).T),(w_list[_].reshape([w_list[_].size()[0], 1])))
            
        
        torch.cuda.empty_cache()
        gc.collect()
        
        
        # supervised, approzimate gradient updates for teacher
        Tloss = criterion(y_hat, y_hat.detach().argmax(axis=1))
        Tloss.backward()
        
        # calculate teacher gradients on unsupervised material with h approximation to backtrace student stack
        ugrad = []
        for param in teacher.modules():
            try:
                ugrad.append(h*param.weight.grad)
            except:
                continue
        
        toptimizer.zero_grad()
        
        # update teacher on real data
        ypred = teacher(x2.to(device))
        Rloss = criterion(ypred, y2.to(device))
        Rloss.backward()
        
        rgrad = []
        for _, param in enumerate(teacher.modules()):
            try:
                param.weight.grad = param.weight.grad + ugrad[_]
            except:
                continue
            
        # the paper calls for an optional UDA update but that is in an appendix, and it is not at the root of the paper so I did not impliment it here
        
        toptimizer.step()
        toptimizer.zero_grad()
        
        torch.cuda.empty_cache()
        gc.collect()

    # find accuracy of models and save it
    loss = 0
    acc = 0
    for _, (x, y) in enumerate(test_loader):
        pred = teacher(x.to(device))
        for l, i in enumerate(pred):
            if i.argmax(axis=0) == y[l].to(device):
                acc += 1

    print(acc/(_*batch_size_test))
    loss_list.append(acc/(_*batch_size_test))
    torch.cuda.empty_cache()
    acc = 0
    for _, (x, y) in enumerate(test_loader):
        pred = student(x.to(device))
        for l, i in enumerate(pred):
            if i.argmax(axis=0) == y[l].to(device):
                acc += 1

    print(acc/(_*batch_size_test))
    loss_list2.append(acc/(_*batch_size_test))
    torch.cuda.empty_cache()
    
# print lists, plot data
print("teacher: ", loss_list)
print("student: ", loss_list2)

import matplotlib.pyplot as plt

plt.plot(range(len(loss_list)), loss_list)
plt.plot(range(len(loss_list2)), loss_list2)
plt.show()
