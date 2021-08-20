import torch
import torch.nn as nn
import torchvision
import numpy as np
import gc
from tqdm import tqdm

#parameters
epochs = 20
batch_size_train = 256
batch_size_test = 512
device = "cuda"

# seed
random_seed = 1
torch.manual_seed(random_seed)

# datasets
trainset = torchvision.datasets.CIFAR10('/files/', train=True, download=True,
										transform=torchvision.transforms.Compose([
											torchvision.transforms.ToTensor(),
											torchvision.transforms.Normalize(
												(0.1307,), (0.3081,))
										]))

trainset_c = trainset
trainset.data = trainset.data[25000:]
trainset.targets = trainset.targets[25000:]
trainset.num_samples = 25000

data_size = trainset.num_samples

trainset_c.data = trainset_c.data[:25000]
trainset_c.targets = trainset_c.targets[:25000]
trainset_c.num_samples = 25000

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)

# in a real application of this paper one would include any unsupervised data here, which highly boosts student performance
# here I am just sifoning a portion of the labeled data and stripping its labels
unsupervised_loader = torch.utils.data.DataLoader(trainset_c, batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
	torchvision.datasets.CIFAR10('/files/', train=False, download=True,
								 transform=torchvision.transforms.Compose([
									 torchvision.transforms.ToTensor(),
									 torchvision.transforms.Normalize(
										 (0.1307,), (0.3081,))
								 ])),
	batch_size=batch_size_test, shuffle=True)


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.list = nn.ModuleList([])

		# slightly more lightweight version of AlexNet - 2 less layers, less depth in FCs

		self.list.append(nn.Conv2d(3, 64, (5, 5), stride=(2, 2)))
		self.list.append(nn.ReLU())
		self.list.append(nn.BatchNorm2d(64))

		self.list.append(nn.Conv2d(64, 256, (5, 5), stride=2))
		self.list.append(nn.ReLU())
		self.list.append(nn.BatchNorm2d(256))

		self.list.append(nn.Conv2d(256, 384, (3, 3), stride=1))
		self.list.append(nn.ReLU())
		self.list.append(nn.BatchNorm2d(384))

		self.list.append(nn.Conv2d(384, 256, (3, 3), stride=1))
		self.list.append(nn.ReLU())
		self.list.append(nn.BatchNorm2d(256))

		self.list.append(nn.Flatten())
		self.list.append(nn.Linear(256, 1028))
		self.list.append(nn.ReLU())
		self.list.append(nn.BatchNorm1d(1028))

		self.list.append(nn.Linear(1028, 512))
		self.list.append(nn.ReLU())
		self.list.append(nn.BatchNorm1d(512))

		self.list.append(nn.Linear(512, 10))
		self.list.append(nn.Softmax())

	def forward(self, x):
		for l in self.list:
			x = l(x)
		return x


del trainset_c, trainset
gc.collect()


# train teacher
teacher = Model().to(device)
criterion = nn.CrossEntropyLoss()

toptimizer = torch.optim.Adam(teacher.parameters(), lr=.0001)

# use soft labels for now - uses more mem, comp exponentially with
# model size, no performace difference but hard harder to imp
loss_list = []
for i in range(epochs):
    for (x, y) in train_loader:
        pred = teacher(x.to(device))

        loss = criterion(pred, y.to(device))
        loss.backward()

        toptimizer.step()
        toptimizer.zero_grad()
    acc = 0
    for _, (x, y) in enumerate(test_loader):
        pred = teacher(x.to(device))

        for l, i in enumerate(pred):
            if i.argmax(axis=0) == y[l].to(device):
                acc += 1

    print(acc / (_ * batch_size_test))
    loss_list.append(acc / (_ * batch_size_test))
    torch.cuda.empty_cache()


student = Model().to(device)
soptimizer = torch.optim.Adam(student.parameters(), lr=.0001)
loss_list2 = []

# student training
for i in range(epochs):
    t = tqdm(enumerate(zip(unsupervised_loader, train_loader)), position=0, leave=True)
    for c, ((x, y), (x2, y2)) in t:  # loop through supervised and unsupervised data at same time

        # as paper proposes, alternate between unsupervised method and supervised freqently, we do this every cycle

        # unsupervised, teacher teaches student
        x = x.to(device)
        y_hat = teacher(x)
        pred = student(x)

        loss = criterion(pred, y_hat.argmax(axis=1))
        loss.backward()

        soptimizer.step()
        soptimizer.zero_grad()

        torch.cuda.empty_cache()
        gc.collect()

        # supervised, trickle through whole graph
        pred = student(x2.to(device))

        loss = criterion(pred, y2.to(device))
        loss.backward()

        t.set_description_str(str(loss.cpu().detach().numpy()))

        soptimizer.step()
        toptimizer.step()
        toptimizer.zero_grad()
        soptimizer.zero_grad()

        torch.cuda.empty_cache()
        gc.collect()

    loss = 0
    acc = 0
    for _, (x, y) in enumerate(test_loader):
        pred = teacher(x.to(device))
        for l, i in enumerate(pred):
            if i.argmax(axis=0) == y[l].to(device):
                acc += 1

    print(acc / (_ * batch_size_test))
    loss_list.append(acc / (_ * batch_size_test))
    torch.cuda.empty_cache()
    acc = 0
    for _, (x, y) in enumerate(test_loader):
        pred = student(x.to(device))
        for l, i in enumerate(pred):
            if i.argmax(axis=0) == y[l].to(device):
                acc += 1

    print(acc / (_ * batch_size_test))
    loss_list2.append(acc / (_ * batch_size_test))
    torch.cuda.empty_cache()



print("teacher: ", loss_list)
print("student: ", loss_list2)

import matplotlib.pyplot as plt

plt.plot(range(len(loss_list)), loss_list)
plt.plot(range(len(loss_list2)), loss_list2)
plt.show()