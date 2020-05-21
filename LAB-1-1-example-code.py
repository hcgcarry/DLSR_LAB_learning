#!/usr/bin/env python
# coding: utf-8



import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from model import Net
import torch.nn as nn

from custom_dataset import custom_dataset_skewed_food


#To determine if your system supports CUDA
print("==> Check devices..")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device: ",device)

print('==> Preparing dataset..')


#The transform function for train data
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224, padding=4),
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#The transform function for test data



#Use API to load CIFAR10 train dataset 
#trainset = torchvision.datasets.CIFAR10(root='D:/homework/研究所/碩0/讀書會一/DLSR_lab01/food11re', train=True, download=False, transform=transform_train)
#trainset = torchvision.datasets.CIFAR10(root='/tmp/dataset-nctu', train=True, download=False, transform=transform_train)
trainset  = custom_dataset_skewed_food("training",transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
shuffle=True, num_workers=0)




print('==> Building model..')




net = Net()
net = net.to(device)

print('==> Defining loss function and optimize..')
import torch.optim as optim

#loss function
criterion = nn.CrossEntropyLoss()
#optimization algorithm
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


print('==> Training model..')


net.train()

#每一個epoch都會做完整個dataset
for epoch in range(75):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
    #inputs 是一個batch 的image labels是一個batch的label
    for i, (inputs, labels) in enumerate(trainloader, 0):

        #change the type into cuda tensor
        inputs = inputs.to(device) 
        labels = labels.to(device) 

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        outputs = net(inputs)
        # select the class with highest probability
        _, pred = outputs.max(1)
        # if the model predicts the same results as the true
        # label, then the correct counter will plus 1
        correct += pred.eq(labels).sum().item()
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    print('%d epoch, training accuracy: %.4f' % (epoch+1, 100.*correct/len(trainset)))
print('Finished Training')

print('==> Saving model..')

#only save model parameters
torch.save(net.state_dict(), './checkpoint.t7')
#you also can store some log information
state = {
    'net': net.state_dict(),
    'acc': 100.*correct/len(trainset),
    'epoch': 75
}
#torch.save(state, './checkpoint.t7')

#save entire model
torch.save(net, './model.pt')

print('Finished Saving')

print('==> Loading model..')

#If you just save the model parameters, you
#need to redefine the model architecture, and
#load the parameters into your model
'''
net = Net()
checkpoint = torch.load('./checkpoint.t7')
net.load_state_dict(checkpoint['net'])
'''




