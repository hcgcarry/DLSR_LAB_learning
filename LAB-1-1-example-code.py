#!/usr/bin/env python
# coding: utf-8

from torch.utils.data.sampler import  WeightedRandomSampler
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from model import Net
import torch.nn as nn

from custom_dataset import custom_dataset_skewed_food

###################variable
epoch_count=55
weight_decay=0.0005
dropout=0


########################

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



sampler = WeightedRandomSampler(trainset.labelImgWeight(),\
                                num_samples=trainset.data_len,\
                                replacement=True)


'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
shuffle=False, num_workers=0,sampler=sampler)
'''

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)



print('==> Building model..')




net = Net()
net.defineDropout(dropout)

net = net.to(device)

print('==> Defining loss function and optimize..')
import torch.optim as optim

#loss function
criterion = nn.CrossEntropyLoss()
#optimization algorithm
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#weight_decay 是l2 regularization
optimizer= optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)



print('==> Training model..')


net.train()

#每一個epoch都會做完整個dataset
for epoch in range(epoch_count):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
    class_count = [0] * 10
    class_correct_count = [0] * 10
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
        #each class
        ans_list = pred.eq(labels)
        for index in range(labels.shape[0]):
            class_count[labels[index]] += 1
            if ans_list[index]:
                class_correct_count[labels[index]] += 1

        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    print('%d epoch, training accuracy: %.4f' % (epoch+1, correct/len(trainset)))
    for i in range(10):
        print('Class %d : %.2f %d/%d' % \
              (i,class_correct_count[i]/class_count[i],class_correct_count[i],class_count[i]))


print('Finished Training')

print('==> Saving model..')


state = {
    'net': net.state_dict(),
    'acc': correct/len(trainset),
    'class_correct_count': class_correct_count,
    'class_count': class_count,
    'actual_class_count': trainset.each_class_size,
    'image_path': custom_dataset_skewed_food.img_path,
    'parameters':{
        'epoch': epoch_count,
        'dropout': dropout,
        'optimizer':optimizer.__repr__
    }
}

torch.save(state, './trainedModel/skew_checkpoint.t7')

#save entire model
torch.save(net, './trainedModel/skew_model.pt')

print('Finished Saving')





