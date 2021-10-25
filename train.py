#!/usr/bin/env python
# coding: utf-8

from imgaug import augmenters as iaa
import torch.optim as optim
from torch.utils.data.sampler import  WeightedRandomSampler
from TripletLoss import TripletLoss
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
from model import Net
import torch.nn as nn
import torchvision.models as models

from augumentation import augumentation
from custom_dataset import brid

###################variable
epoch_count=100
weight_decay=1e-2
weight_decay=0
dropout=0
init_lr=0.001
batch_size = 60
num_of_class=200

########################
def validation_init():
    
    img_final_height = int(375*0.7)
    img_final_width= int(500*0.7)
    transform_test = transforms.Compose([
    transforms.Resize((img_final_height,img_final_width)),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


    validation_set = brid("validation")
    validation_set.setTransform(transform_test)
    #trainset= custom_dataset_skewed_food("training",transform_test)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32,
                                            shuffle=False, num_workers=0)



    #net = torch.load(modelPath,map_location=torch.device('cpu'))

    ######################################training statics
    return validation_loader,validation_set

def validation_run(validation_loader,validation_set):
    correct = 0 
    for i, (inputs, labels) in enumerate(validation_loader , 0):
        print("batch:",i)
        # change the type into cuda tensor
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        outputs = net(inputs)
        # select the class with highest probability
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()

    # writeResult2Answer(testset,resultDict)
    print('validation accuracy: %.4f' % (correct/len(validation_set)))
    

#To determine if your system supports CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device: ",device)


trainset  = brid("training")
trainset = augumentation(trainset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
validation_loader,validation_set = validation_init()



#net = Net()
net = models.resnet50(pretrained=True);
# replace the last layer
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, num_of_class)
# net.fc.requires_grad_

# for idx, (name, param) in enumerate(net.named_parameters()):
#     param.requires_grad_ = False

# net.fc.requires_grad_ = True

#net.defineDropout(dropout)
net = net.to(device)


#loss function
# weights=trainset.getClassWeight()
# class_weights = torch.FloatTensor(weights).to(device)
#optimization algorithm
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#weight_decay 是l2 regularization

optimizer= optim.Adam(net.parameters(), lr=init_lr, amsgrad=False)
criterion = nn.CrossEntropyLoss()
# criterion = TripletLoss()

net.train()


for epoch in range(epoch_count):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0

    for i,(inputs, labels) in enumerate(trainloader,0):
        # print("inputs.shape",inputs.shape)
        # print("labels",labels)

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

        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if count % 20 == 19:    # print every 200 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i+ 1, running_loss / 20))
        running_loss = 0.0
    print('%d epoch, training accuracy: %.4f' % (epoch+1, correct/len(trainset)))
    validation_run(validation_loader,validation_set)


print('Finished Training')

print('==> Saving model..')



state = {
    'net': net.state_dict(),
    'acc': correct/len(trainset),
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





