#!/usr/bin/env python
# coding: utf-8

from imgaug import augmenters as iaa
from torch.utils.data.sampler import  WeightedRandomSampler
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
import torch.nn as nn
from custom_dataset import brid
from model import my_model

from augmentation import training_augmentation,testing_augmentation
    
###################variable
epoch_count=100
weight_decay=1e-2
weight_decay=0
dropout=0
init_lr=0.001
batch_size = 60
num_of_class=200

########################

def validationDataset_init():
    
    # img_final_height = int(375*0.7)
    # img_final_width= int(500*0.7)
    # transform_test = transforms.Compose([
    # transforms.Resize((img_final_height,img_final_width)),
    # transforms.ToTensor(),
    # #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225])
    # ])

    validation_set = brid("validation")
    validation_set = testing_augmentation(validation_set)
    #trainset= custom_dataset_skewed_food("training",transform_test)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32,shuffle=False, num_workers=0)
    #net = torch.load(modelPath,map_location=torch.device('cpu'))
    ######################################training statics
    return validation_loader,validation_set
    
def trainDataset_init():
    trainset  = brid("training")
    trainset = training_augmentation(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    return trainloader,trainset
    

trainloader ,trainset = trainDataset_init()
validation_loader,validation_set = validationDataset_init()


model = my_model(num_of_class,init_lr,epoch_count)
print("model type",type(model))
print("model type",type(model.training))
model.train(trainloader,trainset,validation_loader,validation_set)
model.saveModel()