#!/usr/bin/env python
# coding: utf-8

from imgaug import augmenters as iaa
from torch.utils.data.sampler import  WeightedRandomSampler
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
from model import Net
import torch.nn as nn
import torchvision.models as models

from custom_dataset import custom_dataset_skewed_food

###################variable
epoch_count=15
weight_decay=0.0
dropout=0
batch_size = 32

########################

#To determine if your system supports CUDA
print("==> Check devices..")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device: ",device)

print('==> Preparing dataset..')


#The transform function for train data
transform_train = transforms.Compose([
    transforms.RandomCrop(256, padding=4),
    transforms.Resize(224),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



'''
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
transform_train= transforms.Compose([
    iaa.Sequential([
        iaa.flip.Fliplr(p=0.5),
        iaa.flip.Flipud(p=0.5),
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.MultiplyBrightness(mul=(0.65, 1.35)),
    ]).augment_image,
    transforms.ToTensor()
])

'''
seq = iaa.Sequential([
    iaa.flip.Flipud(p=0.5),
    iaa.MultiplyBrightness(mul=(0.65, 1.35)),
    #iaa.GammaContrast(1.5),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Fliplr(p=0.5), # 水平翻轉影象
    iaa.GaussianBlur(sigma=(0, 3.0)), # 使用0到3.0的sigma模糊影象
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
                  iaa.GaussianBlur(sigma=(0, 0.5))
                  ),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.8)
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
])

'''
iaa.Affine(
    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    rotate=(-25, 25),
    shear=(-8, 8)
)
'''
#The transform function for test data



trainset  = custom_dataset_skewed_food("training")
#trainset.setImgaug(seq)
trainset.setTransform(transform_train)


randomSampler = torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=9000)

weightSampler = WeightedRandomSampler(trainset.labelImgWeight(),\
                                num_samples=trainset.data_len,\
                                replacement=True)


#trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=0,sampler=weightSampler)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)



print('==> Building model..')




#net = Net()
net = models.resnet18(pretrained=True);
# replace the last layer
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 11)

for idx, (name, param) in enumerate(net.named_parameters()):
    if idx < 60:  # count of layers is 62
        param.requires_grad = False

    if param.requires_grad == True:
        print("\t", idx, name)

#net.defineDropout(dropout)

net = net.to(device)

print('==> Defining loss function and optimize..')
import torch.optim as optim

#loss function
weights=trainset.getClassWeight()
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
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
    class_count = [0] * 11
    class_correct_count = [0] * 11
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
    for i in range(11):
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





