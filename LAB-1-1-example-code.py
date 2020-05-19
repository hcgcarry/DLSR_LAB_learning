#!/usr/bin/env python
# coding: utf-8

# # This example show you how to train a classifier using pytorch framework:
# 
# 
# ## step
# --------------------
# 0. check your device
# 1. Load and normalizing the CIFAR10 training and test datasets using
#    ``torchvision``
# 2. Define a Convolution Neural Network
# 3. Define a loss function and optimizer
# 4. Train the network
# 5. Test the network on the test data
# --------------------
# 

# In[5]:


import torch
import torchvision
import torchvision.transforms as transforms


# ########################################################################
# ## 0. check your device
# 
# In the beginning, you have to be sure you have your gpu device available.
# 
#     [Remark] if you want to utilize GPUs for computation, you should check your system supports to CUDA.
#     (refer the following step)
# 
#     [Remark] Pytorch have two types of tensor, one is CPU tensor types, another is CUDA tensor types. 
#     GPU only can use CUDA tensor types for computation. 
# 
# -[Official document]: https://pytorch.org/docs/stable/cuda.html
# ########################################################################

# In[6]:


#To determine if your system supports CUDA
print("==> Check devices..")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device: ",device)

#Also can print your current GPU id, and the number of GPUs you can use.
print("Our selected device: ", torch.cuda.current_device())
print(torch.cuda.device_count(), " GPUs is available")


# ########################################################################
# ## 1. Load and normalizing the CIFAR10 training and test datasets using torchvision
# 
# 
# 1.1     Before building the dataset, knowing how to do data preprocessing is very important. Pytorch provides a package called "torchvision", it consists of some functions for image transformation, like: normalization, rotation, resize...etc., or if the provided functions didn't meet the needs, you can to use other libraries or tools to preprocess the data.
#    
# 
# 1.2     Pytorch provides a class called "dataset", you can create a subclass of it to format your raw data to a more suitable format for DataLoader. Fortunately, pytorch provided some popular datasets, model architectures, and functions of image transformation in "torchvision".
# 
# 
#     [Remark] If you want to build a pytorch dataset for your own data. One you can do is rewriting a new subclass of original dataset class, and anothor is using the API called "ImageFolder" to load your dataset. The return of the "ImageFolder" is also a pytorch dataset class. However, you should adjust the directory architecture to match the need of "ImageFolder". 
# 
# 
# 1.3     After defining "Dataset" class, you can start to define a "DataLoader" class. You can easily to do "Minibatch training" by DataLoader. "Minibatch training" means DataLoader will separate your dataset into several batch. And each batch consists fixed number of data depend on what batch size you set.
# 
# 
# -[How to load common dataset]: https://pytorch.org/docs/stable/torchvision/datasets.html
# 
# -[How to use ImageFolder]: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#afterword-torchvision
# 
# -[torchvision document]: https://pytorch.org/docs/stable/torchvision/index.html
# 
# -[torchvision sourcecode]: https://github.com/pytorch/vision/tree/master/torchvision
# 
# -[Dataset, DataLoader, DataLoaderIter document]: https://pytorch.org/docs/stable/data.html
# 
# -[Dataset, DataLoader, DataLoaderIter sourcecode]: https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
# 
# ########################################################################

# In[ ]:


print('==> Preparing dataset..')


# In[ ]:


"""1.1"""
# The output of torchvision datasets are PILImage images of range [0, 1]
# We transform them to Tensor type
# And normalize the data
# Be sure you do same normalization for your train and test data

#The transform function for train data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#The transform function for test data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# In[ ]:


"""1.2""" 

#Use API to load CIFAR10 train dataset 
trainset = torchvision.datasets.CIFAR10(root='/tmp/dataset-nctu', train=True, download=False, transform=transform_train)

#Use API to load CIFAR10 test dataset
testset = torchvision.datasets.CIFAR10(root='/tmp/dataset-nctu', train=False, download=False, transform=transform_test)

#Dataset definition need to know your customized transform function


# In[ ]:


"""1.3"""

#Create DataLoader to draw samples from the dataset
#In this case, we define a DataLoader to random sample our dataset. 
#For single sampling, we take one batch of data. Each batch consists 4 images
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
shuffle=False, num_workers=2)


# In[ ]:


#Because cifar10 number the data classes in range [0,10]
#However, number representation is unreadable for humans
#So, we manually set the name of each class
classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ########################################################################
# 
# # 2. Define a Convolution Neural Network
#     
# Neural networks can be constructed using the "torch.nn" package, "torch.nn" depends on "autograd" to define model. A complete model definition contains layers declaration and forwarding methods.
# 
#     
# All the model in pytorch inherit the "nn.Module" class. You can define new layer via "torch.nn" library. And, concatenate these layers into a complete model.
# 
# 
# -[How to use nn.Module] https://pytorch.org/docs/stable/nn.html#torch.nn.Module
# 
# ########################################################################

# In[ ]:


print('==> Building model..')


# In[ ]:


import torch.nn as nn


# In[ ]:


# define your own model
class Net(nn.Module):

    #define the layers
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        
    #concatenate these layers
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[ ]:


# declare a new model
tmp_net = Net()
# now, you can see current model architecture
print(tmp_net)


# In[ ]:


# or take a look at one layer of model
print(tmp_net.conv1)


# In[ ]:


# you also can change the layer of model
# but can't edit the forward method
tmp_net.fc3 = nn.Linear(15,2)
print(tmp_net)


# In[ ]:


# just edit the parameter of one layer is OK
tmp_net.fc3.out_features = 10
print(tmp_net)

# [Remark] above two method to change the layer
# architecture is important in [LAB 1-2]


# In[ ]:


#declare a new model
net = Net()
# change all model tensor into cuda type
# something like weight & bias are the tensor 
net = net.to(device) 


# ########################################################################
# 
# # 3. Define a Loss function and optimize
# 
# ########################################################################

# In[ ]:


print('==> Defining loss function and optimize..')


# In[ ]:


import torch.optim as optim

#loss function
criterion = nn.CrossEntropyLoss()
#optimization algorithm
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# ########################################################################
# 
# # 4. Train the network
# 
# Before training the model, we need to analysis the tensor variable.
# 
# 
# Each variable have many attibute, like: .grad_fn, .require_grad, .data, .grad...etc. The ".grad_fn" attribute of "torch.Tensor" is an entry point into the function that has create this "torch.Tensor" variables. Because of ".grad_fn" flag, we can easily create a computing graph in the form of DAG(directed acyclic graph).
# 
# And then, the ".require_grad" attribute allows us to determine whether the backward propagation function is going to calculate the gradient of this "torch.Tensor" variable. If one variable has a false value of require_grad, it represent that you don't want to calculate this variable's gradient, and also its gradient will not be updated.
# 
# ########################################################################

# In[ ]:


print('==> Training model..')


# In[ ]:


#Set the model in training mode
#because some function like: dropout, batchnorm...etc, will have 
#different behaviors in training/evaluation mode
#[document]: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train
net.train()

for epoch in range(75):  # loop over the dataset multiple times
    running_loss = 0.0
    correct = 0
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
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    print('%d epoch, training accuracy: %.4f' % (epoch+1, 100.*correct/len(trainset)))
print('Finished Training')


# In[ ]:


#After training , save the model first
#You can saves only the model parameters or entire model
#Some difference between the two is that entire model 
#not only include parameters but also record how each 
#layer is connected(forward method).
#[document]: https://pytorch.org/docs/master/notes/serialization.html

print('==> Saving model..')

#only save model parameters
torch.save(net.state_dict(), './checkpoint.t7')
#you also can store some log information
state = {
    'net': net.state_dict(),
    'acc': 100.*correct/len(trainset),
    'epoch': 75
}
torch.save(state, './checkpoint.t7')

#save entire model
torch.save(net, './model.pt')

print('Finished Saving')


# ########################################################################
# 
# # 5. Test the network on the test data
# 
# ########################################################################

# In[ ]:


#Before testing, we can load the saved model
#Depend on how you save your model, need 
#different way to use it

print('==> Loading model..')

#If you just save the model parameters, you
#need to redefine the model architecture, and
#load the parameters into your model
net = Net()
checkpoint = torch.load('./checkpoint.t7')
net.load_state_dict(checkpoint['net'])

#If you save the entire model
net = torch.load('./model.pt')

print('Finished Loading')


# In[ ]:


print('==> Testing model..')

#Set the model in evaluation mode
#[document]: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.eval 
net.eval()


# In[ ]:




