import numpy as np
from PIL import Image
import glob
import random
from showsubplot import showsubplot
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
import torchvision.transforms as transforms

class custom_dataset_skewed_food(Dataset):
    skewed_training_path = "./food11re/skewed_training/"
    skewed_testing_path= "./food11re/evaluation/"
    skewed_validation_path= "./food11re/validation/"
    class_num=11
    '''
    skewed_training_path = "./food11re/food11re/training/"
    skewed_testing_path= "./food11re/food11re/evaluation/"
    skewed_validation_path= "./food11re/food11re/validation/"
    '''
    image_path_list = []
    image_label_list = []
    each_class_size = []
    img_weight=[]
    data_len =0
    img_path=""
    transform=None
    Imgaug = None
    
    def __init__(self,mode):
        self.getImage(mode)
        self.data_len = len(self.image_label_list)

    def getImage(self,mode):
        if mode == "training":
            tmp_path = self.skewed_training_path
        elif mode == "testing":
            tmp_path = self.skewed_testing_path
        elif mode == "validation":
            tmp_path = self.skewed_validation_path
        self.img_path=tmp_path
        for index in range(11):
            path = tmp_path + str(index) + "/*"
            images_path = glob.glob(path)
            self.image_path_list.extend(images_path)
            self.image_label_list.extend([index]*len(images_path))
            self.each_class_size.append(len(images_path))

    def setImgaug(self,imgaug):
        self.Imgaug=imgaug
    def setTransform(self,transform):
        self.transform =transform
        
    def __getitem__(self, index):
        single_image_path = self.image_path_list[index]
        '''
        print(index)
        print(self.image_label_list[index])
        print(self.image_path_list[index])
        print(self.img_weight[index])
        print("--------------")
        '''

        img = Image.open(single_image_path).convert('RGB')
        ## 使用imgaug 做augumetation
        # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
        # or a list of 3D numpy arrays, each having shape (height, width, channels).
        # Grayscale images must have shape (height, width, 1) each.
        # All images must have numpy's dtype uint8. Values are expected to be in
        # range 0-255.
        if self.Imgaug is not None:
            im_as_np = np.asarray(img)
            im_as_np = self.Imgaug.augment_image(im_as_np)
            #img = torch.from_numpy(im_as_np)
            img = Image.fromarray(im_as_np, mode='RGB')



        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension
        #im_as_np = np.expand_dims(im_as_np, 0)



        if self.transform is not None:
            img = self.transform(img)
        label = self.image_label_list[index]
        #print("index",index,"img path",single_image_path,"label",label)
        return (img, label)

    def __len__(self):
        return self.data_len

    def getWeight(self):
        each_class_propotional=[self.each_class_size[i]/self.data_len for i in range(self.class_num)]
        each_class_weight=[1/each_class_propotional[i] for i in range(self.class_num)]
        return each_class_weight
    def getClassWeight(self):
        #Max(Number of occurrences in most common class ) / (Number of occurrences in rare classes)
        return [1/self.each_class_size[index] for index in range(self.class_num)]
    def labelImgWeight(self):
        each_class_weight=self.getWeight()
        img_weight=[0] * self.data_len
        for index in range(self.data_len):
            img_weight[index]=each_class_weight[self.image_label_list[index]]
        self.img_weight=img_weight

        return img_weight
    def showImage(self,num=5):

        '''
        img = Image.open(single_image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.image_label_list[index]
        #print("index",index,"img path",single_image_path,"label",label)
        return (img, label)
        plt.imshow(inp)

        if title is not None:
            plt.title(title)
        plt.pause(1000)  # pause a bit so that plots are updated
        '''

    def showImagesNumpy(self):
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224, padding=4),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        self.transform=transform_train;
        inputs = [];
        for index in range(5):
            image, label = self.__getitem__(index);
            inputs.append(np.array(image));
        showsubplot(inputs)
    def showImagesTensor(self,imgaugTransform = None,transform = None):
        def imshow(inp, title=None):
            """Imshow for Tensor."""
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            if title is not None:
                plt.title(title)
            plt.pause(1000)  # pause a bit so that plots are updated


        # The transform function for train data
        self.setTransform(transform)
        self.setImgaug(imgaugTransform)

        trainloader = torch.utils.data.DataLoader(self, batch_size=8,
                                                  shuffle=False, num_workers=0)
        inputs, classes = next(iter(trainloader));
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs);
        imshow(out,title="test");









class brid(Dataset):
    skewed_training_path = "/workspace/CV/hw1/2021VRDL_HW1_datasets/training_dataset"
    skewed_testing_path = "/workspace/CV/hw1/2021VRDL_HW1_datasets/testing_dataset"
    skewed_validation_path = "/workspace/CV/hw1/2021VRDL_HW1_datasets/validation_dataset"

    class_num=200

    transform=None
    Imgaug = None
    
    def __init__(self,mode):
        self.mode = mode
        self.getImage(mode)
        self.getLabel()
        self.data_len = len(self.image_list)
        # print("self.datalen",self.data_len)
        # print("image_name_list",self.image_name_list)
        # print("image_list",self.image_list)
        # print("label",self.labelDict)

    def getImage(self,mode):
        if mode == "training":
            dir_path= self.skewed_training_path
        elif mode == "testing":
            dir_path= self.skewed_testing_path
        elif mode == "validation":
            dir_path= self.skewed_validation_path

        path = dir_path +  "/*"
        images_path = glob.glob(path)
        self.image_list=  []
        self.image_name_list = []
        for image_path in images_path:
            img = Image.open(image_path).convert('RGB')
            self.image_list.append(img)
            self.image_name_list.append(image_path.split('/')[-1])



    def getLabelByIndex(self,index):
        return self.labelDict[self.image_name_list[index]]
        

    def getLabel(self):
        label_path = "/workspace/CV/hw1/2021VRDL_HW1_datasets/training_labels.txt"
        self.labelDict = {}
        self.labelValue2labelName = {}
        
        with open(label_path,'r') as f:
            for line in f.readlines():
                labelValue = int(line.split()[1].split('.')[0])-1
                self.labelDict[line.split()[0]] = labelValue
                label = line.split()[1]
                self.labelValue2labelName[labelValue] = label

        # print("labelDict",self.labelDict)
        # print("labelvalue2labelNmae",self.labelValue2labelName)
    def labelValue2Label(self,labelValue):
        return self.labelValue2labelName[labelValue]


    def setTransform(self,transform):
        self.transform =transform
    def setImgaug(self,imgaug):
        self.Imgaug=imgaug
        
    def __getitem__(self, index):

        ## 使用imgaug 做augumetation
        # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
        # or a list of 3D numpy arrays, each having shape (height, width, channels).
        # Grayscale images must have shape (height, width, 1) each.
        # All images must have numpy's dtype uint8. Values are expected to be in
        # range 0-255.
        # print("getItem",index)
        img = self.image_list[index]

        if self.Imgaug is not None:
            im_as_np = np.asarray(img)
            im_as_np = self.Imgaug.augment_image(im_as_np)
            #img = torch.from_numpy(im_as_np)
            img = Image.fromarray(im_as_np, mode='RGB')



        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension
        #im_as_np = np.expand_dims(im_as_np, 0)



        if self.transform is not None:
            img = self.transform(img)
        if self.mode == "testing":
            label = self.image_name_list[index]
        else:
            label = self.getLabelByIndex(index)
        #print("index",index,"img path",single_image_path,"label",label)
        # label = torch.randn(1)
        return (img, label)

    def __len__(self):
        return self.data_len
