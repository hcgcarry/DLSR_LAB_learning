import numpy as np
from PIL import Image
import glob
import random

import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
import torchvision.transforms as transforms

class custom_dataset_skewed_food(Dataset):
    '''
    skewed_training_path = "./food11re/skewed_training/"
    skewed_testing_path= "./food11re/evaluation/"
    skewed_validation_path= "./food11re/validation/"
    '''
    class_num=10
    skewed_training_path = "./food11re/food11re/training/"
    skewed_testing_path= "./food11re/food11re/evaluation/"
    skewed_validation_path= "./food11re/food11re/validation/"
    image_path_list = []
    image_label_list = []
    each_class_size = []
    img_weight=[]
    data_len =0
    img_path=""

    def __init__(self,mode,transform):
        self.getImage(mode)
        self.data_len = len(self.image_label_list)
        self.transform =transform
    def getImage(self,mode):
        if mode == "training":
            tmp_path = self.skewed_training_path
        elif mode == "testing":
            tmp_path = self.skewed_testing_path
        elif mode == "validation":
            tmp_path = self.skewed_validation_path
        self.img_path=tmp_path
        for index in range(10):
            path = tmp_path + str(index) + "/*"
            images_path = glob.glob(path)
            self.image_path_list.extend(images_path)
            self.image_label_list.extend([index]*len(images_path))
            self.each_class_size.append(len(images_path))
            '''
            self.image_path_list.append(images_path)
            self.image_label_list.append([index]*len(images_path))
            self.data_len = self.data_len + len(images_path)
            '''

    def __getitem__(self, index):
        single_image_path = self.image_path_list[index]
        '''
        print(index)
        print(self.image_label_list[index])
        print(self.image_path_list[index])
        print(self.img_weight[index])
        print("--------------")
        '''

        # Open image
        #im_as_im = Image.open(single_image_path)

        # Do some operations on image
        # Convert to numpy, dim = 28x28
        #im_as_np = np.asarray(im_as_im)/255
        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension
        #im_as_np = np.expand_dims(im_as_np, 0)
        # Some preprocessing operations on numpy array
        # ...
        # ...
        # ...

        # Transform image to tensor, change data type
        #im_as_ten = torch.from_numpy(im_as_np).float()

        # Get label(class) of the image based on the file name
        #class_indicator_location = single_image_path.rfind('_c')
        #label = int(single_image_path[class_indicator_location+2:class_indicator_location+3])

        img = Image.open(single_image_path).convert('RGB')
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
    def labelImgWeight(self):
        each_class_weight=self.getWeight()
        img_weight=[0] * self.data_len
        for index in range(self.data_len):
            img_weight[index]=each_class_weight[self.image_label_list[index]]
        self.img_weight=img_weight

        return img_weight




