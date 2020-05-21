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
    skewed_training_path = "./food11re/food11re/training/"
    skewed_testing_path= "./food11re/food11re/evaluation/"
    skewed_validation_path= "./food11re/food11re/validation/"
    image_path_list = []
    image_label_list = []
    each_class_size = []
    data_len =0

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
        for index in range(10):
            path = tmp_path + str(index) + "/*"
            images_path = glob.glob(path)
            self.image_path_list.extend(images_path)
            self.image_label_list.extend([index]*len(images_path))
            '''
            self.image_path_list.append(images_path)
            self.image_label_list.append([index]*len(images_path))
            self.each_class_size.append(len(images_path))
            self.data_len = self.data_len + len(images_path)
            '''


    def __getitem__(self, index):
        single_image_path = self.image_path_list[index]

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


class custom_dataset_skewed_food_randomGetItem(Dataset):
    '''
    skewed_training_path = "./food11re/skewed_training/"
    skewed_testing_path= "./food11re/evaluation/"
    skewed_validation_path= "./food11re/validation/"
    '''
    skewed_training_path = "./food11re/food11re/training/"
    skewed_testing_path= "./food11re/food11re/evaluation/"
    skewed_validation_path= "./food11re/food11re/validation/"
    image_path_list = []
    image_label_list = []
    each_class_size = []
    data_len =0

    def __init__(self,mode,transform):
        self.getImage(mode)
        #self.data_len = len(self.image_label_list)
        self.transform =transform
    def getImage(self,mode):
        if mode == "training":
            tmp_path = self.skewed_training_path
        elif mode == "testing":
            tmp_path = self.skewed_testing_path
        elif mode == "validation":
            tmp_path = self.skewed_validation_path
        for index in range(10):
            path = tmp_path + str(index) + "/*"
            images_path = glob.glob(path)
            '''
            self.image_path_list.extend(images_path)
            self.image_label_list.extend([index]*len(images_path))
            '''
            self.image_path_list.append(images_path)
            self.image_label_list.append([index]*len(images_path))
            self.each_class_size.append(len(images_path))
            self.data_len = self.data_len + len(images_path)


    def __getitem__(self, index):
        # Get image name from the pandas df
        class_index = random.randint(0,9)
        image_index = random.randint(1,self.each_class_size[class_index]-1)
        single_image_path = self.image_path_list[class_index][image_index]
        #single_image_path = self.image_path_list[index]
        # Open image
        #im_as_im = Image.open(single_image_path)

        # Transform image to tensor, change data type
        #im_as_ten = torch.from_numpy(im_as_np).float()

        # Get label(class) of the image based on the file name
        #class_indicator_location = single_image_path.rfind('_c')
        #label = int(single_image_path[class_indicator_location+2:class_indicator_location+3])

        img = Image.open(single_image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.image_label_list[class_index][image_index]
        #label = self.image_label_list[index]
        return (img, label)

    def __len__(self):
        return self.data_len
