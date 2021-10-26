
from REA import RandomErasing
from imgaug import augmenters as iaa
import torchvision.transforms as transforms
# def training_augumentation(train_set):
    
#     img_final_height = int(375*0.7)
#     img_final_width= int(500*0.7)

#     #The transform function for train data
#     # transform_train = transforms.Compose([
#     #     transforms.Resize((int(img_final_height*1.1),int(img_final_width*1.1))),
#     #     transforms.RandomCrop((img_final_height,img_final_width), padding=4),
#     #     # transforms.Resize(224),
#     #     transforms.ToTensor(),
#     #     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                          std=[0.229, 0.224, 0.225])
#     # ])
#     # transform_train = transforms.Compose([
#     #         transforms.Resize((256,128), interpolation=transforms.InterpolationMode.BICUBIC),
#     #         transforms.RandomHorizontalFlip(0.5),
#     #         transforms.Pad(10),
#     #         transforms.RandomCrop((256,128)),
#     #         transforms.ToTensor(),
#     #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     #         RandomErasing(probability=0.5, mean=([0.485, 0.456, 0.406]))
#     #     ])



#     transform_train = transforms.Compose([
#         transforms.Resize((int(img_final_height*1.1),int(img_final_width*1.1))),
#         transforms.RandomCrop((img_final_height,img_final_width), padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         RandomErasing(probability=0.5, mean=([0.485, 0.456, 0.406]))
#     ])
#     # transform_train= transforms.Compose([
#     #     iaa.Sequential([
#     #         iaa.flip.Fliplr(p=0.5),
#     #         iaa.flip.Flipud(p=0.5),
#     #         iaa.GaussianBlur(sigma=(0.0, 0.1)),
#     #         iaa.MultiplyBrightness(mul=(0.65, 1.35)),
#     #     ]).augment_image,
#     #     transforms.ToTensor()
#     # ])

#     seq = iaa.Sequential([
#         iaa.flip.Flipud(p=0.5),
#         iaa.MultiplyBrightness(mul=(0.65, 1.35)),
#         #iaa.GammaContrast(1.5),
#         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#         iaa.Fliplr(p=0.5), # 水平翻轉影象
#         iaa.GaussianBlur(sigma=(0, 3.0)), # 使用0到3.0的sigma模糊影象
#         # Small gaussian blur with random sigma between 0 and 0.5.
#         # But we only blur about 50% of all images.
#         iaa.Sometimes(0.5,
#                     iaa.GaussianBlur(sigma=(0, 0.5))
#                     ),
#         # Add gaussian noise.
#         # For 50% of all images, we sample the noise once per pixel.
#         # For the other 50% of all images, we sample the noise per pixel AND
#         # channel. This can change the color (not only brightness) of the
#         # pixels.
#         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#         # Make some images brighter and some darker.
#         # In 20% of all cases, we sample the multiplier once per channel,
#         # which can end up changing the color of the images.
#         iaa.Multiply((0.8, 1.2), per_channel=0.8)
#         # Apply affine transformations to each image.
#         # Scale/zoom them, translate/move them, rotate them and shear them.
#     ])

#     #The transform function for test data


#     # trainset.setImgaug(seq)
#     train_set.setTransform(transform_train)
#     return train_set

def testing_augmentation(test_set):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_set.setTransform(transform_test)
    
    return test_set
def training_augmentation(train_set):
    transform= transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_set.setTransform(transform)
    
    return train_set