
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import Net
from custom_dataset import brid
from custom_dataset import custom_dataset_skewed_food
#modelPath='./trainedModel/skewModel.pt'
modelPath='./trainedModel/skew_checkpoint.t7'

def validation():
    num_of_class = 200


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
    #net = Net()
    net = models.resnet50(pretrained=False);
    # replace the last layer
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features,num_of_class )

    checkpoint = torch.load(modelPath,torch.device('cpu'))

    net.load_state_dict(checkpoint['net'])
    net.cuda()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.eval()


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
        


if __name__ == "__main__":
    validation()