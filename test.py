
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


def main():
    testset = brid("testing")
    testset.setTransform(transform_test)
    #trainset= custom_dataset_skewed_food("training",transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
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
    resultDict={}
    for i, (inputs, imageNames) in enumerate(testloader, 0):
        print("batch:",i)
        # change the type into cuda tensor
        inputs = inputs.to(device)

        # forward + backward + optimize
        outputs = net(inputs)
        # select the class with highest probability
        _, pred = outputs.max(1)
        for i in range(len(inputs)):
            resultDict[imageNames[i]] = pred[i].cpu().numpy().tolist()

    print("resulDict",resultDict)
    writeResult2Answer(testset,resultDict)
        

def writeResult2Answer( dataset,resultDict):
    with open('/workspace/CV/hw1/2021VRDL_HW1_datasets/testing_img_order.txt') as f:
        test_images = [x.strip() for x in f.readlines()]  # all the testing images

    submission = []
    for img in test_images:  # image order is important to your result
        predicted_class = resultDict[img]  # the predicted category
        submission.append([img,dataset.labelValue2Label(predicted_class)])
    np.savetxt('answer.txt', submission, fmt='%s')


if __name__ == "__main__":
    main()