
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import Net
from custom_dataset import custom_dataset_skewed_food
#modelPath='./trainedModel/skewModel.pt'
modelPath='./trainedModel/skew_checkpoint.t7'


transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

testset = custom_dataset_skewed_food("testing")
testset.setTransform(transform_test)
#trainset= custom_dataset_skewed_food("training",transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=0)



#net = torch.load(modelPath,map_location=torch.device('cpu'))

######################################training statics
#net = Net()
net = models.resnet18(pretrained=False);
# replace the last layer
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 10)

checkpoint = torch.load(modelPath,torch.device('cpu'))

net.load_state_dict(checkpoint['net'])
#net.cuda()

class_count = checkpoint['class_count']
class_correct_count = checkpoint['class_correct_count']
print("training statics")
print("trainging total accuracy:%f" %(checkpoint['acc']))
print("epoch %d" %(checkpoint['parameters']['epoch']))
for i in range(10):
    print('Class %d : %.2f %d/%d' % \
          (i,class_correct_count[i]/class_count[i],class_correct_count[i],class_count[i]))

print("actual_class_count")
print(checkpoint['actual_class_count'])
############

print('Finished Loading')
print('==> Testing model..')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net.eval()

# 每一個epoch都會做完整個dataset
running_loss = 0.0
correct = 0
# inputs 是一個batch 的image labels是一個batch的label
class_count = [0] * 10
class_correct_count = [0]*10


for i, (inputs, labels) in enumerate(testloader, 0):
    # change the type into cuda tensor
    if i==0 :
        print("type of inputs",type(inputs))
        print(inputs.shape)
    inputs = inputs.to(device)
    labels = labels.to(device)
    if i==0 :
        print("type of inputs after to device",type(inputs))

    # forward + backward + optimize
    outputs = net(inputs)
    # select the class with highest probability
    if i==0 :
        print("outputs",outputs)
        print("type of outputs",type(outputs))
        print("len of outputs",len(outputs))
    _, pred = outputs.max(1)
    if i==0 :
        print("pred",pred)
    # if the model predicts the same results as the true
    # label, then the correct counter will plus 1
    correct += pred.eq(labels).sum().item()
    ans_list = pred.eq(labels)
    for i in range(labels.shape[0]):
        class_count[labels[i]] += 1
        if ans_list[i]:
            class_correct_count[labels[i]] += 1


    #class_count[labels] += 1
    #class_correct_count[labels] += result

print('total testing accuracy: %.4f %d/%d' % (correct / len(testset),correct,len(testset)))
for i in range(10):
    print('Class %d : %.2f %d/%d' % \
          (i,class_correct_count[i]/class_count[i],class_correct_count[i],class_count[i]))

print('Finished Training')
