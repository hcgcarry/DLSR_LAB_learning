
import torch
import torchvision
import torchvision.transforms as transforms
from model import Net
from custom_dataset import custom_dataset_skewed_food

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
testset = custom_dataset_skewed_food("testing",transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=0)
net = torch.load('./model.pt',map_location=torch.device('cpu'))

print('Finished Loading')
print('==> Testing model..')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net.eval()

# 每一個epoch都會做完整個dataset
running_loss = 0.0
correct = 0
# inputs 是一個batch 的image labels是一個batch的label
for i, (inputs, labels) in enumerate(testloader, 0):
    # change the type into cuda tensor
    inputs = inputs.to(device)
    labels = labels.to(device)

    # forward + backward + optimize
    outputs = net(inputs)
    # select the class with highest probability
    _, pred = outputs.max(1)
    # if the model predicts the same results as the true
    # label, then the correct counter will plus 1
    correct += pred.eq(labels).sum().item()

print('testing accuracy: %.4f' % (100. * correct / len(testset)))
print('Finished Training')
