
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import Net
from custom_dataset import custom_dataset_skewed_food
#modelPath='./trainedModel/skewModel.pt'
modelPath='./trainedModel/skew_checkpoint.t7'


batch_size = 32

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


net.eval()


# Input to the model
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

# Export the model
torch.onnx.export(net,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "./trainedModel/super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'] # the model's output names
                )
