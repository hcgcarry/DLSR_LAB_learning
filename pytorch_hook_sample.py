import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from model import Net
import numpy



def main():
    model = Net()
    #model.conv1.register_forward_hook(my_hook_function)
    #input_data = torch.randn(1, 3, 224, 224)
    #out = model(input_data)
    model.reg_forward_hook()
    model.thopCaclate()
    #input_data = torch.randn(1, 3, 224, 224)
    #output=model(input_data)
    model.getTotalParameterCount()
    model.getTotalMACsCount()
    model.getTotalFLOPsCount()

if __name__ == '__main__':
    main()

