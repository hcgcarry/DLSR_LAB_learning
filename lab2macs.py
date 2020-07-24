from torchvision.models import resnet50
import torch
from thop import profile
from torchvision.models import mobilenet_v2
modelResNet = resnet50()
input = torch.randn(1, 3, 224, 224)

macs, params = profile(modelResNet, inputs=(input, ))
print("ResNet:")
print("Total parameter: %.5fM" % (params*1e-06))
print("Total MACs: %.5fM" % (macs*1e-06))

mobileNet= mobilenet_v2()
input = torch.randn(1, 3, 224, 224)

macs, params = profile(mobileNet, inputs=(input, ))
print("mobileNetV2")
print("Total parameter: %.5fM" % (params*1e-06))
print("Total MACs: %.5fM" % (macs*1e-06))
