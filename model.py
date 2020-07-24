
import torch.nn.functional as F
from thop import profile
import torch.nn as nn
import torch
class Net(nn.Module):

    total_FLOPs_count =0
    total_MACs_count = 0
    total_parameter_count = 0
    dropout = 0

    # define the layers
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3,padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(16, 26, 3,padding=1)
        self.fc1 = nn.Linear(26 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()


    def defineDropout(self,dropout):
        self.dropout = dropout

    # concatenate these layers
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 26 * 14 * 14)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

    def reg_forward_hook(self):
        modules = self.named_children()
        for name, module in modules:
            module.register_forward_hook(self.my_hook_function)

    def my_hook_function(self,layer, input, output):
        print("------------------------------------------------------------------------------------------------")
        layer_name=str(layer.__class__.__name__)
        #print("%10d    ".format(layer_name))

        input_shape = str([item for item in input[0].shape])
        output_shape = str([item for item in output.shape])

        layer_params=self.countLayerParam( layer, input, output)
        layer_MACs = self.countLayerMACs(layer,input,output)
        layer_FLOPs = self.countFLOPs(layer,input,output)

        #for index in range(output)
        #print("fdifdis",output.shape.__len__())

        table={'op_type':layer_name,"input_shape":input_shape,"output_shape":output_shape \
            ,"params":layer_params,"MACs":layer_MACs,"FLOPs":layer_FLOPs}

        #print('{op_type:10}'.format(**table))

        print('{0:10} {1:20} {2:20} \
              {3:>10} {4:>10} {5:>10}'.format('op_type','input_shape','output_shape', \
                                              'params','MACs','FLOPs'))

        print('{op_type:10} {input_shape:20} {output_shape:20} \
              {params:10d} {MACs:10d} {FLOPs:10d}'.format(**table))
        '''
        print('{op_type:10} {input_shape:10} '
              .format(**table))
        for param in layer.parameters():
            print("params shape: {}".format(list(param.size())))
        '''
        #print("-----------------------------------------------------------------------------")


    def countLayerMACs(self, layer, input, output):
        ##################count MACs
        layer_name=layer.__class__.__name__
        layer_MACs_count = 0
        if layer_name == "Conv2d":
            layer_MACs_count = 1
            for item in layer.weight.shape:
                layer_MACs_count *= item

            layer_MACs_count *= output.shape[-1]
            layer_MACs_count *= output.shape[-2]

            self.total_MACs_count+= layer_MACs_count
        if layer_name == "Linear":
            layer_MACs_count = 1
            for item in layer.weight.shape:
                layer_MACs_count *= item

            self.total_MACs_count += layer_MACs_count
        return layer_MACs_count


    def countFLOPs(self, layer, input, output):
        ##################count MACs
        layer_name=layer.__class__.__name__
        layer_FLOPs_count = 0
        if layer_name == "Conv2d":
            layer_FLOPs_count = 1
            for item in layer.weight.shape:
                layer_FLOPs_count *= item

            layer_FLOPs_count *= output.shape[-1]
            layer_FLOPs_count *= output.shape[-2]
            layer_FLOPs_count *= 2

            self.total_FLOPs_count += layer_FLOPs_count
        if layer_name == "Linear":
            layer_FLOPs_count = 1
            for item in layer.weight.shape:
                layer_FLOPs_count *= item

            layer_FLOPs_count *= 2
            layer_FLOPs_count += layer.bias.shape[0]
            self.total_FLOPs_count += layer_FLOPs_count

        if layer_name == "MaxPool2d":

            layer_FLOPs_count = pow(layer.kernel_size,2) - 1
            for item in output.shape:
                layer_FLOPs_count *= item

            self.total_FLOPs_count += layer_FLOPs_count
        return layer_FLOPs_count

    def countLayerParam(self, layer, input, output):
        ###############count total parameter
        layer_name=layer.__class__.__name__
        layer_parameter_count = 0
        if layer_name == "Conv2d" or layer_name == "Linear":
            layer_parameter_count = 1
            for item in layer.weight.shape:
                layer_parameter_count *= item
            layer_parameter_count+= layer.bias.shape[0]
            self.total_parameter_count +=layer_parameter_count
            #print("layer parameter: %.5fM" % (layer_parameter_count* 1e-06))

        return layer_parameter_count


    def thopCaclate(self):
        #### thop
        input_data = torch.randn(1, 3, 224, 224)
        print("thop output:")
        macs, params = profile(self, inputs=(input_data,))
        #print("Total parameter: %.2fM" % (params * 1e-06))
        #print("Total MACs: %.2fM" % (macs * 1e-06))
        print("thop Total parameter:",params)
        print("thop Total MACs:",macs)

    def getTotalParameterCount(self):
        print("total_parameter_count",self.total_parameter_count)


    def getTotalMACsCount(self):
        print("total_MACs_count",self.total_MACs_count)
    def getTotalFLOPsCount(self):
        print("total_FLOPs_count",self.total_FLOPs_count)

