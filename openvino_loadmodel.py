from __future__ import print_function
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from custom_dataset import custom_dataset_skewed_food

import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore

batch_size =49


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
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)



#net = torch.load(modelPath,map_location=torch.device('cpu'))


################################################

#!/usr/bin/env python
"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 
 change the net.input_info to net.inputs
"""


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                       default="/workspace/DLSR_LAB_learning/trainedModel/super_resolution.xml",
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                     default="/workspace/DLSR_LAB_learning/food11re/evaluation/",
                      type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)

    return parser


def maintest():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    print("net inputs",net.inputs)
    print("net outputs",net.outputs)
    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.input)
    print("input_blob",input_blob)
    print("out_blob",out_blob)
    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape
    print("input shape",net.inputs[input_blob].shape)
    images = np.ndarray(shape=(n, c, h, w))
    import glob
    args.input = glob.glob(args.input + '/*/*')
    for i in range(n):
        image = cv2.imread(args.input[i])
        print("image shape",image.shape)
        
        if image.shape[:-1] != (h, w):
            log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image
    log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    res = exec_net.infer(inputs={input_blob: images})

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]
    print("res",res)
    log.info("Top {} results: ".format(args.number_top))
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
    classid_str = "classid"
    probability_str = "probability"
    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-args.number_top:][::-1]
        print("Image {}\n".format(args.input[i]))
        print(classid_str, probability_str)
        print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
        for id in top_ind:
            det_label = labels_map[id] if labels_map else "{}".format(id)
            label_length = len(det_label)
            space_num_before = (len(classid_str) - label_length) // 2
            space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
            print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                          ' ' * space_num_after, ' ' * space_num_before_prob,
                                          probs[id]))
        print("\n")
    log.info("This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n")

############################## 設定network
log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
args = build_argparser().parse_args()
model_xml = args.model
model_bin = os.path.splitext(model_xml)[0] + ".bin"

# Plugin initialization for specified device and load extensions library if specified
log.info("Creating Inference Engine")
ie = IECore()
if args.cpu_extension and 'CPU' in args.device:
    ie.add_extension(args.cpu_extension, "CPU")
# Read IR
log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
net = ie.read_network(model=model_xml, weights=model_bin)

if "CPU" in args.device:
    supported_layers = ie.query_network(net, "CPU")
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                  format(args.device, ', '.join(not_supported_layers)))
        log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                  "or --cpu_extension command line argument")
        sys.exit(1)

assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
assert len(net.outputs) == 1, "Sample supports only single output topologies"
print("net inputs",net.inputs)
print("net outputs",net.outputs)
log.info("Preparing input blobs")
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
net.batch_size = len(args.input)
print("input_blob",input_blob)
print("out_blob",out_blob)
# Read and pre-process input images
n, c, h, w = net.inputs[input_blob].shape
print("input shape",net.inputs[input_blob].shape)
log.info("Batch size is {}".format(n))

# Loading model to the plugin
log.info("Loading model to the plugin")
exec_net = ie.load_network(network=net, device_name=args.device)


############

print('Finished Loading')
print('==> Testing model..')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 每一個epoch都會做完整個dataset
running_loss = 0.0
correct = 0
# inputs 是一個batch 的image labels是一個batch的label
class_count = [0] * 10
class_correct_count = [0]*10


for i, (inputs, labels) in enumerate(testloader, 0):
    # change the type into cuda tensor
    if len(inputs) < batch_size:
        break
    inputs = inputs.to(device)
    labels = labels.to(device)

    # forward + backward + optimize
    outputs = exec_net.infer(inputs={input_blob: inputs})
    
    outputs = torch.from_numpy(outputs[out_blob])
    # select the class with highest probability
    print("type of outputs",type(outputs))
    print("outputs",outputs)
    print(len(outputs.max(1)))
    print(type(outputs.max(1)))
    _, pred = outputs.max(1)
    print("pred",pred)
    print("labels",labels)
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
