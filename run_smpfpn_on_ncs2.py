#! /usr/bin/python3

import time
import torch
import os

import segmentation_models_pytorch as smp

from openvino.inference_engine import IENetwork, IECore


DEVICE = "MYRIAD"
MODEL = "efficientnet-b3"
CLASSES = ['leaf']
ACTIVATION = 'sigmoid'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = "MYRIAD"  # "CPU" | "MYRIAD" | "GPU"

model = smp.FPN(
    encoder_name=MODEL,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)
model.encoder.set_in_channels(1)
model.encoder.set_swish(memory_efficient=False)  # required for correct export to onnx

inputs = torch.randn((1, 1, 128, 128))

torch.onnx.export(model, inputs, f"smp_{MODEL}.onnx", opset_version=11)  # opset_ver 11 is required for interpolation

os.system(f"mo_onnx.py --input_model smp_{MODEL}.onnx")

# RUN on accelerator
ie = IECore()
net = IENetwork(model=f"smp_{MODEL}.xml", weights=f"smp_{MODEL}.bin")

# Set up the input and output blobs
gn_input_blob = next(iter(net.inputs))
gn_output_blob = next(iter(net.outputs))
gn_input_shape = net.inputs[gn_input_blob].shape
gn_output_shape = net.outputs[gn_output_blob].shape

print("Input_shape:", gn_input_shape)
print("Output_shape:", gn_output_shape)

# Load the network
gn_exec_net = ie.load_network(network=net, device_name=DEVICE)

# execute and measure time
start_time = time.time()
gn_exec_net.infer({gn_input_blob: inputs.numpy()})
print("End time:", time.time() - start_time)

print("Done.")
