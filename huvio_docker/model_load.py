import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

weights, view_img, save_txt, imgsz, trace = './models/best.pt', False, True, 640, True
save_img = True
webcam = False

# Initialize
set_logging()
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA


# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, 640)
    
if half:
    model.half()
    
# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
colors = [80,205,180,120]

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1
