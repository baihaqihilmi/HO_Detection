import torch
from pathlib import Path
from ultralytics import YOLO
import os.path as osp
model = YOLO(osp.join('model' , 'best.pt'))

model.export(format='openvino' )
