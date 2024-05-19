     
DATA_DIR = '/home/hypersoft-gpu/Mahrukh_Ali_Khan/test/coco/train/images'
     
import os

from ultralytics import YOLO


model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
model.train(data='/home/hypersoft-gpu/Mahrukh_Ali_Khan/test/coco/data.yaml', epochs=100, imgsz=640)