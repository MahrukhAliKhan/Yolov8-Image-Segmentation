from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
#model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('path to your trained model', task='segment')  # load a custom model
image = cv2.imread('test1.jpg')
# Predict with the model
results = model(image, imgsz=640)  # predict on an image

if results[0].masks is not None:
    mask_shape = (image.shape[0], image.shape[1])
    mask = np.zeros(mask_shape, dtype=np.uint8)
    segments = results[0].masks.xyn
    for i in segments:
        polygon_vertices_rescaled = np.round(i * np.array(mask_shape[::-1])).astype(np.int32)
        cv2.fillPoly(mask, [polygon_vertices_rescaled], color=(255, 255, 255))

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    result_image = np.where(mask[..., None] == 0, color_image, image)

cv2.imwrite("res.jpg",result_image)
        
    