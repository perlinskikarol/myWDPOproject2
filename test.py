import json
from ultralytics import YOLO
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm

img_path = "0040.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#TODO: Implement detection method.
model = YOLO('last.pt')
# results = model.predict(img, conf=0.50, iou=0.4)
results = model.predict(img, conf=0.45)
result = results[0]
aspen = 0
birch = 0
hazel = 0
maple = 0
oak = 0
for box in result.boxes:
    class_id = box.cls[0].item()
    class_id = int(class_id)
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    conf = round(box.conf[0].item(), 2)
    print("Object type:", class_id)
    print("Coordinates:", cords)
    print("Probability:", conf)
    print("---")

