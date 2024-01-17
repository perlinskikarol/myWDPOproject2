from ultralytics import YOLO
import torch

model = YOLO("yolov8m.pt")

def startTraining():
    model.train(data="data.yaml", epochs=600, patience=800, batch=16)

if __name__ == "__main__":
    startTraining()
