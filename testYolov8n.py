from ultralytics import YOLO
model = YOLO('yolov8n.pt')
import os
import time


folder_path = "D:\\TestYOLOv8n\\Data"

image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
image_paths  = []
for i, image_file in enumerate(image_files):
    image_paths.append(os.path.join(folder_path, image_file))

st = time.time()
for i, image_path in enumerate(image_paths):
    results = model(image_path)
ed = time.time()

print((ed - st) / 10)