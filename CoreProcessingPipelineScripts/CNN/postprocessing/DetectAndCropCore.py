#!/usr/bin/env python

from ultralytics import YOLO

model = YOLO('yolov8n-obb.pt')

results = model("/Users/miroslav.polacek/Pictures/TC_selection_of_cores/F4b_P1.png")
