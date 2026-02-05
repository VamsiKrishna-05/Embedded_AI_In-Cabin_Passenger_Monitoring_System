import cv2
import numpy as np
from ultralytics import YOLO
from config import config

class PassengerDetector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.person_class_id = 0
        
    def detect_passengers(self, frame):
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        passengers = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id == self.person_class_id:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        w = x2 - x1
                        h = y2 - y1
                        
                        passengers.append({
                            'bbox': (int(x1), int(y1), int(w), int(h)),
                            'confidence': confidence
                        })
        
        return passengers
