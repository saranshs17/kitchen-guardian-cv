import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any, List
from src.config import CONFIDENCE_THRESHOLD

class VisionSystem:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize the VisionSystem with a YOLO model.
        """
        self.model = YOLO(model_path)
        self.classes = self.model.names

    def detect_objects(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect objects in the frame.
        Returns a dictionary with detection status and bounding boxes.
        """
        results = self.model(frame, stream=True, verbose=False)
        
        person_detected = False
        flame_detected = False
        boxes = []

        # Process results
        for r in results:
            for box in r.boxes:
                # Get class ID and confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                # Class 0 is 'person' in COCO dataset
                if cls_id == 0:
                    person_detected = True
                    # Store box coordinates for visualization [x1, y1, x2, y2]
                    coords = box.xyxy[0].tolist()
                    boxes.append({'class': 'person', 'box': coords, 'conf': conf})

        # Mock Flame Detection
        # In a real scenario, this would be a custom trained model or class
        # For now, we simulate it. 
        # Let's say we check if 'f' key is pressed in main loop, but here 
        # we can just use a placeholder mechanism or it acts as a pass-through.
        # The key press handling is better done in main.py where cv2.waitKey is called.
        # For simplicity in this method, we will default to False, and allow main.py to override or 
        # we can check for a specific visual feature if we wanted (like a red color blob).

        return {
            'person_detected': person_detected,
            'flame_detected': flame_detected, 
            'boxes': boxes
        }
