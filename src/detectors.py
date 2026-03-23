import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Any, List, Optional, Tuple
from src.config import PERSON_CONFIDENCE_THRESHOLD,FLAME_CONFIDENCE_THRESHOLD, FIRE_CONFIDENCE_THRESHOLD, BURNER_RADIUS_PIXELS, FLAME_MODEL_PATH, PERSON_MODEL_PATH, FIRE_MODEL_PATH
from src.spatial import calculate_center, is_point_in_zones

class VisionSystem:
    def __init__(self, person_model_path: str = PERSON_MODEL_PATH,flame_model_path: str = FLAME_MODEL_PATH, fire_model_path: str = FIRE_MODEL_PATH):
        """
        Initialize separate models for person detection, stove flame detection,
        and general fire detection.
        """

        if not Path(flame_model_path).exists():
            raise FileNotFoundError(
                f"Flame model not found: {flame_model_path}\n"
                f"Place stove_fire_best.pt inside the models/ directory."
            )
        
        if not Path(fire_model_path).exists():
            raise FileNotFoundError(
                f"Fire model not found: {fire_model_path}\n"
                f"Place fire_best.pt inside the models/ directory."
            )
        
        self.person_model = YOLO(person_model_path)
        self.flame_model = YOLO(flame_model_path)
        self.fire_model = YOLO(fire_model_path)

    def _detect_persons(self, results) -> Tuple[bool, List[Dict[str, Any]]]:

        boxes = []
        detected = False

        for r in results:
            for box in r.boxes:

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # COCO class 0 = person
                if cls_id == 0 and conf >= PERSON_CONFIDENCE_THRESHOLD:

                    coords = box.xyxy[0].tolist()

                    boxes.append(
                        {
                            "class": "person",
                            "box": coords,
                            "conf": conf,
                        }
                    )

                    detected = True

        return detected, boxes
    
    def _detect_flames(self, results) -> Tuple[bool, List[Dict[str, Any]]]:

        boxes = []
        detected = False

        for r in results:
            for box in r.boxes:

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Custom model → only class 0 = stove-fire
                if cls_id == 0 and conf >= FLAME_CONFIDENCE_THRESHOLD:

                    coords = box.xyxy[0].tolist()

                    boxes.append(
                        {
                            "class": "flame",
                            "box": coords,
                            "conf": conf,
                        }
                    )

                    detected = True

        return detected, boxes
    
    def _detect_fire(self, results):

        boxes = []
        detected = False

        for r in results:
            for box in r.boxes:

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id == 0 and conf >= FIRE_CONFIDENCE_THRESHOLD:

                    coords = box.xyxy[0].tolist()

                    boxes.append({
                        "class": "fire",
                        "box": coords,
                        "conf": conf
                    })

                    detected = True

        return detected, boxes
    
    def detect_objects(self, frame: np.ndarray, burner_zones: List[tuple], mock_flame_box: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Detect objects in the frame and validate against burner zones.
        Args:
            frame: Video frame.
            burner_zones: List of (x, y) coordinates for safe burners.
            mock_flame_box: Optional [x1, y1, x2, y2] to simulate a fire.
        Returns a dictionary with detection status and bounding boxes.
        """
        # results = self.model(frame, stream=True, verbose=False)
        person_results = self.person_model(frame, verbose=False)
        flame_results = self.flame_model(frame, verbose=False)
        fire_results = self.fire_model(frame, verbose=False)

        person_detected, person_boxes = self._detect_persons(person_results)
        flame_detected, flame_boxes = self._detect_flames(flame_results)
        fire_detected, fire_boxes = self._detect_fire(fire_results)

        # Demo override: use the mock flame instead of real flame detections
        if mock_flame_box is not None:
            flame_detected = True
            flame_boxes = [
                {
                    "class": "flame",
                    "box": mock_flame_box,
                    "conf": 1.0,
                }
            ]
        
        is_safe_fire = True
        
        all_boxes = person_boxes + flame_boxes + fire_boxes
        heat_boxes = flame_boxes + fire_boxes

        if heat_boxes and burner_zones:

            for item in heat_boxes:

                center = calculate_center(item["box"])

                if not is_point_in_zones(center, burner_zones, BURNER_RADIUS_PIXELS):
                    is_safe_fire = False
                    break

        return {
            "person_detected": person_detected,
            "flame_detected": flame_detected,
            "is_safe_fire": is_safe_fire,
            "dangerous_fire": fire_detected,
            "boxes": all_boxes,
            "person_boxes": person_boxes,
            "flame_boxes": flame_boxes,
            "fire_boxes": fire_boxes,
        }
