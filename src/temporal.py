from typing import Literal, List, Optional
from collections import deque
from src.config import GROWTH_WARNING_MULTIPLIER, GROWTH_CRITICAL_MULTIPLIER, BASELINE_FRAMES, CURRENT_AREA_FRAMES

class FlameTracker:
    def __init__(self):
        """
        Initialize the FlameTracker to monitor the size of detected flames.
        """
        self.recent_areas = deque(maxlen=CURRENT_AREA_FRAMES)
        self.baseline_history = deque(maxlen=BASELINE_FRAMES)
        self.baseline_area = 0.0
        self.is_baseline_locked = False

    def _calculate_area(self, box: list) -> float:
        """
        Calculate the area of a bounding box [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        return float(width * height)

    def update(self, flame_boxes: List[dict], person_present: bool) -> Literal["SAFE", "GROWTH_WARNING", "GROWTH_CRITICAL"]:
        """
        Track flame growth using a smoothed current area and a contextual baseline.
        The baseline only updates when a person is present. When they leave, it locks.
        """
        if not flame_boxes:
            # If no flames are detected, forget the baseline entirely.
            self.reset()
            return "SAFE"

        # Calculate total area of all detected flames in the current frame
        total_current_area = sum(self._calculate_area(item['box']) for item in flame_boxes)

        # 1. Smoothed Current Area
        self.recent_areas.append(total_current_area)
        smoothed_current_area = sum(self.recent_areas) / len(self.recent_areas)

        # 2. Contextual Baseline
        if person_present or self.baseline_area <= 0:
            # Person is present -> Actively cooking
            # OR this is the very first detection
            # Slowly update the baseline to reflect intentional changes in flame size
            self.is_baseline_locked = False
            self.baseline_history.append(total_current_area)
            if len(self.baseline_history) > 0:
                self.baseline_area = sum(self.baseline_history) / len(self.baseline_history)
        else:
            # Person left -> Lock the baseline
            self.is_baseline_locked = True
            
        # If we haven't established a valid baseline yet, assume safe
        if self.baseline_area <= 0:
            return "SAFE"

        # 3. Growth Check
        growth_ratio = smoothed_current_area / self.baseline_area

        if growth_ratio >= GROWTH_CRITICAL_MULTIPLIER:
            return "GROWTH_CRITICAL"
        elif growth_ratio >= GROWTH_WARNING_MULTIPLIER:
            return "GROWTH_WARNING"
        
        return "SAFE"

    def reset(self):
        """
        Reset the tracker when no flame is detected.
        """
        self.recent_areas.clear()
        self.baseline_history.clear()
        self.baseline_area = 0.0
        self.is_baseline_locked = False
        
    def get_stats(self) -> dict:
        """
        Return tracking statistics for debugging/UI.
        """
        smoothed_area = 0.0
        if self.recent_areas:
            smoothed_area = sum(self.recent_areas) / len(self.recent_areas)
            
        return {
            "baseline_area": self.baseline_area,
            "locked": self.is_baseline_locked,
            "smoothed_current": smoothed_area
        }

class ShutoffDebouncer:
    def __init__(self, alpha=0.4, threshold=0.85):
        self.alpha = alpha
        self.threshold = threshold
        self.ema_value = 0.0
        
    def update(self, is_critical_this_frame: bool) -> bool:
        current = 1.0 if is_critical_this_frame else 0.0
        self.ema_value = self.alpha * current + (1.0 - self.alpha) * self.ema_value
        return self.ema_value >= self.threshold
        
    def reset(self):
        self.ema_value = 0.0
