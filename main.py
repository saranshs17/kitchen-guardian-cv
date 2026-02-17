import cv2
import time
from src.config import CAMERA_INDEX
from src.detectors import VisionSystem
from src.state_machine import SafetyGuardian

def main():
    # Initialize Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
        return

    # Initialize Systems
    print("Initializing Vision System...")
    vision_system = VisionSystem()
    print("Initializing Safety Guardian...")
    guardian = SafetyGuardian()

    # Mock flame state
    mock_flame_active = False

    print("System detection started. Press 'q' to quit.")
    print("Press 'f' to toggle mock flame.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # 1. Detect Objects
            detection_result = vision_system.detect_objects(frame)
            
            # Override flame detection with mock state
            # In a real system, detection_result['flame_detected'] would come from the model
            detection_result['flame_detected'] = mock_flame_active

            # 2. Update Safety Status
            status = guardian.update_status(
                flame_on=detection_result['flame_detected'],
                person_present=detection_result['person_detected']
            )

            # 3. Visualization
            # Draw Bounding Boxes
            for item in detection_result['boxes']:
                box = item['box']
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{item['class']} {item['conf']:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw Status Info
            # Top Left: System Status
            status_color = (0, 255, 0) # Green for SAFE
            if status == "WARNING":
                status_color = (0, 255, 255) # Yellow
            elif status == "CRITICAL_SHUTOFF":
                status_color = (0, 0, 255) # Red

            cv2.putText(frame, f"STATUS: {status}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Draw Flame Status
            flame_text = "FLAME: ON" if mock_flame_active else "FLAME: OFF"
            flame_color = (0, 0, 255) if mock_flame_active else (255, 255, 255)
            cv2.putText(frame, flame_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, flame_color, 2)
            
            # Draw Person Status
            person_text = "PERSON: YES" if detection_result['person_detected'] else "PERSON: NO"
            cv2.putText(frame, person_text, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show Frame
            cv2.imshow('Kitchen Guardian', frame)

            # 4. Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                mock_flame_active = not mock_flame_active
                print(f"Mock Flame toggled: {mock_flame_active}")

    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("System shutdown.")

if __name__ == "__main__":
    main()
