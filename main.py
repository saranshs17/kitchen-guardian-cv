import cv2
import time
import json
import os
from src.config import BURNER_RADIUS_PIXELS, BURNERS_FILE, VIDEO_SOURCE, ENABLE_DEBUG_MOCK_FLAME, SHUTOFF_ALPHA, SHUTOFF_THRESHOLD
from src.detectors import VisionSystem
from src.state_machine import SafetyGuardian
from src.temporal import FlameTracker, ShutoffDebouncer

burner_zones = []
mock_flame_pos = None
drawing_center = None
drawing_radius = 0

def load_burners():
    global burner_zones
    if os.path.exists(BURNERS_FILE):
        try:
            with open(BURNERS_FILE, 'r') as f:
                burner_zones = json.load(f)
            print(f"Loaded {len(burner_zones)} burner zones.")
        except Exception as e:
            print(f"Failed to load burners: {e}")

def save_burners():
    try:
        with open(BURNERS_FILE, 'w') as f:
            json.dump(burner_zones, f)
        print(f"Saved {len(burner_zones)} burner zones to {BURNERS_FILE}.")
    except Exception as e:
        print(f"Failed to save burners: {e}")

def handle_mouse_events(event, x, y, flags, param):
    global burner_zones, mock_flame_pos, drawing_center, drawing_radius
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a new burner zone radius
        drawing_center = (x, y)
        drawing_radius = 0
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_center is not None:
            # Dynamically calculate the Euclidean distance for the radius
            drawing_radius = int(((x - drawing_center[0])**2 + (y - drawing_center[1])**2)**0.5)
            
        if ENABLE_DEBUG_MOCK_FLAME:
            # Track mouse for mock flame if active
            mock_flame_pos = (x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing_center is not None:
            # Default to 100
            final_radius = drawing_radius if drawing_radius > 5 else BURNER_RADIUS_PIXELS
            burner_zones.append((drawing_center[0], drawing_center[1], final_radius))
            print(f"Added Burner Zone at {drawing_center} with radius {final_radius}")
            drawing_center = None
            drawing_radius = 0
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Clear all burner zones
        burner_zones = []

def draw_status_panel(frame, status, detection_result, growth_status, flame_tracker, mock_flame_active):
    
    y = 40
    line_gap = 35

    def draw(text, color=(255, 255, 255), scale=0.7):
        nonlocal y
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2
        )
        y += line_gap

    status_color = (0, 255, 0)
    if "WARNING" in status:
        status_color = (0, 255, 255)
    elif "CRITICAL" in status:
        status_color = (0, 0, 255)

    draw(f"STATUS: {status}", status_color, 0.8)

    flame_text = "FLAME: DETECTED" if detection_result["flame_detected"] else "FLAME: NONE"
    flame_color = (0, 0, 255) if detection_result["flame_detected"] else (255, 255, 255)
    draw(flame_text, flame_color)

    fire_text = "FIRE: DETECTED" if detection_result["dangerous_fire"] else "FIRE: NONE"
    fire_color = (0, 0, 255) if detection_result["dangerous_fire"] else (255, 255, 255)
    draw(fire_text, fire_color)

    person_text = "PERSON: YES" if detection_result["person_detected"] else "PERSON: NO"
    draw(person_text)

    stats = flame_tracker.get_stats()
    if stats["baseline_area"] > 0:
        stats_text = f"FLAME AREA: {stats['smoothed_current']:.0f} (Base: {stats['baseline_area']:.0f})"
        if stats["locked"]:
            stats_text += " [LOCKED]"
        if growth_status != "SAFE":
            stats_text += f" [{growth_status}]"
        stats_color = (0, 0, 255) if growth_status != "SAFE" else (255, 255, 255)
        draw(stats_text, stats_color)

    if ENABLE_DEBUG_MOCK_FLAME:
        mock_text = f"MOCK FLAME: {'ON' if mock_flame_active else 'OFF'}"
        draw(mock_text, (255, 200, 0))

    controls = "L-Click add burner | R-Click clear | s save | f mock | r reset | q quit"
    cv2.putText(frame, controls, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

def main():
    global mock_flame_pos
    # Initialize Camera
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {VIDEO_SOURCE}.")
        return

    # Initialize Systems
    print("Initializing Vision System...")
    vision_system = VisionSystem()
    print("Initializing Safety Guardian...")
    guardian = SafetyGuardian()
    print("Initializing Flame Tracker...")
    flame_tracker = FlameTracker()
    debouncer = ShutoffDebouncer(alpha=SHUTOFF_ALPHA, threshold=SHUTOFF_THRESHOLD)

    # Load previously saved burners
    load_burners()

    # Mock flame state
    mock_flame_active = False

    print("System detection started.")
    print("UI Controls:")
    print(" - Left Click: Add Burner Center")
    print(" - Right Click: Clear All Burners")
    print(" - Press 's': Save Burners to File")
    if ENABLE_DEBUG_MOCK_FLAME:
        print(" - Press 'f': Toggle Mock Flame (follows mouse)")
    print(" - Press 'r': Reset Flame Tracker")
    print(" - Press 'q': Quit")

    cv2.namedWindow('Kitchen Guardian')
    cv2.setMouseCallback('Kitchen Guardian', handle_mouse_events)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop local video files
                if isinstance(VIDEO_SOURCE, str) and os.path.isfile(VIDEO_SOURCE):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # For live streams, wait briefly and try again
                time.sleep(0.02)
                continue

            # Generate the mock flame box if active
            current_mock_box = None
            if ENABLE_DEBUG_MOCK_FLAME and mock_flame_active and mock_flame_pos:
                mx, my = mock_flame_pos
                # Create a 50x50 box around the mouse cursor
                current_mock_box = [mx - 25, my - 25, mx + 25, my + 25]

            # 1. Detect Objects & Validate Zones
            detection_result = vision_system.detect_objects(
                frame=frame, 
                burner_zones=burner_zones, 
                mock_flame_box=current_mock_box
            )

            # Filter boxes to only contain heat sources (flames + fires)
            heat_boxes = detection_result["flame_boxes"] + detection_result["fire_boxes"]
            
            # Update temporal logic (Growth tracking)
            growth_status = flame_tracker.update(flame_boxes=heat_boxes, person_present=detection_result['person_detected'])
            
            status = guardian.update_status(
                flame_on=detection_result["flame_detected"],
                person_present=detection_result["person_detected"],
                growth_status=growth_status,
            )
            
            # If the fire is NOT safe spatially, we override the temporal state machine
            is_safe_fire = detection_result['is_safe_fire']
            
            if (detection_result["dangerous_fire"] or detection_result["flame_detected"]) and not is_safe_fire:
                status = "CRITICAL_SHUTOFF (OUTSIDE SAFE ZONE)"
                
            # EMA Debouncing logic to suppress flickering UI
            is_critical = "CRITICAL_SHUTOFF" in status
            actual_shutoff = debouncer.update(is_critical)
            
            if is_critical and not actual_shutoff:
                # Downgrade visually to a critical warning until the EMA breaches 0.85
                status = status.replace("CRITICAL_SHUTOFF", "CRITICAL_WARNING")
                
            elif actual_shutoff:
                # This locks in the irreversible status
                pass

            # 3. Visualization
            # Draw Bounding Boxes
            for item in detection_result['boxes']:
                box = item['box']
                x1, y1, x2, y2 = map(int, box)
                if item["class"] == "person":
                    color = (0, 255, 0)
                elif item["class"] == "flame":
                    color = (0, 140, 255)
                else:  
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{item['class']} {item['conf']:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw the spatial anchor point if we computed it (for flames/fires)
                if "anchor_point" in item:
                    ax, ay = item["anchor_point"]
                    # Draw a tiny magenta dot to represent where the flame is anchored
                    cv2.circle(frame, (ax, ay), 4, (255, 0, 255), -1)

            # Draw Burner Zones
            for zx, zy, r in burner_zones:
                # Draw the radius
                cv2.circle(frame, (zx, zy), r, (255, 150, 0), 2)
                # Draw the center point
                cv2.circle(frame, (zx, zy), 4, (255, 150, 0), -1)

            # Draw preview ring if actively dragging the mouse
            global drawing_center, drawing_radius
            if drawing_center is not None:
                cv2.circle(frame, drawing_center, drawing_radius, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw mock flame box explicitly if active
            if ENABLE_DEBUG_MOCK_FLAME and mock_flame_active and current_mock_box is not None:
                x1, y1, x2, y2 = current_mock_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(
                    frame,
                    "MOCK FLAME",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 140, 255),
                    2,
                )

            draw_status_panel(frame, status, detection_result, growth_status, flame_tracker, mock_flame_active)

            # Show Frame
            cv2.imshow('Kitchen Guardian', frame)

            # 4. Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_burners()
            elif key == ord("r"):
                flame_tracker.reset()
                debouncer.reset()
                print("Flame tracker reset.")
            elif key == ord('f') and ENABLE_DEBUG_MOCK_FLAME:
                mock_flame_active = not mock_flame_active
                print(f"Mock Flame toggled: {mock_flame_active}")
                if not mock_flame_active:
                    mock_flame_pos = None
                    flame_tracker.reset()

    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("System shutdown.")

if __name__ == "__main__":
    main()
