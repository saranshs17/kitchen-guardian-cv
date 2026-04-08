# Kitchen Guardian: Smart Retrofit Stove Safety

An IoT safety device that monitors a stove using Computer Vision. The system runs on an edge device (like a Raspberry Pi or Android device) and uses a camera paired with multiple YOLO models to detect people, stove flames, and dangerous fires. It combines spatial zone validation, temporal flame growth tracking, and a state machine to make intelligent shutoff decisions — distinguishing safe cooking from emergencies.

---

## Features

To reduce false positives and ensure the reliable actuation of a physical shutoff motor, this application goes beyond simple object detection by utilizing robust **Spatial** and **Temporal** state machine logic.

### Spatial Logic
The vision model maps detected flames against pre-defined safe zones (the stove burners) to understand their context.
- **Interactive Calibration:** A click-and-drag UI lets users define circular safe zones around their existing stove burners.
- **Base-Anchored Zone Validation:** Because safe cooking flames can grow tall and lean outside the strictly defined zone, our spatial tracking specifically anchors to the lower 15% of the bounding box. If the base of the fire is on the stove, it's considered safe; if it spreads beyond the boundaries, it triggers an immediate critical state.

### Temporal Logic
Accidental fires behave differently than cooking fires, they grow rapidly and out of control. 
- **Context-Aware Flame Growth Tracking:** We implement a sliding-window memory that tracks flame area over time. When a cook is present, the system continuously updates a *contextual baseline* of normal flame size. The moment the cook leaves, this baseline locks. If the flame area drastically multiplies compared to this baseline while unattended, it indicates an escalating fire and triggers an emergency shutoff.

### Signal Reliability
Actuating a physical motor requires certainty to prevent annoying false alarms while cooking.
- **EMA Shutoff Debouncing:** Frame dropouts or bounding box jitters from the YOLO model could cause the state machine to aggressively bounce between `SAFE`, `WARNING`, and `CRITICAL`. We solve this by applying an Exponential Moving Average (EMA) filter to the state outputs. A full system shutoff only commits when the confidence signal sustains a high threshold over multiple frames.

### System Interfaces
- **Multi-Model Vision Pipeline:** Separate YOLO models running in parallel for person detection (YOLOv8n), stove flame detection, and general fire detection.
- **Dual Edge Architecture:** The system provides a native OpenCV window for direct hardware I/O and calibration, running concurrently with a Streamlit WebRTC dashboard for remote browser monitoring.
---

## Getting Started

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Download the trained detection models from `Releases` and place them in a `models` directory in the project root.

4. Run the native OpenCV application:
```bash
python main.py
```

5. Or run the Streamlit web dashboard:
```bash
streamlit run dashboard.py
```

### UI Controls (main.py)

| Control | Action |
|---|---|
| **Left Click + Drag** | Draw a burner safe zone |
| **Right Click** | Clear all burner zones |
| **`s`** | Save burner zones to file |
| **`r`** | Reset flame tracker & debouncer |
| **`f`** | Toggle mock flame (debug mode) |
| **`q`** | Quit |

---

For deeper technical details, including class diagrams, sequence diagrams, and the dashboard threading architecture, see the [Architecture Documentation](docs/architecture.md).
