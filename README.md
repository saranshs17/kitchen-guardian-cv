# Kitchen Guardian: Smart Retrofit Stove Safety

An IoT safety device that monitors a stove using Computer Vision. The system runs on an edge device (like a Raspberry Pi or Android device) and uses a camera to detect people and flames. It applies a State Machine (future revisions plan to improve upon this) to track unattended cooking and ensure safety.

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

4. Run the edge detection application:
```bash
python main.py
```

---

## Planned Architecture & Features

The long-term vision focuses on building a robust, context-aware safety system that separates the edge computing layer from the physical actuation layer while incorporating advanced logic to understand the cooking context.

### 1. Hardware Actuation
- **Non-Invasive Control**: A motorized clamp mechanism that attaches to the existing stove knob, requiring no permanent modifications to the stove. 
- **Emergency Shutoff**: Capable of physically rotating the knob to the "OFF" position when a critical danger is confirmed.

### 2. Advanced Vision Intelligence
- **Safe Zones (Spatial Logic)**: The vision system will map the camera's view to differentiate between a flame safely contained *inside* the stove’s bounds versus a flame spreading *outside* limits.
- **Temporal Logic (Flame Growth Rate)**: Accidental fires grow rapidly. By tracking the area and growth rate of the flame over time, the system will quickly distinguish normal cooking activity from an escalating emergency.

### 3. Remote Monitoring
- A remote dashboard interface enabling users to view system status, receive safety alerts in real time, and trigger a manual emergency shutoff if necessary.

---

For deeper technical details, including software Class Diagrams and Sequence Diagrams of the current logic, please see the [Architecture Documentation](docs/architecture.md).
