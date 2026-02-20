# System Architecture

## Class Diagram

This diagram illustrates the structure of the *Kitchen Guardian* system, showing the relationships between the main execution loop, the vision system, and the safety state machine.

```mermaid
classDiagram
    class Main {
        +main()
        -setup_camera()
    }
    
    class Config {
        +TIMEOUT_SECONDS: int
        +WARNING_SECONDS: int
        +CRITICAL_SECONDS: int
        +CAMERA_INDEX: int
        +CONFIDENCE_THRESHOLD: float
    }

    class VisionSystem {
        -model: YOLO
        -classes: List[str]
        +__init__(model_path: str)
        +detect_objects(frame: ndarray) Dict
    }

    class SafetyGuardian {
        -last_person_seen_time: float
        -state: str
        +__init__()
        +update_status(flame_on: bool, person_present: bool) str
    }

    Main ..> VisionSystem : uses
    Main ..> SafetyGuardian : uses
    Main ..> Config : uses
    VisionSystem ..> Config : uses
    SafetyGuardian ..> Config : uses
```

## Sequence Diagram

This diagram shows the runtime flow of a single frame processing cycle within the application loop.

```mermaid
sequenceDiagram
    participant C as Camera
    participant M as Main Loop
    participant V as VisionSystem
    participant S as SafetyGuardian
    participant U as UI/Display

    loop Every Frame
        M->>C: read()
        C-->>M: frame
        
        M->>V: detect_objects(frame)
        activate V
        V->>V: YOLO inference
        V-->>M: {person: bool, flame: bool, boxes: []}
        deactivate V

        M->>S: update_status(flame_on, person_present)
        activate S
        alt is person present
            S->>S: reset timer
            S-->>M: "SAFE"
        else is flame on
            S->>S: check time elapsed
            S-->>M: "SAFE" / "WARNING" / "CRITICAL"
        else flame off
            S-->>M: "SAFE"
        end
        deactivate S

        M->>U: draw_boxes(boxes)
        M->>U: draw_status(state)
        M->>U: show(frame)
    end
```