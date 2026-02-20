# Computer Vision Pipeline for the IoT Project: Kitchen Guardian

## High-Level Architecture (Hardware + Software)

The following diagram illustrates the **end to end architecture** of *Kitchen Guardian*, showing how hardware components interface with the computer vision pipeline and the safety decision logic running on the edge device.


```mermaid
graph LR
  subgraph HW [Hardware Layer]
    Camera["Camera<br/>USB or Pi Camera"]
    MQ6["MQ-6 Gas Sensor<br/>(LPG / Butane)"]
    ADC["ADC<br/>(MCP3008 / ADS1115)"]
    Servo["Servo Motor<br/>(MG996R or similar)"]
    PSU["Power Supply<br/>(5–6V for servo)"]
  end
  
  subgraph EDGE [Edge Device]
    Pi["Raspberry Pi / Android"]
    Vision["VisionSystem<br/>(Fire & Person Detection)"]
    Safety["SafetyGuardian<br/>(Decision Logic)"]
  end

  Mobile["Mobile App / Cloud"]

  Camera -->|Video Frames| Vision
  Vision --> Safety
  MQ6 -->|Analog Signal| ADC
  ADC -->|I²C / SPI| Pi
  Safety -->|KILL / STATUS| Pi
  Pi -->|PWM / I²C PWM| Servo
  PSU --> Servo
  PSU --> Pi
  Pi -->|Alerts / Telemetry| Mobile