import streamlit as st
import cv2
import time
import json
import os
import av
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="Kitchen Guardian", layout="wide", initial_sidebar_state="expanded")

from src.config import BURNER_RADIUS_PIXELS, BURNERS_FILE, VIDEO_SOURCE, SHUTOFF_ALPHA, SHUTOFF_THRESHOLD
from src.detectors import VisionSystem
from src.state_machine import SafetyGuardian
from src.temporal import FlameTracker, ShutoffDebouncer

st.title("Kitchen Guardian")
st.markdown("Real-time monitoring and safety oversight.")

def load_burners():
    if os.path.exists(BURNERS_FILE):
        try:
            with open(BURNERS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return []

# --- Thread-Safe Application State ---
class SystemState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "SAFE"
        self.flame_detected = False
        self.dangerous_fire = False
        self.person_detected = False
        self.flame_area = 0
        self.baseline_area = 0
        self.manual_shutoff = False
        self.reset_requested = False

if "sys_state" not in st.session_state:
    st.session_state.sys_state = SystemState()

sys_state = st.session_state.sys_state

# --- WebRTC Video Processor ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.vision = VisionSystem()
        self.guardian = SafetyGuardian()
        self.tracker = FlameTracker()
        self.debouncer = ShutoffDebouncer(alpha=SHUTOFF_ALPHA, threshold=SHUTOFF_THRESHOLD)
        self.burner_zones = load_burners()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # 1. Thread-safe read of user inputs
        with sys_state.lock:
            manual_shutoff = sys_state.manual_shutoff
            reset_requested = sys_state.reset_requested
            
        if reset_requested:
            self.tracker.reset()
            self.debouncer.reset()
            with sys_state.lock:
                sys_state.reset_requested = False

        # 2. ML and Detection Logic
        detection_result = self.vision.detect_objects(
            frame=img,
            burner_zones=self.burner_zones,
            mock_flame_box=None
        )
        
        heat_boxes = detection_result["flame_boxes"] + detection_result["fire_boxes"]
        growth_status = self.tracker.update(flame_boxes=heat_boxes, person_present=detection_result['person_detected'])
        
        status = self.guardian.update_status(
            flame_on=detection_result["flame_detected"],
            person_present=detection_result["person_detected"],
            growth_status=growth_status,
        )
        
        # Override rules
        if (detection_result["dangerous_fire"] or detection_result["flame_detected"]) and not detection_result['is_safe_fire']:
            status = "CRITICAL_SHUTOFF (OUTSIDE SAFE ZONE)"
            
        if manual_shutoff:
            status = "CRITICAL_SHUTOFF (MANUAL)"
            
        # 3. EMA Debouncing
        is_critical = "CRITICAL_SHUTOFF" in status
        actual_shutoff = self.debouncer.update(is_critical)
        
        if is_critical and not actual_shutoff and not manual_shutoff:
            # Downgrade to critical warning until fully triggered computationally
            status = status.replace("CRITICAL_SHUTOFF", "CRITICAL_WARNING")

        # 4. Thread-safe write to Streamlit UI State
        stats = self.tracker.get_stats()
        with sys_state.lock:
            sys_state.status = status
            sys_state.flame_detected = detection_result["flame_detected"]
            sys_state.dangerous_fire = detection_result["dangerous_fire"]
            sys_state.person_detected = detection_result["person_detected"]
            sys_state.flame_area = stats['smoothed_current']
            sys_state.baseline_area = stats['baseline_area']

        # 4. Rendering Visualization (directly on the WebRTC stream buffer)
        for item in detection_result['boxes']:
            box = item['box']
            x1, y1, x2, y2 = map(int, box)
            if item["class"] == "person":
                color = (0, 255, 0)
            elif item["class"] == "flame":
                color = (0, 140, 255)
            else:  
                color = (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{item['class']} {item['conf']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw the magenta spatial anchor point so user sees the flame origin
            if "anchor_point" in item:
                ax, ay = item["anchor_point"]
                cv2.circle(img, (ax, ay), 4, (255, 0, 255), -1)

        for zx, zy, r in self.burner_zones:
            cv2.circle(img, (zx, zy), r, (255, 150, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def get_media_player():
    from aiortc.contrib.media import MediaPlayer
    return MediaPlayer(VIDEO_SOURCE)

# --- UI Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Camera Feed")
    # Streamlit WebRTC configuration
    is_file_source = isinstance(VIDEO_SOURCE, str) and os.path.isfile(VIDEO_SOURCE)
    
    if is_file_source:
        st.info("Streaming from local server file.")
        webrtc_ctx = webrtc_streamer(
            key="demo-stream",
            mode=WebRtcMode.RECVONLY,
            video_processor_factory=VideoProcessor,
            source_video_track=get_media_player().video,
            media_stream_constraints={"video": True, "audio": False},
        )
    else:
        st.info("Streaming from browser client webcam.")
        webrtc_ctx = webrtc_streamer(
            key="webcam-stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

with col2:
    st.subheader("Controls & Metrics")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("EMERGENCY", type="primary", use_container_width=True):
            with sys_state.lock:
                sys_state.manual_shutoff = True
    with b2:
        if st.button("Reset Tracker", use_container_width=True):
            with sys_state.lock:
                sys_state.manual_shutoff = False
                sys_state.reset_requested = True

    st.divider()
    
    @st.fragment(run_every=0.5)
    def render_live_metrics():
        with sys_state.lock:
            status = sys_state.status
            f_det = sys_state.flame_detected
            d_fire = sys_state.dangerous_fire
            p_det = sys_state.person_detected
            f_area = sys_state.flame_area
            b_area = sys_state.baseline_area
            
        status_color = "green"
        if "WARNING" in status:
            status_color = "orange"
        elif "CRITICAL" in status:
            status_color = "red"
            
        st.markdown(f"### System: :{status_color}[{status}]")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Flame", "Detected" if f_det else "None")
            st.metric("Fire", "Detected" if d_fire else "None")
        with c2:
            st.metric("Person", "Present" if p_det else "Away")
            delta = f"{f_area - b_area:.0f}" if b_area > 0 else None
            st.metric("Flame Area", f"{f_area:.0f}" if b_area > 0 else "0", delta=delta)
            
    if webrtc_ctx and webrtc_ctx.state.playing:
        render_live_metrics()
    else:
        st.info("Metrics will populate when stream is started.")