from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from datetime import datetime

# Camera Configuration
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAMERATE = 15

# Define 8 ROIs (x1, y1, x2, y2) - adjust these coordinates
ROIs = [
    (50, 100, 150, 200),   # ROI 1
    (180, 100, 280, 200),  # ROI 2
    (310, 100, 410, 200),  # ROI 3
    (440, 100, 540, 200),  # ROI 4
    (50, 300, 150, 400),   # ROI 5
    (180, 300, 280, 400),  # ROI 6
    (310, 300, 410, 400),  # ROI 7
    (440, 300, 540, 400)   # ROI 8
]

# Model Configuration
MODEL_PATH = "sample_model/detect.tflite"
CONF_THRESHOLD = 0.3  # Minimum detection confidence
IOU_THRESHOLD = 0.2   # Intersection-over-Union threshold

def initialize_camera():
    """Set up Raspberry Pi camera module"""
    camera = PiCamera()
    camera.resolution = CAMERA_RESOLUTION
    camera.framerate = CAMERA_FRAMERATE
    raw_capture = PiRGBArray(camera, size=CAMERA_RESOLUTION)
    time.sleep(0.1)  # Camera warm-up
    return camera, raw_capture

def load_tflite_model():
    """Load and configure TFLite model"""
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Initialize components
camera, raw_capture = initialize_camera()
interpreter = load_tflite_model()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# State tracking
last_print_time = time.time()
print_interval = 1  # Seconds between updates
prev_status = [False] * len(ROIs)

def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union between two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    return inter_area / float(boxA_area + boxB_area - inter_area)

def process_frame(frame):
    """Process frame and detect objects in ROIs"""
    h, w = frame.shape[:2]
    
    # Preprocess for model
    resized = cv2.resize(frame, (300, 300))
    input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    # Get outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    status = [False] * len(ROIs)
    
    # Process detections
    for i, score in enumerate(scores):
        if score < CONF_THRESHOLD:
            continue
            
        # Convert normalized coordinates to pixels
        ymin, xmin, ymax, xmax = boxes[i]
        xmin = int(xmin * w)
        ymin = int(ymin * h)
        xmax = int(xmax * w)
        ymax = int(ymax * h)
        
        # Check ROI overlaps
        for idx, roi in enumerate(ROIs):
            if calculate_iou((xmin, ymin, xmax, ymax), roi) > IOU_THRESHOLD:
                status[idx] = True
                break
    
    # Draw ROI boxes and labels
    for idx, (x1, y1, x2, y2) in enumerate(ROIs):
        color = (0, 255, 0) if not status[idx] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, str(idx+1), (x1+5, y1+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    return frame, status

def print_status(status):
    """Print formatted status update"""
    ts = datetime.now().strftime("%H:%M:%S")
    status_str = " | ".join([f"ROI {i+1:02}: {'OCCUPIED' if s else 'FREE'}" 
                           for i, s in enumerate(status)])
    print(f"[{ts}] {status_str}")

try:
    print("Starting monitoring... (Press Q to quit)")
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array
        processed, curr_status = process_frame(image)
        
        # Display feed
        cv2.imshow('ROI Monitoring System', processed)
        
        # Update console on status change or interval
        if curr_status != prev_status or (time.time() - last_print_time) > print_interval:
            print_status(curr_status)
            prev_status = curr_status.copy()
            last_print_time = time.time()
        
        # Prepare for next frame
        raw_capture.truncate(0)
        
        # Exit on Q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    camera.close()
    cv2.destroyAllWindows()
    print("\nMonitoring stopped. Final status:")
    print_status(prev_status)