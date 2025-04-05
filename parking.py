# sudo apt-get install libatlas-base-dev libopenjp2-7 libtiff5
# pip3 install tensorflow tensorflow-lite opencv-python-headless picamera
# 
# !wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

# !unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d sample_model
import tensorflow as tf
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from datetime import datetime

# Raspberry Pi Camera Configuration
CAMERA_RESOLUTION = (640, 480)  # Max resolution for Pi Camera V2
CAMERA_FRAMERATE = 15           # Adjust based on Pi model

# Define 8 ROIs (x1, y1, x2, y2) - adjust these values for your setup
ROIs = [
    # Top row
    (50, 100, 150, 200),   # ROI 1
    (180, 100, 280, 200),  # ROI 2
    (310, 100, 410, 200),  # ROI 3
    (440, 100, 540, 200),  # ROI 4
    # Bottom row
    (50, 300, 150, 400),   # ROI 5
    (180, 300, 280, 400),  # ROI 6
    (310, 300, 410, 400),  # ROI 7
    (440, 300, 540, 400)   # ROI 8
]

# Model Configuration
MODEL_PATH = "sample_model/detect.tflite"
CONFIDENCE_THRESHOLD = 0.3  # Minimum detection confidence
IOU_THRESHOLD = 0.2         # Intersection-over-Union threshold

def initialize_camera():
    """Initialize Raspberry Pi camera module"""
    camera = PiCamera()
    camera.resolution = CAMERA_RESOLUTION
    camera.framerate = CAMERA_FRAMERATE
    raw_capture = PiRGBArray(camera, size=CAMERA_RESOLUTION)
    time.sleep(0.1)  # Camera warm-up
    return camera, raw_capture

def load_model():
    """Load TFLite model and allocate tensors"""
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Initialize components
camera, raw_capture = initialize_camera()
interpreter = load_model()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# State tracking
last_print_time = time.time()
print_interval = 1  # Seconds between status updates
prev_status = [False] * len(ROIs)

def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two boxes"""
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
    
    # Preprocess frame for model
    resized = cv2.resize(frame, (300, 300))
    input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    # Get results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    status = [False] * len(ROIs)
    
    # Process detections
    for i, score in enumerate(scores):
        if score < CONFIDENCE_THRESHOLD:
            continue
            
        # Convert normalized coordinates to pixel values
        ymin, xmin, ymax, xmax = boxes[i]
        xmin = int(xmin * w)
        ymin = int(ymin * h)
        xmax = int(xmax * w)
        ymax = int(ymax * h)
        
        # Check against all ROIs
        for idx, roi in enumerate(ROIs):
            if calculate_iou((xmin, ymin, xmax, ymax), roi) > IOU_THRESHOLD:
                status[idx] = True
                break  # Each detection can only trigger one ROI
    
    # Visual feedback
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
        
        # Display monitoring feed
        cv2.imshow('Pi ROI Monitor', processed)
        
        # Update console only on status change or time interval
        if curr_status != prev_status or (time.time() - last_print_time) > print_interval:
            print_status(curr_status)
            prev_status = curr_status.copy()
            last_print_time = time.time()
        
        # Clear stream for next frame
        raw_capture.truncate(0)
        
        # Exit on Q key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup resources
    camera.close()
    cv2.destroyAllWindows()
    print("\nMonitoring stopped. Final status:")
    print_status(prev_status)