import tensorflow as tf
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from datetime import datetime

# Define 8 ROIs in (x1, y1, x2, y2) format - adjust coordinates to your needs
ROIs = [
   
    (47, 30, 138, 51),   # ROI 1
    (40, 54, 145, 81),   # ROI 2
    (34, 86, 152, 125),  # ROI 3
    (24, 130, 162, 181), # ROI 4
    (139, 30, 234, 51),  # ROI 5
    (144, 56, 250, 81),  # ROI 6
    (156, 87, 283, 125), # ROI 7
    (165, 134, 295, 170) # ROI 8

]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="sample_model/detect.tflite")
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 10
raw_capture = PiRGBArray(camera, size=camera.resolution)
time.sleep(0.1)

# State tracking
last_print_time = time.time()
print_interval = 1  # Seconds between status updates
prev_status = [False] * len(ROIs)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)

def process_frame(frame, conf=0.3, iou=0.1):
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (300, 300))
    input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    status = [False]*len(ROIs)
    
    for i, score in enumerate(scores):
        if score < conf:
            continue
            
        ymin, xmin, ymax, xmax = boxes[i]
        xmin = int(xmin * w)
        ymin = int(ymin * h)
        xmax = int(xmax * w)
        ymax = int(ymax * h)
        
        for idx, roi in enumerate(ROIs):
            if calculate_iou((xmin, ymin, xmax, ymax), roi) > iou:
                status[idx] = True
                break
    
    # Draw ROIs
    for idx, (x1, y1, x2, y2) in enumerate(ROIs):
        color = (0, 255, 0) if not status[idx] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    return frame, status

def print_status(status):
    ts = datetime.now().strftime("%H:%M:%S")
    status_str = " | ".join([f"ROI {i+1:02}: {int(s)}" for i, s in enumerate(status)])
    print(f"[{ts}] {status_str}")

try:
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array
        processed, curr_status = process_frame(image)
        
        cv2.imshow('ROI Monitor', processed)
        
        # Update terminal if status changes or interval passed
        if curr_status != prev_status or (time.time() - last_print_time) > print_interval:
            print_status(curr_status)
            prev_status = curr_status.copy()
            last_print_time = time.time()
        
        raw_capture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.close()
    cv2.destroyAllWindows()
    print("\nMonitoring stopped. Final status:")
    print_status(prev_status)
