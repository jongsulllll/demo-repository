import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# Check for GPU availability
device = torch("cuda:0")
model = YOLO("/home/dev/runs/detect/train34/weights/epoch180.pt").to(device)
print("Camera initializing with", device, '...')

# Initialize DeepSort
tracker = DeepSort(max_age=10,
                   n_init=3,
                   nms_max_overlap=0.5,
                   embedder='mobilenet',
                   half=True,
                   embedder_gpu=(device == 'cuda'))  # Enable GPU for DeepSORT if available

# Confidence threshold
CONF_THRESHOLD = 0.76  # You can adjust this value based on your needs

# Video capture (default webcam)
cap = cv2.VideoCapture(0)  # Change to video file path if needed

# Load class names (modify if using custom dataset)
class_names = model.names

# Initialize time for FPS calculation
prev_time = time.time()
fps = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection with YOLOv8
    results = model(frame) 

    # Extract bounding boxes, confidence scores, and class IDs
    detections = results[0].boxes  # First image result (for batch processing)

    # Convert to DeepSort format and filter based on confidence
    detections_list = []
    for box in detections:
        x1, y1, x2, y2 = box.xyxy.tolist()[0]  # Bounding box
        confidence = box.conf[0].item()        # Confidence score
        class_id = int(box.cls[0].item())      # Class ID
        
        # Only add detection if confidence is above threshold
        if confidence >= CONF_THRESHOLD:
            detections_list.append(([x1, y1, x2, y2], confidence, class_id))

    # Update tracker with filtered detections
    tracks = tracker.update_tracks(detections_list, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get bounding box in [left, top, right, bottom] format
        cls = track.get_det_class()

        # Draw bounding box
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])),
                      (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        # Display Track ID
        cv2.putText(frame, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display class name if available
        if cls is not None and cls < len(class_names):
            class_name = class_names[cls]
            cv2.putText(frame, f'Class: {class_name}', (int(ltrb[0]), int(ltrb[1]-30)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Print FPS in terminal
    print(f'FPS: {fps:.2f}')

    # Show result
    end = time.time()
    print('=== ', end - start, ' ===')
    start = time.time()
    cv2.imshow('YOLOv8 + DeepSort', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

