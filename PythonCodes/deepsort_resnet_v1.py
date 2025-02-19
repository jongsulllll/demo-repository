import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 Model
model = YOLO("/home/dev/runs/detect/train34/weights/epoch180.pt").to(device)
print("YOLO model loaded on", device)

# Initialize DeepSort with GPU support
tracker = DeepSort(max_age=10,
                   n_init=3,
                   nms_max_overlap=0.5,
                   embedder='resnet',  # ResNet으로 변경하여 GPU 최적화
                   half=True,
                   embedder_gpu=True)  # 강제 GPU 사용

# 강제로 DeepSORT의 모델을 GPU로 이동
if device == 'cuda':
    tracker.model.to(torch.device('cuda'))
    tracker.model.half()

# Confidence threshold
CONF_THRESHOLD = 0.76

# Video capture
cap = cv2.VideoCapture(2)

# Load class names
class_names = model.names

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection with YOLOv8
    results = model(frame)  # GPU에서 YOLO 실행

    # Extract bounding boxes, confidence scores, and class IDs
    detections = results[0].boxes  

    # Convert to DeepSort format and filter based on confidence
    detections_list = []
    for box in detections:
        x1, y1, x2, y2 = box.xyxy.tolist()[0]  
        confidence = box.conf[0].item()        
        class_id = int(box.cls[0].item())      

        if confidence >= CONF_THRESHOLD:
            detections_list.append(([x1, y1, x2, y2], confidence, class_id))

    # Update tracker
    tracks = tracker.update_tracks(detections_list, frame=frame)  

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  
        cls = track.get_det_class()

        # ✅ frame을 GPU 텐서가 아니라 OpenCV 이미지 그대로 사용
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])),
                      (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if cls is not None and cls < len(class_names):
            class_name = class_names[cls]
            cv2.putText(frame, f'Class: {class_name}', (int(ltrb[0]), int(ltrb[1]-30)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # ✅ frame을 GPU 텐서가 아니라 OpenCV 이미지 그대로 사용
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    print(f'FPS: {fps:.2f}')
    cv2.imshow('YOLOv8 + DeepSort', frame)  # ✅ frame을 그대로 사용

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
