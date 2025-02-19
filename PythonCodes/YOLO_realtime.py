from ultralytics import YOLO
import torch
import cv2
import time
import mediapipe as mp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO("runs/detect/train53/weights/epoch11.pt").to(device)
#model = YOLO("yolov8n.pt").to(device)
print("camera initializing...with", device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 600)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'camera connected with {width} X {height}, {fps} FPS')  # 0 , 2 USB

fps = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("no web cam")
        break

    results = model(frame, verbose=True)  # enable print, logging??
    # results = model(frame)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if box.cls == 1:  # person class = 0
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = "person"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        elif box.cls == 2:  # gun class = 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = 'gun'
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        elif box.cls == 0:
        	cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
        	label = 'knife'
        	cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)

    
    frame_count += 1
    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("person Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

