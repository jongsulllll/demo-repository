import cv2
import numpy as np
import threading
import multiprocessing as mp
from queue import Queue
from sklearn.metrics.pairwise import cosine_similarity
import torch
from ultralytics import YOLO
import pyrealsense2 as rs
import insightface
from insightface.app import FaceAnalysis
import mediapipe as mp_pose
import time

############################
# (A) 위험인 판별 기준
############################
dangerous_ids = set()
face_threshold = 0.5  # 얼굴 임베딩 유사도 기준

############################
# (B) FaceAnalysis 모델 초기화 (멀티프로세싱 용)
############################
def init_face_model():
    global face_app
    face_app = FaceAnalysis(name="buffalo_sc")
    face_app.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(face_image):
    faces = face_app.get(face_image)
    if faces:
        return faces[0].embedding
    return None

############################
# (C) YOLO로 사람, 총, 칼 탐지 (멀티프로세싱)
############################
def process_detection(input_queue, output_queue, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path).to(device)
    
    while True:
        frame = input_queue.get()
        if frame is None:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame, conf=0.5)
        
        det = results[0]
        boxes = det.boxes
        
        person_detections = []
        weapon_boxes = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id == 0:  # 사람
                w, h = x2 - x1, y2 - y1
                person_detections.append(((x1, y1, w, h), conf, i))
            elif class_id in [1, 2]:  # 총 또는 칼
                weapon_boxes.append((x1, y1, x2, y2))
        
        output_queue.put((person_detections, weapon_boxes))

############################
# (D) 위험인 분류 로직
############################
def classify_dangerous_person(person_detections, weapon_boxes, face_embeddings, my_embedding, frame):
    global dangerous_ids
    for (bbox, conf, tid), embedding in zip(person_detections, face_embeddings):
        x, y, w, h = bbox
        person_id = hash(tuple(bbox))
        
        # 1. ArcFace 임베딩 비교 (Me vs NotMe)
        similarity = cosine_similarity([embedding], [my_embedding])[0][0]
        is_me = similarity >= face_threshold
        
        # 2. 총/칼과 겹치는지 체크
        overlap = any(x < wx2 and x + w > wx1 and y < wy2 and y + h > wy1 for wx1, wy1, wx2, wy2 in weapon_boxes)
        
        # 3. NotMe이면서 무기와 겹치면 위험인으로 등록
        if not is_me and overlap:
            dangerous_ids.add(person_id)
        
        # 4. Bounding Box 및 Tracking ID 표시
        label = f"ID: {tid}" if person_id not in dangerous_ids else f"Dangerous Person ID: {tid}"
        color = (0, 0, 255) if person_id in dangerous_ids else (0, 255, 0)
        
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

############################
# (E) FPS 계산
############################
def draw_fps(frame, start_time):
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

############################
# (F) 메인 실행
############################
if __name__ == "__main__":
    # 내 얼굴 임베딩 불러오기
    my_embedding = np.load("my_mobileface_embedding.npy")
    
    # 큐 설정
    frame_queue = Queue()
    detection_input_queue = mp.Queue()
    detection_output_queue = mp.Queue()
    face_input_queue = Queue()
    face_output_queue = Queue()
    
    # YOLO 프로세스 시작
    detection_process = mp.Process(target=process_detection, args=(detection_input_queue, detection_output_queue, "C:/epoch180.pt"))
    detection_process.start()
    
    # Face 모델 초기화
    init_face_model()
    
    # RealSense 설정
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    while True:
        start_time = time.time()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        
        # YOLO 탐지 실행
        detection_input_queue.put(frame)
        person_detections, weapon_boxes = detection_output_queue.get()
        
        # ArcFace 얼굴 분석
        face_embeddings = [get_face_embedding(frame[y:y+h, x:x+w]) for (x, y, w, h), _, _ in person_detections]
        face_embeddings = [emb for emb in face_embeddings if emb is not None]
        
        # 위험인 분류 및 표시
        classify_dangerous_person(person_detections, weapon_boxes, face_embeddings, my_embedding, frame)
        
        # FPS 출력
        draw_fps(frame, start_time)
        
        # 화면 표시
        cv2.imshow("RealSense Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 종료 처리
    pipeline.stop()
    detection_process.terminate()
    cv2.destroyAllWindows()
