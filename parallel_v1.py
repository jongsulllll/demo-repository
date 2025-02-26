# parallel_v1: arcface + yolo + mediapipe + realsense
# 1. face 임베딩 생성 (멀티프로세싱 - 각 프로세스가 개별적으로 모델 초기화,병렬 임베딩 추출)
# 2. realsense + yolo + mobilefacenet 병렬 처리 
# realsense 프레임 캡처 - 멀티스레딩
# yolo human detection - 멀티프로세싱
# 얼굴 임베딩 비교 (멀티스레딩)
# 전체 실행 - yolo는 멀티프로세싱, realsense와 face는 멀티스레딩


import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from concurrent.futures import ProcessPoolExecutor

def init_model():
    """각 프로세스에서 개별적으로 FaceAnalysis 모델을 초기화"""
    global app
    app = FaceAnalysis(name="buffalo_sc")  # MobileFaceNet 계열
    app.prepare(ctx_id=0, det_size=(640, 640))  # 🔹 GPU 모드 사용

def get_mobileface_embedding(image_path):
    """이미지를 로드하여 MobileFaceNet 임베딩을 추출"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    faces = app.get(img_bgr)
    if len(faces) == 0:
        return None
    return faces[0].embedding

def process_image(file, folder):
    """각 프로세스에서 실행될 함수 (my_face_folder를 인자로 받음)"""
    init_model()  # 🔹 각 프로세스에서 MobileFaceNet 모델을 개별적으로 초기화
    path = os.path.join(folder, file)
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        return get_mobileface_embedding(path)
    return None

if __name__ == "__main__":
    my_face_folder = "C:/myface"
    image_files = [file for file in os.listdir(my_face_folder) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    embeddings = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_image, image_files, [my_face_folder] * len(image_files))  # 🔹 인자로 전달

        for emb in results:
            if emb is not None:
                embeddings.append(emb)

    if len(embeddings) == 0:
        raise ValueError("No embeddings generated from MobileFaceNet.")

    avg_embedding = np.mean(embeddings, axis=0)
    print("MobileFaceNet embedding shape:", avg_embedding.shape)
    
    np.save("my_mobileface_embedding.npy", avg_embedding)
    print("Saved my_mobileface_embedding.npy")


import cv2
import mediapipe as mp
import time
import torch
import numpy as np
import warnings
import threading
import multiprocessing as mp_process
from queue import Queue

from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import pyrealsense2 as rs
import insightface
from insightface.app import FaceAnalysis

############################
# (A) RealSense 프레임 캡처를 별도의 스레드에서 실행
############################
class RealSenseCapture(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.running = True

    def run(self):
        while self.running:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame and depth_frame:
                self.queue.put((color_frame, depth_frame))

    def stop(self):
        self.running = False
        self.pipeline.stop()

############################
# (B) YOLO를 멀티프로세싱으로 실행
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
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id == 0:  # 사람만 탐지
                w, h = x2 - x1, y2 - y1
                person_detections.append(((x1, y1, w, h), conf, 0))
        
        output_queue.put(person_detections)

############################
# (C) MobileFaceNet 병렬화
############################
class FaceEmbeddingProcessor(threading.Thread):
    def __init__(self, input_queue, output_queue, my_face_embedding):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.my_face_embedding = my_face_embedding
        self.app = FaceAnalysis(name="buffalo_sc")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.running = True

    def run(self):
        while self.running:
            data = self.input_queue.get()
            if data is None:
                break
            tid, face_crop = data
            embedding = self.get_mobileface_embedding(face_crop)
            if embedding is not None:
                same_person, sim = self.is_my_face(embedding, self.my_face_embedding, threshold=0.4)
                self.output_queue.put((tid, same_person, sim))

    def get_mobileface_embedding(self, image_bgr):
        faces = self.app.get(image_bgr)
        return faces[0].embedding if faces else None

    def is_my_face(self, face_embedding, my_embedding, threshold=0.4):
        sim = cosine_similarity([face_embedding], [my_embedding])[0][0]
        return sim > threshold, sim

    def stop(self):
        self.running = False

############################
# (D) 메인 실행
############################
if __name__ == "__main__":
    frame_queue = Queue()
    detection_input_queue = mp_process.Queue()
    detection_output_queue = mp_process.Queue()
    face_input_queue = Queue()
    face_output_queue = Queue()
    
    # RealSense 프레임 캡처 시작
    realsense_capture = RealSenseCapture(frame_queue)
    realsense_capture.start()
    
    # YOLO 프로세스 시작
    detection_process = mp_process.Process(target=process_detection, args=(detection_input_queue, detection_output_queue, "C:/epoch180.pt"))
    detection_process.start()
    
    # MobileFaceNet 스레드 시작
    my_face_embedding = np.load("my_mobileface_embedding.npy")
    face_processor = FaceEmbeddingProcessor(face_input_queue, face_output_queue, my_face_embedding)
    face_processor.start()
    
    # OpenCV 윈도우 설정
    window_name = "Parallel Processing"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    
    while True:
        if not frame_queue.empty():
            color_frame, depth_frame = frame_queue.get()
            frame = np.asanyarray(color_frame.get_data())
            detection_input_queue.put(frame)
            
            if not detection_output_queue.empty():
                person_detections = detection_output_queue.get()
                for detection in person_detections:
                    x1, y1, w, h = detection[0]
                    face_crop = frame[y1:y1+h, x1:x1+w]
                    if face_crop.size > 0:
                        face_input_queue.put((x1, face_crop))

            while not face_output_queue.empty():
                tid, same_person, sim = face_output_queue.get()
                color = (0, 255, 0) if same_person else (0, 0, 255)
                cv2.putText(frame, f"ID {tid}: {sim:.2f}", (10, 30 + tid * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    realsense_capture.stop()
    detection_input_queue.put(None)
    detection_process.join()
    face_processor.stop()
    cv2.destroyAllWindows()
