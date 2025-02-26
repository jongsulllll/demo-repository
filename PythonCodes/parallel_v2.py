# parallel_v1: arcface + yolo + realsense
# 1. face 임베딩 생성 (멀티프로세싱 - 각 프로세스가 개별적으로 모델 초기화,병렬 임베딩 추출)
# 2. realsense + yolo + mobilefacenet 병렬 처리 
# realsense 프레임 캡처 - 멀티스레딩
# yolo human detection - 멀티프로세싱
# 얼굴 임베딩 비교 (멀티스레딩)
# 전체 실행 - yolo는 멀티프로세싱, realsense와 face는 멀티스레딩

# v2: mediapipe 추가, 병렬화 o

# 1. 사람 + 총 + 칼 인식   - yolo 
사람인 경우 yolo segmentation까지 
# 2. Arcface를 모든 사람에 대해 적용(Me, NotME)  - yolo face model 사용해서 얼굴 부분에서만 buffalo sc 사용해서 얼굴 분석
# 3. NotMe이면서 사람과 총 혹은 칼의 bounding box가 겹치는 경우에는 위험인으로 분류 -> 한 번 위험인으로 분류된 사람의 ID는 dangerous_ids로 관리됨 -> dangerous_ids 중 하나에 해당하는 사람은 위에 "Dangerous person"이라고 뜸
# 4. pose estimation을 dangerous person에 대해 적용(왼팔들기, 오른팔들기, 양팔들기)  


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

#####################여기부터 수정##############################
import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from concurrent.futures import ProcessPoolExecutor
import mediapipe as mp
import time
import torch
import warnings
import threading
import multiprocessing as mp_process
from queue import Queue
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import pyrealsense2 as rs

############################
# (A) Mediapipe Pose 처리 (멀티스레딩)
############################
class MediapipePoseProcessor(threading.Thread):
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False,
                                               model_complexity=1,
                                               enable_segmentation=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
        self.running = True
    
    def run(self):
        while self.running:
            data = self.input_queue.get()
            if data is None:
                break
            tid, person_crop = data
            rgb_frame = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            result = self.mp_pose.process(rgb_frame)
            self.output_queue.put((tid, result))
    
    def stop(self):
        self.running = False

############################
# (B) RealSense 프레임 캡처 (멀티스레딩)
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
# (C) YOLO를 멀티프로세싱으로 실행 (사람 + 총 + 칼 탐지)
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
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id == 0:  # 사람 탐지
                w, h = x2 - x1, y2 - y1
                person_detections.append(((x1, y1, w, h), conf, 0))
            elif class_id in [1, 2]:  # 총 또는 칼 탐지
                weapon_boxes.append((x1, y1, x2, y2))
        
        output_queue.put((person_detections, weapon_boxes))

############################
# (D) 위험인 분류 및 ArcFace 처리
############################
dangerous_ids = set()

############################
# (E) 메인 실행
############################
if __name__ == "__main__":
    frame_queue = Queue()
    detection_input_queue = mp_process.Queue()
    detection_output_queue = mp_process.Queue()
    face_input_queue = Queue()
    face_output_queue = Queue()
    pose_input_queue = Queue()
    pose_output_queue = Queue()
    
    # RealSense 프레임 캡처 시작
    realsense_capture = RealSenseCapture(frame_queue)
    realsense_capture.start()
    
    # YOLO 프로세스 시작 (사람 + 총 + 칼 탐지)
    detection_process = mp_process.Process(target=process_detection, args=(detection_input_queue, detection_output_queue, "C:/epoch180.pt"))
    detection_process.start()
    
    # Mediapipe Pose 스레드 시작
    pose_processor = MediapipePoseProcessor(pose_input_queue, pose_output_queue)
    pose_processor.start()
    
    # Dangerous Person 분류 및 FPS 출력 등 관리
    cv2.destroyAllWindows()
