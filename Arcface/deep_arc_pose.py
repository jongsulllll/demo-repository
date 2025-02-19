import cv2
import mediapipe as mp
import time
import torch
import numpy as np
import warnings
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from deep_sort_realtime.deepsort_tracker import DeepSort

cv2.ocl.setUseOpenCL(True)

# YOLO 모델 (사람=0, 총=1, 칼=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("/home/dev/runs/detect/train55/weights/epoch180.pt").to(device)
print("Camera initializing with", device, '...')
warnings.filterwarnings("ignore", category=FutureWarning)
face_model = YOLO("/home/dev/runs/detect/train34/weights/best.pt").to(device)

# ArcFace
arc_app = FaceAnalysis(name="buffalo_s")
arc_app.prepare(ctx_id=0, det_size=(320,320))
my_face_embedding = np.load("my_face.npy")

def get_face_embedding(arc_app, face_img_bgr):
    #s=time.time()
    #print(face_img_bgr.shape)
    faces = arc_app.get(face_img_bgr)
    if len(faces) == 0:
        return None
    #print("get face embedding time", time.time() - s)
    return faces[0].embedding

def is_my_face(face_embedding, my_embedding, threshold=0.4):
    #s=time.time()
    sim = cosine_similarity([face_embedding], [my_embedding])[0][0]
    #print("is my face time",  time.time() - s)
    return (sim > threshold), sim

# Pose 전용: Dangerous person만 크롭 영상에 대해 Pose
mp_pose = mp.solutions.pose
pose_danger = mp_pose.Pose(
    static_image_mode=True,  # 정적 이미지 모드로 사용
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# 팔 들기 판별용(부위 인덱스)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16

def is_arm_raised(shoulder_y, wrist_y, threshold=0.05):
    return wrist_y < (shoulder_y - threshold)

# 바운딩박스 overlap
def boxes_overlap(boxA, boxB):
    (x1A, y1A, x2A, y2A) = boxA
    (x1B, y1B, x2B, y2B) = boxB
    overlap_x = not (x2A < x1B or x2B < x1A)
    overlap_y = not (y2A < y1B or y2B < y1A)
    return overlap_x and overlap_y

############################
# 2) DeepSORT 초기화
############################
tracker = DeepSort(max_age=5,
                   n_init=1,
                   nms_max_overlap=1.0,
                   embedder='mobilenet',
                   half=True,
                   embedder_gpu=True)

############################
# 3) 메인 루프
############################
cap = cv2.VideoCapture(0)
# Set width, height, and FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 1280 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 720 pixels
cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30
# Get width, height, and FPS
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()
else:
    print(f"Width: {width}, Height: {height}, FPS: {fps}")

prev_time = time.time()

# 위험 인물(무기와 교차 & 내가 아닌) track_id
dangerous_ids = set()

#window resize하는 부분
window_name = "DeepSORT + YOLO + ArcFace + Weapons - Pose(DangerOnly)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)  #  960*720 크기로 창 설정




while True:
    t1=time.time()
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패!")
        break

    # (1) YOLO 추론
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame, verbose=False)

    person_detections = []
    weapon_boxes = []

    for box in results[0].boxes:
        conf = float(box.conf)
        if conf < 0.4:
            continue
        class_id = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = model.names[class_id]

        # 바운딩박스 시각화
        label = f"{class_name}: {conf:.2f}"
        if class_id == 0:  # person
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            w = x2 - x1
            h = y2 - y1
            person_detections.append(((x1, y1, w, h), conf, 0))
        elif class_id in [1, 2]:  # gun/knife
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            weapon_boxes.append((x1,y1,x2,y2))
    # if person_detections == []:    # pass if there is no person 

    t2=time.time() #		~ 0.6 sec
    
    # (2) DeepSORT로 사람 추적
    tracks = tracker.update_tracks(person_detections, frame=rgb_frame)
    tracked_boxes = []  # (track_id, x1,y1,x2,y2)

    for t in tracks:
        if not t.is_confirmed() or t.time_since_update > 1:
            continue
        track_id = t.track_id
        ltrb = t.to_ltrb()  # (left, top, right, bottom)
        x1t, y1t, x2t, y2t = map(int, ltrb)
        tracked_boxes.append((track_id, x1t, y1t, x2t, y2t))

        # tID 표시
        cv2.putText(frame, f"ID:{track_id}", (x1t-10, y1t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # (3) ArcFace: 모든 사람(트랙)에 대해 실행
    track_is_me = {}

    t3=time.time() ##########################################################################################
    #print("current time is ", t3)
    for (tid, px1, py1, px2, py2) in tracked_boxes:
        px1 = max(0, px1)
        s=time.time()
        face_results = face_model(frame[py1:py2, px1:px2], verbose=False)
        #print('yolo time is ', time.time() - s)   # 5~10ms
        for box in face_results[0].boxes:
            if int(box.cls) == 2:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = frame[py1+y1-50:py1+y2+50, px1+x1-50:px1+x2+50]
                cv2.rectangle(frame, (px1 + x1, py1 + y1), (px1 + x2, py1 + y2), (0, 0, 255), 2)
                #person_crop = cv2.resize(person_crop, (320,320), interpolation=cv2.INTER_AREA)
                break
        #print("first for loop", time.time() - t3)
        #t3=time.time() 
        
        
        #person_crop = frame[py1:300, px1:px2]
        #person_crop = cv2.resize(person_crop, (0, 0), fx=0.2, fy=0.2)
        if person_crop.size == 0:
            track_is_me[tid] = False
            continue	
        face_embedding = get_face_embedding(arc_app, person_crop)
        if face_embedding is not None:
            same_person, sim = is_my_face(face_embedding, my_face_embedding, threshold=0.3)
            if same_person:
                color = (0,255,0)
                text_arc = f"          Me(sim={sim:.2f})"
                track_is_me[tid] = True
                
                #same_person인 경우 dangerous_ids에서 제거
                if tid in dangerous_ids:
                    dangerous_ids.remove(tid)
            else:
                color = (0,0,255)
                text_arc = f"          NotMe(sim={sim:.2f})"
                track_is_me[tid] = False

            cv2.putText(frame, text_arc, (px1, py1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            track_is_me[tid] = False

    t4=time.time() #####################################################################################

    # (4) 무기 교차 & 내가 아닌 => dangerous_ids
    for (tid, px1, py1, px2, py2) in tracked_boxes:
        if track_is_me.get(tid) == False:  # 내가 아닌 경우
            person_box = (px1, py1, px2, py2)
            for wb in weapon_boxes:
                if boxes_overlap(person_box, wb):
                    dangerous_ids.add(tid)
                    break

    t5=time.time()

    # (5) Dangerous person => Pose 처리 (크롭 이미지)
    #     - Pose만 "위험 인물"에 대해 수행
    for (tid, px1, py1, px2, py2) in tracked_boxes:
        if tid in dangerous_ids:
            # Pose on subimage
            person_crop = frame[py1:py2, px1:px2]
            if person_crop.size == 0:
                continue

            # BGR->RGB
            crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_result = pose_danger.process(crop_rgb)

            if pose_result.pose_landmarks:
                # Mediapipe Pose는 랜드마크가 [0..1] 범위
                # 각 랜드마크를 원본 frame 좌표로 변환
                landmarks = pose_result.pose_landmarks.landmark

                # 간단히 팔 들기 판별
                # (상대 좌표(0..1)를 사용하므로 y값만 떼어다 계산 가능하나,
                #  실제론 bounding box가 부분만 포함해서 정확도가 떨어질 수 있음.)
                left_shoulder_y = landmarks[11].y
                right_shoulder_y = landmarks[12].y
                left_wrist_y = landmarks[15].y
                right_wrist_y = landmarks[16].y

                left_arm_up = is_arm_raised(left_shoulder_y, left_wrist_y)
                right_arm_up = is_arm_raised(right_shoulder_y, right_wrist_y)

                # 시각화: 랜드마크를 원본 frame에 매핑
                sub_w = (px2 - px1)
                sub_h = (py2 - py1)

                action_text = ""
                if left_arm_up and right_arm_up:
                    action_text = "both arms up"
                elif left_arm_up:
                    action_text = "left arm up"
                elif right_arm_up:
                    action_text = "right arm up"
                else:
                    action_text = "do nothing"

                # subimage 랜드마크 → 원본 frame 좌표
                for lm in pose_result.pose_landmarks.landmark:
                    cx = px1 + int(lm.x * sub_w)
                    cy = py1 + int(lm.y * sub_h)
                    cv2.circle(frame, (cx, cy), 3, (0,255,255), -1)

                # "Dangerous person" 표시
                cv2.putText(frame, f"Dangerous person: {action_text}",
                            (px1, py1+20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,0,255), 2)

    

    # FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    #print(f'estimated time is {current_time - prev_time:.2f}ms')

    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    t6=time.time()
    print(f'{t2-t1:.6f}\t{t3-t2:.6f}\t{t4-t3:.6f}\t{t5-t4:.6f}\t{t6-t5:.6f}')
    
cap.release()
cv2.destroyAllWindows()
