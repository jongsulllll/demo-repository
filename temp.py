#!/usr/bin/env python
import cv2
import mediapipe as mp
import time
import torch
import numpy as np
import warnings

from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

############################
# (A) MobileFaceNet (InsightFace) 준비
############################
import insightface
from insightface.app import FaceAnalysis

# === 변경점 1) DeepSORT → OC-SORT ===
# from deep_sort_realtime.deepsort_tracker import DeepSort  # 제거
from ocsort.ocsort import OCSort  # 추가

app = FaceAnalysis(name="buffalo_sc")  # MobileFaceNet(or ArcFace) 모델
app.prepare(ctx_id=0, det_size=(640,640))  # GPU: ctx_id=0, CPU: -1

def get_mobileface_embedding(image_bgr):
    if image_bgr is None or image_bgr.size==0:
        return None
    faces = app.get(image_bgr)
    if len(faces)==0:
        return None
    return faces[0].embedding  # shape 예: (128,)

def is_my_face(face_embedding, my_embedding, threshold=0.4):
    sim = cosine_similarity([face_embedding], [my_embedding])[0][0]
    return (sim > threshold), sim

############################
# (B) 모델/함수 초기화
############################
warnings.filterwarnings("ignore", category=FutureWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO("/home/dev/2*/codes/epoch180.pt").to(device)
model_seg = YOLO("yolov8n-seg.pt").to(device)

# MobileFaceNet 임베딩
my_face_embedding = np.load("/home/dev/2*/codes/arcface/myface.npy")  # (128,) or (512, etc.)

############################
# (C) Mediapipe Pose
############################
mp_pose = mp.solutions.pose
pose_danger = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_drawing = mp.solutions.drawing_utils
LEFT_SHOULDER = 11
RIGHT_SHOULDER= 12
LEFT_WRIST   = 15
RIGHT_WRIST  = 16

def is_arm_raised(shoulder_y, wrist_y, threshold=0.05):
    return wrist_y < (shoulder_y - threshold)

def boxes_overlap(boxA, boxB):
    (x1A, y1A, x2A, y2A) = boxA
    (x1B, y1B, x2B, y2B) = boxB
    overlap_x = not (x2A < x1B or x2B < x1A)
    overlap_y = not (y2A < y1B or y2B < y1A)
    return overlap_x and overlap_y

############################
# (D) OC-SORT
############################
# === 변경점 2) OC-SORT 초기화 ===
tracker = OCSort(
    det_thresh=0.5,
    iou_threshold=0.3,
    use_byte=False,
    max_age=30,
    min_hits=3
)

############################
# (E) 메인 루프
############################
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

dangerous_ids = set()
track_me_status = {}
track_arcface_count= {}
MAX_ARCFACE_FRAMES= 20
sim = 0

window_name = "OC-SORT + YOLO(SEG) + MobileFaceNet + Pose"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960,720)

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패!")
        break

    # (1) YOLO 추론
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame, conf=0.5)
    results_seg = model_seg(rgb_frame, conf=0.5)

    det = results[0]
    boxes2 = det.boxes
    masks2 = results_seg[0].masks

    weapon_boxes= []

    # === OC-SORT에 넘길 detection 형식: Nx5 = [x1,y1,x2,y2,score]
    # (DeepSORT 때는 person_detections=((x,y,w,h), conf, class_id) 였지만 여기선 Nx5 필요) 
    detection_list = []

    if boxes2 is not None:
        for i, box in enumerate(boxes2):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf_val = float(box.conf[0])

            # 시각화
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)
            label = f"{model.names[class_id]}: {conf_val:.2f}"
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            # 세그 윤곽선(사람만)
            if masks2 is not None and class_id==0:
                if i < len(masks2.data):
                    single_mask= masks2.data[i].cpu().numpy()
                    mask_bin= (single_mask>0.5).astype(np.uint8)
                    contours,_= cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, (0,255,255), 2)

            # 무기(1,2) -> weapon_boxes
            if class_id in [1,2]:
                weapon_boxes.append((x1,y1,x2,y2))

            # === OC-SORT에선 객체 클래스 상관 없이 Nx5 => [x1,y1,x2,y2,score]
            # 사람만 추적하려면 if class_id==0: 만 추가
            #detection_list.append([x1, y1, x2, y2, conf_val])
            detection_list.append([x1, y1, x2, y2, conf_val, float(class_id)])

    # (2) OC-SORT update
    time1 = time.time()
    if len(detection_list)>0:
        det_array = np.array(detection_list, dtype=np.float32)
        det_tensor = torch.from_numpy(det_array)  # device='cpu'
    else:
        det_tensor = torch.empty((0, 6), dtype=torch.float32)

    # oc-sort update -> Nx6 = [x1,y1,x2,y2,track_id,score]
    tracks = tracker.update(det_tensor, frame.shape)  # image shape param

    # re-format
    tracked_boxes=[]
    for t_ in tracks:
        # t_ 길이가 6이상이라고 가정 => x1,y1,x2,y2,track_id,score
        x1_, y1_, x2_, y2_, track_id_, score_ = t_[:6]
        x1_, y1_, x2_, y2_, track_id_ = int(x1_), int(y1_), int(x2_), int(y2_), int(track_id_)
        tracked_boxes.append((track_id_, x1_, y1_, x2_, y2_))

        cv2.putText(frame,f"ID:{track_id_}", (x1_-10,y1_-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    time2 = time.time()
    print("OCsort running time:", time2-time1)
    # (3) MobileFaceNet(ArcFace) 로직

    for (tid, px1, py1, px2, py2) in tracked_boxes:
        if tid not in track_me_status:
            track_me_status[tid]= False
        if tid not in track_arcface_count:
            track_arcface_count[tid]=0

        # Arcface 실행 (여기서는 매 프레임)
        if track_me_status[tid]==False and track_arcface_count[tid]<MAX_ARCFACE_FRAMES:
            track_arcface_count[tid]+=1
            print("Running ArcFace for ID:", tid)
            track_arcface_count[tid]+=1

            # 얼굴 크롭
            PAD=10
            sub_face= frame[max(0,py1-PAD): py2+PAD, max(0,px1-PAD): px2+PAD]
            if sub_face.size==0:
                continue

            emb_image = get_mobileface_embedding(sub_face)
            if emb_image is not None:
                same_person, sim = is_my_face(emb_image, my_face_embedding, threshold=0.4)
                if same_person:
                    track_me_status[tid]=True
                    # 위험 id 해제
                    

        # 시각화
        if track_me_status[tid]:
            text_arc= f"          Me(sim={sim:.2f})"
            color=(0,255,0)
            if tid in dangerous_ids:
                dangerous_ids.remove(tid)
        else:
            text_arc= "           NotMe"
            color=(0,0,255)
        cv2.putText(frame,text_arc,(px1, py1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)


    # (4) 무기 교차 & NotMe => dangerous
    for (tid, px1, py1, px2, py2) in tracked_boxes:
        if not track_me_status[tid]:
            pbox= (px1, py1, px2, py2)
            for wb in weapon_boxes:
                if boxes_overlap(pbox, wb):
                    dangerous_ids.add(tid)
                    break

    # (5) Dangerous => Mediapipe Pose
    for (tid, px1, py1, px2, py2) in tracked_boxes:
        if tid in dangerous_ids:
            sub= frame[py1:py2, px1:px2]
            if sub.size==0:
                continue
            c_rgb= cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
            pose_result= pose_danger.process(c_rgb)
            if pose_result.pose_landmarks:
                lms= pose_result.pose_landmarks.landmark
                sub_w= px2 - px1
                sub_h= py2 - py1

                left_shoulder_y= lms[LEFT_SHOULDER].y
                right_shoulder_y= lms[RIGHT_SHOULDER].y
                left_wrist_y= lms[LEFT_WRIST].y
                right_wrist_y= lms[RIGHT_WRIST].y

                la_up= (left_wrist_y< (left_shoulder_y-0.05))
                ra_up= (right_wrist_y<(right_shoulder_y-0.05))
                if la_up and ra_up:
                    a_text= "both arms up"
                elif la_up:
                    a_text= "left arm up"
                elif ra_up:
                    a_text= "right arm up"
                else:
                    a_text= "do nothing"

                for lm in lms:
                    cx= px1+int(lm.x*sub_w)
                    cy= py1+int(lm.y*sub_h)
                    cv2.circle(frame,(cx,cy),3,(0,255,255),-1)
                cv2.putText(frame,f"Dangerous person: {a_text}",
                            (px1,py1+20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    # FPS
    now= time.time()
    fps= 1.0/(now - prev_time)
    prev_time= now
    cv2.putText(frame,f"FPS:{fps:.2f}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
