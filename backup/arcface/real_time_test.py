import cv2
import mediapipe as mp
import time
import torch
import numpy as np
import warnings

from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from deep_sort_realtime.deepsort_tracker import DeepSort

import pyrealsense2 as rs


############################
# (A) Initialize RealSense Pipeline
############################
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

############################
# (B) Depth Processing Function
############################
def get_center_window_distance(depth_frame, x1, y1, x2, y2, window_size=30):
    """Get the average distance of a window at the center of the bounding box."""
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Define window
    half_window = window_size // 2
    window_x1 = max(0, cx - half_window)
    window_y1 = max(0, cy - half_window)
    window_x2 = min(depth_frame.get_width(), cx + half_window)
    window_y2 = min(depth_frame.get_height(), cy + half_window)

    distances = []
    for x in range(window_x1, window_x2):
        for y in range(window_y1, window_y2):
            distance = depth_frame.get_distance(x, y)
            if distance > 0:
                distances.append(distance)

    return np.mean(distances) if distances else 0  # Return 0 if no valid distances


############################
# (C) MobileFaceNet (InsightFace) 준비
############################
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_sc")  # "buffalo_m"은 MobileFaceNet 기반
app.prepare(ctx_id=0, det_size=(640,640))  # GPU라면 ctx_id=0, CPU는 -1

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
# (D) 모델/함수 초기화
############################
warnings.filterwarnings("ignore", category=FutureWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO("C:/epoch180.pt").to(device)
model_seg = YOLO("yolov8n-seg.pt").to(device)

# MobileFaceNet 임베딩
my_face_embedding = np.load("my_mobileface_embedding.npy")  # (128,) or (512,) etc.

############################
# (E) Mediapipe Pose 등 동일
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
# (F) DeepSORT
############################
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    embedder='mobilenet',
    half=True,
    embedder_gpu=True
)




############################
# (G) 메인 루프
############################
window_name = "DeepSORT + YOLO(SEG) + MobileFaceNet + Pose +Depth"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 720)


dangerous_ids = set()
track_me_status = {}
track_arcface_count= {}
MAX_ARCFACE_FRAMES= 20
sim = 0


prev_time = time.time()

while True:
    # RealSense 프레임 가져오기
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    # OpenCV에서 사용할 수 있도록 변환
    frame = np.asanyarray(color_frame.get_data())

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame, conf=0.5)
    results_seg = model_seg(rgb_frame, conf=0.5)


    # YOLO detection
    det= results[0]
    boxes2 = det.boxes
    masks2 = results_seg[0].masks

    person_detections= []
    weapon_boxes= []

    if boxes2 is not None:
        for i, box in enumerate(boxes2):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)
            label = f"{model.names[class_id]}: {conf:.2f}"
            cv2.putText(frame, label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

            # 세그 윤곽선(사람만)
            if masks2 is not None and class_id==0:
                if i < len(masks2.data):
                    single_mask= masks2.data[i].cpu().numpy()
                    mask_bin= (single_mask>0.5).astype(np.uint8)
                    contours,_= cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, (0,255,255), 2)

            # DeepSORT
            if class_id==0:
                w= x2 - x1
                h= y2 - y1
                person_detections.append(((x1,y1,w,h), conf,0))
            elif class_id in [1,2]:
                weapon_boxes.append((x1,y1,x2,y2))

    # DeepSORT update
    tracks= tracker.update_tracks(person_detections, frame=rgb_frame)
    tracked_boxes=[]
    for t in tracks:
        if not t.is_confirmed() or t.time_since_update>1:
            continue
        tid= t.track_id
        l,t_,r,b_ = map(int,t.to_ltrb())
        tracked_boxes.append((tid,l,t_,r,b_))


        # 깊이 정보 가져오기
        distance = get_center_window_distance(depth_frame, l, t_, r, b_)
        distance_text = f"{distance:.2f} m" if distance > 0 else "N/A"

        # 박스 및 거리 정보 표시
        cv2.rectangle(frame, (l, t_), (r, b_), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{tid}", (l, t_ - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Dist: {distance_text}", (l, t_ - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)



    # MobileFaceNet 로직
    time1 = time.time()
    max_arcface_frames=10 # id 별로 초기 10프레임만 얼굴 인식
    
    for (tid, px1, py1, px2, py2) in tracked_boxes:
        if tid not in track_me_status:
            track_me_status[tid]= False
        if tid not in track_arcface_count:
            track_arcface_count[tid]=0

        if track_me_status[tid]==False and track_arcface_count[tid]<max_arcface_frames:
            print(f"Running MobileFacenet for ID {tid} (Frame {track_arcface_count[tid]+1}/{MAX_ARCFACE_FRAMES})")
            track_arcface_count[tid]+=1

            
            # MobileFaceNet 임베딩 추출
            PAD=10
            sub_face= frame[max(0,py1-PAD): py2+PAD, max(0,px1-PAD): px2+PAD]
            if sub_face.size==0:
                continue

            emb_image = get_mobileface_embedding(sub_face)
            if emb_image is not None:
                same_person, sim = is_my_face(emb_image, my_face_embedding, threshold=0.4)
                if same_person:
                    track_me_status[tid]=True
                    if tid in dangerous_ids:
                        dangerous_ids.remove(tid)

        # 시각화
        if track_me_status[tid]:
            text_arc= f"          Me(sim={sim:.2f})"
            color=(0,255,0)
        else:
            text_arc= "           NotMe(sim={sim:.2f})"
            color=(0,0,255)
        cv2.putText(frame,text_arc,(px1, py1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
    time2 = time.time()
    print("MobileFacenet running time: ", time2-time1)

    # 무기 교차 & NotMe => dangerous
    for (tid, px1, py1, px2, py2) in tracked_boxes:
        if not track_me_status[tid]:
            pbox= (px1, py1, px2, py2)
            for wb in weapon_boxes:
                if boxes_overlap(pbox, wb):
                    dangerous_ids.add(tid)
                    break

    # Dangerous => Mediapipe pose
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

    # FPS 표시
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
