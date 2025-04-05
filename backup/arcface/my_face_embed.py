import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis

# 1) MobileFaceNet 모델(InsightFace) 초기화
app = FaceAnalysis(name="buffalo_sc")  # buffalo_s = MobileFaceNet 계열
app.prepare(ctx_id=0, det_size=(160,160))  # GPU 사용(ctx_id=0), CPU는 -1

def get_mobileface_embedding(image_bgr):
    # image_bgr: np.array(BGR)
    faces = app.get(image_bgr)
    if len(faces)==0:
        return None
    # 보통 faces[0].embedding shape = (128,) or (512,) 모델에 따라 다름
    return faces[0].embedding

# 내 얼굴 사진 폴더
my_face_folder = "face"
embeddings = []

for file in os.listdir(my_face_folder):
    if file.lower().endswith(('.jpg','.png','.jpeg')):
        path = os.path.join(my_face_folder, file)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        emb = get_mobileface_embedding(img_bgr)
        if emb is not None:
            embeddings.append(emb)

if len(embeddings)==0:
    raise ValueError("No embeddings generated from MobileFaceNet.")

# 평균 임베딩
avg_embedding = np.mean(embeddings, axis=0)
print("MobileFaceNet embedding shape:", avg_embedding.shape)

# 파일 저장
np.save("my_mobileface_embedding.npy", avg_embedding)
print("Saved my_mobileface_embedding.npy")
