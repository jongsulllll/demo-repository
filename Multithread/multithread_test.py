import cv2
import threading
import queue
import time

cap = cv2.VideoCapture(0)
frame_queue1= queue.Queue()
frame_queue2 = queue.Queue()
frame_queue3 = queue.Queue()

def detect_objects():
    c=0
    while True:
        c=c+1
        frame = frame_queue1.get()
        if frame is None:
            break
        #print("1 객체 감지 수행")
        time.sleep(0.12)
        print("1:  ",frame[100][100],'cnt : ',c)
        

def track_objects():
    c=0
    while True:
        c=c+1
        frame = frame_queue2.get()
        if frame is None:
            break
        #print("2 객체 추적 수행")
        time.sleep(0.01)
        print("2:  ",frame[100][100],'cnt : ',c)

def recognize_faces():
    c=0
    while True:
        c=c+1
        frame = frame_queue3.get()
        if frame is None:
            break
        #print("3 굴 인식 수행")
        #time.sleep(0.01)
        print("3:  ",frame[100][100],'cnt : ',c)

# 스레드 생성 및 시작
threads = [
    threading.Thread(target=detect_objects, daemon=True),
    threading.Thread(target=track_objects, daemon=True),
    threading.Thread(target=recognize_faces, daemon=True),
]

for t in threads:
    t.start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패!")
        break

    # 같은 프레임을 3개의 스레드가 가져갈 수 있도록 3번 삽입
    frame_queue1.put(frame)
    frame_queue2.put(frame)
    frame_queue3.put(frame)

# 종료 신호 (None) 삽입
for _ in range(3):
    frame_queue1.put(None)
    frame_queue2.put(None)
    frame_queue3.put(None)

# 스레드 종료 대기
for t in threads:
    t.join()

cap.release()
cv2.destroyAllWindows()

