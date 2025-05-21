#!/usr/bin/env python3

import rospy
import threading
import queue
from std_msgs.msg import Float32
from sensor_msgs.msg import Image as Image_ROS
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.data.augment import LetterBox

# YOLO model path
MODEL_PATH = 'weights/yolov8n-seg.pt'

class SegmentPublisher:
    def __init__(self):
        rospy.init_node('segment_pid_publisher')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/custom/color/image_raw', Image_ROS, self.image_callback)
        self.dist_pub = rospy.Publisher('/target_distance_meter', Float32, queue_size=10)

        self.device = select_device('')
        self.model = YOLO(MODEL_PATH).to(self.device)
        self.model.fuse()
        self.imgsz = check_imgsz((640, 640), stride=32)

        self.frame_queue = queue.Queue(maxsize=1)
        self.run_thread = threading.Thread(target=self.run_detection)
        self.run_thread.start()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def run_detection(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.frame_queue.empty():
                rate.sleep()
                continue

            frame = self.frame_queue.get()
            img = cv2.resize(frame, self.imgsz)
            img_tensor = torch.from_numpy(img).to(self.device)
            img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            results = self.model(img_tensor)[0]
            if results.masks is not None and len(results.masks.xy) > 0:
                # 첫 번째 객체만 사용 (예시)
                mask = results.masks.data[0].cpu().numpy()
                mask_bin = (mask > 0.5).astype(np.uint8)

                M = cv2.moments(mask_bin)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # 중심 좌표 cx를 기반으로 가상 거리 생성 (실제는 depth 센서 사용)
                    fake_distance = 1.6  # 예시 거리 (m)
                    publish_value = cx * 100 + fake_distance
                    self.dist_pub.publish(Float32(data=publish_value))

            rate.sleep()

if __name__ == '__main__':
    try:
        SegmentPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
