#!/usr/bin/env python

import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from geometry_msgs.msg import Twist

# ROS Initialization
rospy.init_node('kobuki_follow_person', anonymous=True)
cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)

# PID Parameters
TARGET_DISTANCE = 1.0  # Target distance in meters
Kp = 0.5   # Proportional gain
Ki = 0.01  # Integral gain
Kd = 0.1   # Derivative gain
prev_error = 0
integral = 0

# Load YOLOv8 Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("/home/dev/runs/detect/train55/weights/epoch180.pt").to(device)
print("Camera initializing with", device, '...')

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

def compute_pid(error):
    """ Compute PID velocity command. """
    global prev_error, integral
    integral += error
    derivative = error - prev_error
    prev_error = error
    return (Kp * error) + (Ki * integral) + (Kd * derivative)

try:
    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLOv8 inference
        results = model(color_image, device=device)

        detected = False
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])  # Class index

                if cls == 0:  # Assuming class 0 is 'person'
                    detected = True
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    depth = depth_frame.get_distance(cx, cy)

                    # PID control
                    error = TARGET_DISTANCE - depth
                    velocity = compute_pid(error)

                    # Publish velocity command
                    cmd_vel = Twist()
                    cmd_vel.linear.x = max(min(velocity, 0.5), -0.5)  # Limit speed
                    cmd_vel_pub.publish(cmd_vel)

                    # Draw bounding box and display distance
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_image, f"{depth:.2f}m", (x1, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break  # Track only the first detected person

        # Stop robot if no person is detected
        if not detected:
            cmd_vel_pub.publish(Twist())

        # Show detection
        cv2.imshow("Kobuki Person Following", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    cmd_vel_pub.publish(Twist())  # Stop robot before exiting
