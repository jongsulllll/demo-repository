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
Kp_linear = 0.4   # Proportional gain for linear velocity
Ki_linear = 0.01  # Integral gain for linear velocity
Kd_linear = 0.1   # Derivative gain for linear velocity

Kp_angular = 0.4   # Proportional gain for angular velocity
Ki_angular = 0.01  # Integral gain for angular velocity
Kd_angular = 0.1   # Derivative gain for angular velocity

prev_error_linear = 0
integral_linear = 0
prev_error_angular = 0
integral_angular = 0

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

def compute_pid(error, prev_error, integral, Kp, Ki, Kd):
    """ Compute PID velocity command. """
    integral += error
    derivative = error - prev_error
    prev_error = error
    return (Kp * error) + (Ki * integral) + (Kd * derivative), prev_error, integral

def get_center_window_distance(depth_frame, x1, y1, x2, y2, window_size=30):
    """ Get the average distance of a 10x10 window at the center of the bounding box. """
    # Calculate the center of the bounding box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Define the window around the center
    half_window = window_size // 2
    window_x1 = max(0, cx - half_window)
    window_y1 = max(0, cy - half_window)
    window_x2 = min(depth_frame.get_width(), cx + half_window)
    window_y2 = min(depth_frame.get_height(), cy + half_window)
    
    # Get distances for pixels in the 10x10 window
    distances = []
    for x in range(window_x1, window_x2):
        for y in range(window_y1, window_y2):
            distance = depth_frame.get_distance(x, y)
            if distance > 0:  # Ignore invalid distances
                distances.append(distance)

    if distances:
        return np.mean(distances)
    else:
        return 0  # Return 0 if no valid distances

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

                    # Check if the bounding box is bigger than 10x10
                    box_width = x2 - x1  #add rotate codes to set person bounding box middle match camera middle PID
                    box_height = y2 - y1
                    if box_width > 10 and box_height > 10:
                        # Calculate the center 10x10 window's distance average
                        depth = get_center_window_distance(depth_frame, x1, y1, x2, y2, window_size=30)

                        # PID control for linear velocity (distance control)
                        error_linear = TARGET_DISTANCE - depth
                        velocity_linear, prev_error_linear, integral_linear = compute_pid(
                            error_linear, prev_error_linear, integral_linear, Kp_linear, Ki_linear, Kd_linear
                        )

                        # PID control for angular velocity (angle control)
                        image_center_x = color_image.shape[1] // 2
                        box_center_x = (x1 + x2) // 2
                        error_angular = image_center_x - box_center_x
                        angular_velocity, prev_error_angular, integral_angular = compute_pid(
                            error_angular, prev_error_angular, integral_angular, Kp_angular, Ki_angular, Kd_angular
                        )

                        # Limit velocity and angular velocity
                        velocity_linear = max(min(velocity_linear, 0.4), -0.4)  # Limit linear speed
                        angular_velocity = max(min(angular_velocity, 0.4), -0.4)  # Limit angular speed

                        # Publish velocity command
                        cmd_vel = Twist()
                        cmd_vel.linear.x = velocity_linear
                        cmd_vel.angular.z = angular_velocity
                        cmd_vel_pub.publish(cmd_vel)

                        # Draw bounding box and display distance
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(color_image, f"{depth:.2f}m", (x1, y1 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # If the box is too small, stop moving
                        cmd_vel_pub.publish(Twist())  # Stop robot

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
