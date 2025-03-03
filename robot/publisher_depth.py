#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import time 

class RealSensePublisher:
    def __init__(self):
        rospy.init_node("realsense_publisher", anonymous=True)
        self.bridge = CvBridge()
        
        # Create publishers for color and depth images
        self.color_pub = rospy.Publisher("/custom/color/image_raw", Image, queue_size=10)
        self.depth_pub = rospy.Publisher("/custom/depth/image_raw", Image, queue_size=10)
                
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        
    def publish_images(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
        rate = rospy.Rate(30)  # 30 FPS
        while not rospy.is_shutdown():
            frames = self.pipeline.wait_for_frames()
            # color_frame = frames.get_color_frame()
            # depth_frame = frames.get_depth_frame()

            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            
            if not color_frame or not depth_frame:
                continue
            
            # Convert frames to OpenCV format
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Convert to ROS Image messages
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
            
            # Publish images
            self.color_pub.publish(color_msg)
            self.depth_pub.publish(depth_msg)
            
            rate.sleep()

if __name__ == "__main__":
    try:
        node = RealSensePublisher()
        node.publish_images()
    except rospy.ROSInterruptException:
        pass
