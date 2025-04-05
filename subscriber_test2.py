#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def image_callback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv2.imshow("Camera2", cv_image)
    cv2.waitKey(1)

rospy.init_node('camera_listener2')
rospy.Subscriber("/topcam/image_raw", Image, image_callback)
rospy.spin()
