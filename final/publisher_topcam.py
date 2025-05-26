#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def webcam_publisher():
    rospy.init_node('webcam_publisher', anonymous=True)
    
    # Create a publisher for the webcam image
    image_pub = rospy.Publisher("/topcam/image_raw", Image, queue_size=10)
    
    bridge = CvBridge()
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Open default webcam (change to 1, 2, etc. for other cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution if needed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
    cap.set(cv2.CAP_PROP_FPS, 30)            # Set desired FPS (check your webcam specs)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer lag

    if not cap.isOpened():
        rospy.logerr("Cannot open webcam")
        return

    rate = rospy.Rate(30)  # Set FPS (Adjust as needed)
    
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("Failed to capture image")
            continue
        
        # Convert OpenCV image to ROS Image message
        ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        
        # Publish the image
        image_pub.publish(ros_image)
        
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        webcam_publisher()
    except rospy.ROSInterruptException:
        pass

