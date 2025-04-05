#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class RGBDepthSubscriber:
    def __init__(self):
        rospy.init_node('rgb_depth_subscriber', anonymous=True)
        
        self.bridge = CvBridge()

        # Subscribe to RGB and Depth images
        self.rgb_sub = rospy.Subscriber("/custom/color/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/custom/depth/image_raw", Image, self.depth_callback)

    def rgb_callback(self, msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert ROS image to OpenCV format
            cv2.imshow("RGB Image", rgb_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("RGB Callback Error: %s", str(e))

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")  # Depth images are usually 16-bit
            cv2.imshow("Depth Image", depth_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("Depth Callback Error: %s", str(e))

if __name__ == '__main__':
    try:
        RGBDepthSubscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
