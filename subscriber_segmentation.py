#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray, Int32MultiArray, Float32
#from ultralytics.nn.autobackend import AutoBackend
import numpy as np
import cv2

class segment:
    def __init__(self):
        #model = AutoBackend('/home/dev/kobuki_ws/src/kobuki/kobuki_node/yolov8n-seg.pt', device=device, dnn=dnn, fp16=half)
        rospy.init_node('segmentation_depth_node')  
        # rospy.Subscriber("/custom/color/image_raw", Image, self.image_callback, queue_size = 1)
        # rospy.Subscriber("/custom/depth/image_raw", Image, self.depth_callback, queue_size = 1)
        rospy.Subscriber("/depth/RGB_image", Image, self.image_callback, queue_size = 1)
        rospy.Subscriber("/depth/Depth_image", Image, self.depth_callback, queue_size = 1)
        rospy.Subscriber("/target_bbox", Int32MultiArray, self.target_bbox_callback, queue_size = 1)

        self.seg_pub = rospy.Publisher("/target_distance_meter", Float32, queue_size = 1)
        #rospy.spin()
        self.bridge = CvBridge()
        self.rgb_frame = None
        self.depth_frame = None


    def image_callback(self, msg):
        self.rgb_frameframe = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        self.depth_frame = self.depth_frame.astype(float) / 1000.0  

    def target_bbox_callback(self, msg):
        x1, y1, x2, y2 = map(int, msg.data)
        cx, cy = (x1+x2)/2, (y1+y2)/2
        dx, dy = (x2-x1)/8, (y2-y1)/8
        window = self.depth_frame[int(cx-dx):int(cx+dx), int(cy-dy):int(cy+dy)]

        # Filter valid depth values (0.5 < depth < 8)
        valid_depths = window[(window > 0.3) & (window < 8)]
        depth = np.mean(valid_depths)
        if not np.isnan(depth):
            # Return mean depth or 0 if no valid values
            print(f'calculated depth is {depth:.4f}m')
            self.seg_pub.publish(depth + 100*int(cx))


    # def get_distance(self):
    #     w

if __name__ == '__main__':
    try:
        segment()
        rospy.spin()
        #rate = rospy.Rate(20)  # 10 Hz
        # while not rospy.is_shutdown():
        #     segment.get_distance()  # Process and visualize images
        #     rate.sleep()
    except rospy.ROSInterruptException:
        pass


