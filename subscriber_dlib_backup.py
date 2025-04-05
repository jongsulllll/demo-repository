#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
import dlib
import time
import numpy as np
from cv_bridge import CvBridge

sp = dlib.shape_predictor("/home/dev/kobuki_ws/src/kobuki/kobuki_node/face_train/shape_predictor_68_face_landmarks.dat")  # Landmark predictor
facerec = dlib.face_recognition_model_v1("/home/dev/kobuki_ws/src/kobuki/kobuki_node/face_train/dlib_face_recognition_resnet_model_v1.dat")  # Face encoder
my_face_embedding = np.load("/home/dev/kobuki_ws/src/kobuki/kobuki_node/face_train/face_embeddings.npy")  # Load your pre-saved face embedding

class RGBDepthSubscriber:
    def __init__(self):
        rospy.init_node('rgb_dlib_subscriber', anonymous=True)
        
        self.bridge = CvBridge()
        self.face_dist = []
        self.image_update = 0
        self.win=None
        self.frame_count = 0
        self.toggle = 0
        # Subscribe to RGB and Depth images
        self.rgb_sub = rospy.Subscriber("/custom/color/image_raw", Image, self.rgb_callback, queue_size = 1)
        self.face_bbox = rospy.Subscriber("/yolo_det/face_bbox", Float32MultiArray, self.process_face, queue_size = 1)
        self.person_bbox_with_id = rospy.Subscriber("/SORT/person_bbox_with_id", Float32MultiArray, self.face_in_personbox)

        # Publisher for processed output
        # self.is_my_face = rospy.Publisher("/my_face_id", Int32, queue_size=1)
        # self.is_dangerous_person = rospy.Publisher("/dangerous_id", Int32, queue_size=1)
        time.sleep(3)
        while True:
            cv2.imshow('qwer',self.win)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert ROS image to OpenCV format
            #cv2.imshow("RGB Image", rgb_image)            
            #cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("RGB Callback Error: %s", str(e))
    
    def process_face(self, msg): 
        # if self.toggle == 0:
        #     self.toggle = 1
        # else:
        #     self.toggle = 0
        
        # use face xyxy to get face_descriptor
        try:
            self.win = self.rgb_image
            a=time.time()
            image = self.rgb_image
            cv2.rectangle(self.win, (int(msg.data[0]), int(msg.data[1]) + int(0.33*(int(msg.data[3]) - int(msg.data[1])))), \
                (int(msg.data[2]), int(msg.data[3])), (0,255,0),2)  
            time.sleep(0.03)
            shape = sp(image, dlib.rectangle(int(msg.data[0]), int(msg.data[1]) + int(0.33*(int(msg.data[3]) - int(msg.data[1]))), int(msg.data[2]), int(msg.data[3])))
            face_descriptor = facerec.compute_face_descriptor(image, shape)
            dist = np.linalg.norm(np.array(face_descriptor) - my_face_embedding)
            print(dist)
            print(f'box time i s {time.time() -a}')
            # t5=time.time()
            # print(f'{t2-t1:.6f} {t3-t2:.6f} {t4-t3:.6f} {t5-t4:.6f}')
            if dist < 0.5:
                print("hi")
            self.image_update = 0
            
            # if cnt%30 == 0:
            #cv2.imwrite(f'image_{cnt}.jpg')

            # bbox = msg.data[:]
            
        except Exception as e:
            rospy.logerr("process face Callback Error: %s", str(e))

    def face_in_personbox(self, msg):
        try:
            pass #print('person', msg.data)
            
        except Exception as e:
            rospy.logerr("process face Callback Error: %s", str(e))
    
        
cnt = 1

if __name__ == '__main__':
    try:
        RGBDepthSubscriber()
        rospy.spin()
        # while True:
        #     print('qwe0')
        #     cv2.imshow('asdf', win)
        #     if cv2.waitKey(1) == ord('q'):  # 1 millisecond
        #         exit()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
