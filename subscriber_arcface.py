#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32
import warnings
import dlib
import time
import numpy as np
from cv_bridge import CvBridge
from sklearn.metrics.pairwise import cosine_similarity
import insightface
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore", category=FutureWarning)
app = FaceAnalysis(name="buffalo_sc")  # "buffalo_m"은 MobileFaceNet 기반
app.prepare(ctx_id=0, det_size=(160,160))  # GPU라면 ctx_id=0, CPU는 -1

def get_mobileface_embedding(image_bgr):
    if image_bgr is None or image_bgr.size==0:
        return None
    faces = app.get(image_bgr)
    if len(faces)==0:
        return None
    return faces[0].embedding  # shape 예: (128,)

def is_my_face(face_embedding, my_embedding, threshold=0.4):
    sim = cosine_similarity([face_embedding], [my_embedding])[0][0]
    return (sim > threshold), sim


#sp = dlib.shape_predictor("./face_train/shape_predictor_68_face_landmarks.dat")  # Landmark predictor
#facerec = dlib.face_recognition_model_v1("./face_train/dlib_face_recognition_resnet_model_v1.dat")  # Face encoder
my_face_embedding = np.load("/home/dev/kobuki_ws/src/kobuki/kobuki_node/arcface/my_mobileface_embedding.npy")  # Load your pre-saved face embedding

class RGBDepthSubscriber:
    def __init__(self):
        rospy.init_node('rgb_dlib_subscriber', anonymous=True)
        
        self.bridge = CvBridge()
        self.face_dist = []
        
        # Subscribe to RGB and Depth images
        self.rgb_sub = rospy.Subscriber("/custom/color/image_raw", Image, self.rgb_callback)
        self.face_bbox = rospy.Subscriber("/yolo_det/face_bbox", Float32MultiArray, self.process_face)
        self.person_bbox_with_id = rospy.Subscriber("/SORT/person_bbox_with_id", Float32MultiArray, self.face_in_personbox)

        # Publisher for processed output
        self.is_my_face = rospy.Publisher("/my_face_id", Int32, queue_size=1)
        self.is_dangerous_person = rospy.Publisher("/dangerous_id", Int32, queue_size=1)

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert ROS image to OpenCV format
            #cv2.imshow("RGB Image", rgb_image)            
            #cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("RGB Callback Error: %s", str(e))
    
    def process_face(self, msg): 
        # use face xyxy to get face_descriptor
        try:  
            tit=time.time()
            #emb_image = get_mobileface_embedding(self.rgb_image[int(msg.data[1]):int(msg.data[3]), int(msg.data[0]):int(msg.data[2])])
            emb_image = get_mobileface_embedding(self.rgb_image)
            same_person, sim = is_my_face(emb_image, my_face_embedding, threshold=0.4)
            
            

            #bbox = msg.data[:]
            
        except Exception as e:
            rospy.logerr("process face Callback Error: %s", str(e))
    
    def face_in_personbox(self, msg):
        try:
            pass #print('person', msg.data)
            
        except Exception as e:
            rospy.logerr("process face Callback Error: %s", str(e))
    
        
        

if __name__ == '__main__':
    try:
        RGBDepthSubscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
