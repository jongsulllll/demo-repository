import cv2
import time
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90) #MAX 90fps
pipeline.start(config)

#cap = cv2.VideoCapture(1)
#cap.set(cv2.CAP_PROP_FPS, 300)
#fps = cap.get(cv2.CAP_PROP_FPS)
#print("current FPS is ", fps)
prev_time = time.time()
while True:
	start_time = time.time()
	frames = pipeline.wait_for_frames()
	depth_frame = frames.get_depth_frame()
	if not depth_frame:
		continue
	depth_image = np.asanyarray(depth_frame.get_data())
	depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
	
#	cv2.imshow('dd',frame)
	cv2.imshow('depth', depth_image)
	end_time = time.time()
	print(-start_time + end_time)
	if cv2.waitKey(1) & 0xFF == 27:
		break
cap.release()
cv2.destroyAllWindow()
