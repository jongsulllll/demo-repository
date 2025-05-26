#!/usr/bin/env python3

######## coord_v10기반 수정
########카메라 시야 중심부 가중치 포함된 버전
import cv2
import cv2.aruco as aruco
import numpy as np
import math
import rospy
from geometry_msgs.msg import Pose2D
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Int32
rospy.init_node("camera_localization_node")
pose_pub = rospy.Publisher("/robot_pose", Pose2D, queue_size=10)

cameraMatrix = np.load('/home/dev/kobuki_ws/src/kobuki/kobuki_node/calibration_matrix.npy')
distCoeffs = np.load('/home/dev/kobuki_ws/src/kobuki/kobuki_node/distortion_coefficients.npy')
marker_length = 0.16  # meters
FIXED_Z = -2.5

marker_world_pos = {
    0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.5, 0.0, 0.0]),
    2: np.array([3.0, 0.0, 0.0]), 3: np.array([4.5, 0.0, 0.0]),
    4: np.array([0.0, 1.5, 0.0]), 5: np.array([1.5, 1.5, 0.0]),
    6: np.array([3.0, 1.5, 0.0]), 7: np.array([4.5, 1.5, 0.0]),
    8: np.array([0.0, 3.0, 0.0]), 9: np.array([1.5, 3.0, 0.0]),
    10: np.array([3.0, 3.0, 0.0]), 11: np.array([4.5, 3.0, 0.0]),
    12: np.array([0.7, 0.7, 0.0]), 13: np.array([2.3, 0.7, 0.0]),
    14: np.array([3.7, 0.7, 0.0]), 15: np.array([0.7, 2.3, 0.0]),
    16: np.array([2.3, 2.3, 0.0]), 17: np.array([3.7, 2.3, 0.0])
}

marker_orientation = {
    0: 0, 1: 0, 2: -180, 3: 0, 4: -180, 5: -180, 6: 0, 7: -180,
    8: 0, 9: 0, 10: -180, 11: 0, 12: -180, 13: -90, 14: -270,
    15: 0, 16: -90, 17: -90
}

def get_z_rotation_matrix(degrees):
    return R.from_euler('z', degrees, degrees=True).as_matrix()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_EXPOSURE, 20)

while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    frame_center = np.array([w // 2, h // 2])

    #corners, ids, _ = aruco.detectMarkers(gray, aruco_dict := aruco.getPredefinedDictionary(aruco.DICT_4X4_50), parameters := aruco.DetectorParameters())
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, cameraMatrix, distCoeffs)

        cam_positions, yaws, weights = [], [], []

        for i, m_id in enumerate(ids.flatten()):
            if m_id in marker_world_pos:
                rvec, tvec = rvecs[i], tvecs[i]
                world_pos = marker_world_pos[m_id]

                # 보정된 회전 행렬
                R_marker, _ = cv2.Rodrigues(rvec)
                correction_R = get_z_rotation_matrix(marker_orientation.get(m_id, 0))
                R_corrected = correction_R @ R_marker
                rvec_corrected, _ = cv2.Rodrigues(R_corrected)
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec_corrected, tvec, 0.1)

                # 마커 중심 위치
                marker_center = np.mean(corners[i][0], axis=0)
                pixel_dist = np.linalg.norm(marker_center - frame_center)

                # 거리 및 중심 기반 가중치 계산
                dist_weight = 1.0 / (np.linalg.norm(tvec) + 1e-6)
                center_weight = 1.0 / (pixel_dist + 10)  # 중심 보정
                total_weight = dist_weight * center_weight

                # 위치 변환
                yaw = R.from_matrix(R_corrected).as_euler('xyz', degrees=False)[2]
                cam_pos = -R_marker.T @ tvec.reshape(3, 1)
                cam_pos[2] = FIXED_Z
                cam_pos = (correction_R @ cam_pos + world_pos.reshape(3, 1)).flatten()

                cam_positions.append(cam_pos)
                yaws.append(yaw)
                weights.append(total_weight)

        weights = np.array(weights)
        weights /= np.sum(weights)
        camera_world_pos = np.average(cam_positions, axis=0, weights=weights)
        camera_yaw = np.average(yaws, weights=weights)

        # 퍼블리시
        pose_msg = Pose2D()
        pose_msg.x = float(camera_world_pos[0])
        pose_msg.y = float(camera_world_pos[1])
        pose_msg.theta = float(camera_yaw)
        pose_pub.publish(pose_msg)

        # 시각화
        text = f"X={pose_msg.x:.2f} Y={pose_msg.y:.2f} Yaw={math.degrees(pose_msg.theta):.2f}deg"
        cv2.putText(frame, text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (0, 0, 255), 5)

    cv2.imshow("Aruco Localization (Weighted)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
