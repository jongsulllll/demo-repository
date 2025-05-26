#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32

class ArucoPIDFollower:
    def __init__(self):
        rospy.init_node('aruco_pid_follower')
        self.cmd_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('/robot_pose', Pose2D, self.pose_callback, queue_size=1)
        self.state_sub = rospy.Subscriber('/state', Int32, self.state_update, queue_size=1)

        # 목표 경로 (x, y, yaw)
        self.waypoints = [
            (0.6,0.6, 0.0),
            (0.6, 2.3, 0.0),
            (3, 2.3, 0.0),
            (3, 0.5, 0.0),
            (0.6, 0.6, 0.0),
            (0.6, 2.3,0.0)
        ]
        self.current_pose = None
        self.current_index = 0
        self.state = 0

        # PID 제어 계수
        self.Kp_lin = 0.4
        self.Ki_lin = 0.0
        self.Kd_lin = 0.0
        self.Kp_ang = 0.3
        self.Ki_ang = 0.0
        self.Kd_ang = 0.0

        self.error_sum_lin = 0
        self.prev_error_lin = 0
        self.error_sum_ang = 0
        self.prev_error_ang = 0

        self.goal_tolerance = 0.15  # 도착 허용 거리

        self.rate = rospy.Rate(10)  # 10Hz
        self.run()

    def state_update(self, msg):
        self.state = msg.data

    def pose_callback(self, msg):
        self.current_pose = msg

    def compute_pid(self, error, prev_error, error_sum, Kp, Ki, Kd, limit):
        error_sum += error
        d_error = error - prev_error
        output = Kp * error + Ki * error_sum + Kd * d_error
        output = max(min(output, limit), -limit)  # 속도 제한
        return output, error_sum, error

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def run(self):
        while not rospy.is_shutdown():

            if self.state == 0:
                if self.current_pose is None or self.current_index >= len(self.waypoints):
                    self.rate.sleep()
                    continue

                target_x, target_y, _ = self.waypoints[self.current_index]
                print(target_x, target_y)
                dx = target_x - self.current_pose.x
                dy = target_y - self.current_pose.y
                distance = math.hypot(dx, dy)

                # 방향 오차 계산
                desired_yaw = math.atan2(dy, dx)
                yaw_error = self.normalize_angle(desired_yaw - self.current_pose.theta)
    #            print(f'desire yaw : {desired_yaw},  yaw error = {yaw_error}')
                # PID 계산
                lin_vel, self.error_sum_lin, self.prev_error_lin = self.compute_pid(
                    distance, self.prev_error_lin, self.error_sum_lin, self.Kp_lin, self.Ki_lin, self.Kd_lin, 0.08
                )
                ang_vel, self.error_sum_ang, self.prev_error_ang = self.compute_pid(
                    yaw_error, self.prev_error_ang, self.error_sum_ang, self.Kp_ang, self.Ki_ang, self.Kd_ang, 0.9
                )
    #            print(ang_vel, '--------')
                cmd = Twist()
                if distance > self.goal_tolerance:
                    cmd.linear.x = lin_vel
                    cmd.angular.z = -ang_vel
                else:
                    rospy.loginfo(f"🟩 Waypoint {self.current_index} reached")
                    self.current_index += 1
                if self.state == 0: # state 1 follow person
                    self.cmd_pub.publish(cmd)

            self.rate.sleep()

        # 정지 명령
        self.cmd_pub.publish(Twist())
        rospy.loginfo("✅ Path following complete.")

if __name__ == '__main__':
    ArucoPIDFollower()
    

