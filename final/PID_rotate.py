#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32

class KobukiPIDController:
    def __init__(self):
        rospy.init_node('kobuki')
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
        self.seg_pub = rospy.Subscriber('/target_distance_meter', Float32, self.distance_callback, queue_size=1)
        self.state_sub = rospy.Subscriber('/state', Int32, self.state_update, queue_size=1)

        self.state = 0
        # PID Parameters
        self.TARGET_DISTANCE = 1.6  # Target distance in meters
        self.Kp_linear = 0.4   # Proportional gain for linear velocity
        self.Ki_linear = 0.01  # Integral gain for linear velocity
        self.Kd_linear = 0.1   # Derivative gain for linear velocity

        self.Kp_angular = 0.5   # Proportional gain for angular velocity
        self.Ki_angular = 0.01  # Integral gain for angular velocity
        self.Kd_angular = 0.2   # Derivative gain for angular velocity

        self.prev_error_linear = 0
        self.integral_linear = 0
        self.linear_limit = 0.3

        self.prev_error_angular = 0
        self.integral_angular = 0
        self.angular_limit = 0.6

        rospy.spin()

    def state_update(self, msg):
        self.state = msg.data

    def compute_pid(self, error, prev_error, integral, Kp, Ki, Kd, vel_limit):
        integral += error
        derivative = error - prev_error
        prev_error = error
        output = (Kp * error) + (Ki * integral) + (Kd * derivative)
        if abs(output) > vel_limit:
            output = vel_limit * (1 if output > 0 else -1)
            integral = 0
        return output, prev_error, integral

    def distance_callback(self, msg):
        dy = int(msg.data//100)
        distance = msg.data - dy*100
        angle_error = 320 - dy
        print('calculated center pixel is ', dy, 'distance is ', distance)
        error_linear = self.TARGET_DISTANCE - distance
        velocity_linear, self.prev_error_linear, self.integral_linear = self.compute_pid(
            error_linear, self.prev_error_linear, self.integral_linear, self.Kp_linear, self.Ki_linear, self.Kd_linear, self.linear_limit
        )
        angular_velocity, self.prev_error_angular, self.integral_angular = self.compute_pid(
            angle_error, self.prev_error_angular, self.integral_angular, self.Kp_angular, self.Ki_angular, self.Kd_angular, self.angular_limit
        )
        print(f'vel ={velocity_linear}    ang: {angular_velocity}')
        
        cmd_vel = Twist()
        cmd_vel.linear.x = velocity_linear
        cmd_vel.angular.z = angular_velocity
        if self.state == 1:
            self.cmd_vel_pub.publish(cmd_vel)

if __name__ == '__main__':
    KobukiPIDController()
