#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

def move_kobuki():
    # Initialize the ROS node
    rospy.init_node('move_kobuki', anonymous=True)

    # Create a publisher to send velocity commands
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # Set loop rate (10 Hz)
    rate = rospy.Rate(10)

    # Create a Twist message
    move_cmd = Twist()

    # Move forward at 0.2 m/s
    move_cmd.linear.x = 0.2
    move_cmd.angular.z = 0.0

    rospy.loginfo("Moving forward")
    for _ in range(50):  # 10 Hz * 5 seconds = 50 iterations
        pub.publish(move_cmd)
        rate.sleep()

    # Stop the robot
    move_cmd.linear.x = 0.0
    rospy.loginfo("Stopping robot")
    pub.publish(move_cmd)
    rospy.sleep(1)

if __name__ == '__main__':
    try:
        move_kobuki()
    except rospy.ROSInterruptException:
        pass
