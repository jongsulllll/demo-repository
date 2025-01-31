#!/usr/bin/env python3


# cmd_val X
# move_cmd.xyz -> control, 0.2 ~ 0.5 max


import rospy
from geometry_msgs.msg import Twist

def move_kobuki():
    # Initialize the ROS node
    rospy.init_node('move_kobuki', anonymous=True)

    # Create a publisher to send velocity commands to the robot
    pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)

    # Set the rate to control how often messages are published
    rate = rospy.Rate(10)

    # Create a Twist message to control velocity
    move_cmd = Twist()

    # Move forward with a linear velocity of 0.2 m/s and no angular velocity
    move_cmd.linear.x = 0.5
    move_cmd.angular.z = 0.0

    # Send movement commands for 5 seconds
    rospy.loginfo("Moving forward")
    for _ in range(12):  # 10 Hz * 5 seconds = 50 iterations
        pub.publish(move_cmd)
        rate.sleep()

    # Stop the robot
    move_cmd.linear.x = 0.0
    rospy.loginfo("Stopping robot")
    pub.publish(move_cmd)
    rospy.sleep(1)

    # Rotate the robot in place
    move_cmd.linear.x = 0.0
    move_cmd.angular.z = 0.3  # Rotate at 0.3 rad/s

    rospy.loginfo("Rotating robot")
    for _ in range(50):  # 10 Hz * 5 seconds = 50 iterations
        pub.publish(move_cmd)
        rate.sleep()

    # Stop the robot after rotation
    move_cmd.angular.z = 0.0
    rospy.loginfo("Stopping robot after rotation")
    pub.publish(move_cmd)

if __name__ == '__main__':
    try:
        move_kobuki()
    except rospy.ROSInterruptException:
        pass
