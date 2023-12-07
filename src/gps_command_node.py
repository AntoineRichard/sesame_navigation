#!/usr/bin/env python3

import rospy

from geographic_msgs.msg import GeoPoseStamped
from std_msgs.msg import Bool


class GeoPosePublisher:
    def __init__(self):
        self.rate = rospy.get_param("~rate", 20)

        self.geo_pose_pub = rospy.Publisher(
            "/uav_1/mavros/setpoint_position/global", GeoPoseStamped, queue_size=1
        )
        self.geo_pose_sub = rospy.Subscriber(
            "/uav_1/control_stack/gps_command", GeoPoseStamped, self.geoPoseCallback
        )
        self.enabled_sub = rospy.Subscriber(
            "/uav_1/control_stack/enable_gps_commands", Bool, self.enableCallback
        )

        self.enabled = True
        self.geo_pose = None

    def geoPoseCallback(self, msg):
        self.geo_pose = msg

    def enableCallback(self, msg):
        self.enabled = msg.data

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if (self.enabled) and (self.geo_pose is not None):
                self.geo_pose_pub.publish(self.geo_pose)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("gps_command_node", anonymous=True)
    geo_pose_publisher = GeoPosePublisher()
    geo_pose_publisher.run()
