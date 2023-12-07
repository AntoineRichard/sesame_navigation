#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import PoseStamped, TransformStamped

from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


class TFBroadcaster:
    """
    This class is responsible for broadcasting the transforms between the map frame
    and the odom frames of the UAVs.
    """

    def __init__(self):
        self.frame_uav_1_ = rospy.get_param("~frame_uav_1", "uav_1/odom")
        self.frame_uav_2_ = rospy.get_param("~frame_uav_2", "uav_2/odom")
        self.parent_frame_ = rospy.get_param("~parent_frame", "map")

        self.uav_1_px_ = rospy.get_param("~uav_1_px", 0)
        self.uav_1_py_ = rospy.get_param("~uav_1_py", 0)
        self.uav_1_pz_ = rospy.get_param("~uav_1_pz", 0)
        self.uav_2_px_ = rospy.get_param("~uav_2_px", 1)
        self.uav_2_py_ = rospy.get_param("~uav_2_py", 1)
        self.uav_2_pz_ = rospy.get_param("~uav_2_pz", 0)

        self.tf_broadcaster_ = TransformBroadcaster()
        self.publishTfs()

        self.pose = None

    def publishTfs(self) -> None:
        """
        Publishes the transforms between the map frame and the odom frames of the UAVs.
        """

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.parent_frame_
        t.child_frame_id = self.frame_uav_1_
        t.transform.translation.x = self.uav_1_px_
        t.transform.translation.y = self.uav_1_py_
        t.transform.translation.z = self.uav_1_pz_
        t.transform.rotation.x = 0
        t.transform.rotation.y = 0
        t.transform.rotation.z = 0
        t.transform.rotation.w = 1

        self.tf_broadcaster_.sendTransform(t)

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.parent_frame_
        t.child_frame_id = self.frame_uav_2_
        t.transform.translation.x = self.uav_2_px_
        t.transform.translation.y = self.uav_2_py_
        t.transform.translation.z = self.uav_2_pz_
        t.transform.rotation.x = 0
        t.transform.rotation.y = 0
        t.transform.rotation.z = 0
        t.transform.rotation.w = 1

        self.tf_broadcaster_.sendTransform(t)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publishTfs()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("tf_broadcaster", anonymous=True)
    tf_broadcaster = TFBroadcaster()
    tf_broadcaster.run()
