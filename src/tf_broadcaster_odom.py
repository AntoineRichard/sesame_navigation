#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import TransformStamped, PoseStamped

from tf2_ros import TransformBroadcaster


class TFBroadcaster:
    """
    This class is responsible for broadcasting the transforms between the odom frame
    and the base_link frame of the UAV.
    """

    def __init__(self):
        self.child_frame_ = rospy.get_param("~child_frame", "base_link")
        self.parent_frame_ = rospy.get_param("~parent_frame", "odom")

        self.tf_broadcaster_ = TransformBroadcaster()

        self.pose_sub_ = rospy.Subscriber(
            "/mavros/local_position/pose", PoseStamped, self.poseCallback
        )
        self.pose_pub_ = rospy.Publisher(
            "/mavros/local_position/pose_stamped", PoseStamped, queue_size=1
        )

        self.pose = None

    def poseCallback(self, msg: PoseStamped) -> None:
        """
        Callback function for the pose subscriber. It receives the pose of the UAV
        and broadcasts the transform between the odom frame and the base_link frame.

        Args:
            msg (PoseStamped): PoseStamped message containing the pose of the UAV.
        """

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.parent_frame_
        t.child_frame_id = self.child_frame_
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation.x = msg.pose.orientation.x
        t.transform.rotation.y = msg.pose.orientation.y
        t.transform.rotation.z = msg.pose.orientation.z
        t.transform.rotation.w = msg.pose.orientation.w

        self.tf_broadcaster_.sendTransform(t)
        msg.header.frame_id = self.parent_frame_
        msg.header.stamp = t.header.stamp
        self.pose_pub_.publish(msg)


if __name__ == "__main__":
    rospy.init_node("tf_broadcaster", anonymous=True)
    tf_broadcaster = TFBroadcaster()
    rospy.spin()
