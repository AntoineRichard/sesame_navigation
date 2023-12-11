#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

import tf2_ros
import tf2_geometry_msgs


class SafetySwitch:
    def __init__(self):
        self.max_error_ = rospy.get_param("~max_error", 0.5)
        self.global_frame_ = rospy.get_param("~global_frame", "map")
        self.use_estimated_pose_ = False

        self.true_pose_ = None
        self.est_pose_ = None

        self.tf_buffer_ = tf2_ros.Buffer(rospy.Duration(100.0))
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_)

        true_pose_sub_ = rospy.Subscriber(
            "true_pose", PoseStamped, self.true_pose_callback, queue_size=1
        )
        estimated_pose_sub_ = rospy.Subscriber(
            "estimated_pose", PoseStamped, self.est_pose_callback, queue_size=1
        )
        switch_sub_ = rospy.Subscriber(
            "switch", Bool, self.switch_callback, queue_size=1
        )
        self.output_pub_ = rospy.Publisher("vision_pose", PoseStamped, queue_size=1)

    def true_pose_callback(self, msg: PoseStamped) -> None:
        """
        Callback for the true pose of the drone.

        Args:
            msg (PoseStamped): The true pose of the drone.
        """
        if not (self.use_estimated_pose_):
            self.true_pose_ = msg
            self.output_pub_.publish(msg)

    def est_pose_callback(self, msg: PoseStamped) -> None:
        """
        Callback for the estimated pose of the drone.

        If the estimated pose is too far from the true pose, the estimated pose will
        not be used.

        Args:
            msg (PoseStamped): The estimated pose of the drone.
        """
        if self.measure_error(self.true_pose_, msg) > self.max_error_:
            rospy.logwarn("Estimated pose is too far from true pose")
            self.use_estimated_pose_ = False

        if self.use_estimated_pose_:
            self.est_pose = msg
            self.output_pub_.publish(msg)

    def switch_callback(self, msg: Bool) -> None:
        """
        Callback for the switch.

        Switches between using the estimated pose and the true pose.

        Args:
            msg (Bool): The switch message.
        """
        self.use_estimated_pose_ = msg.data
        if msg.data:
            rospy.loginfo("Using estimated pose")
        else:
            rospy.loginfo("Using true pose")

    def measure_error(self, true_pose: PoseStamped, est_pose: PoseStamped) -> float:
        """
        Measure the error between the true pose and the estimated pose.

        Args:
            true_pose (PoseStamped): The true pose of the drone.
            est_pose (PoseStamped): The estimated pose of the drone.

        Returns:
            float: The error between the true pose and the estimated pose.
        """
        transform = self.tf_buffer_.lookup_transform(
            self.global_frame_, true_pose.header.frame_id, rospy.Time(0)
        )
        true_pose = tf2_geometry_msgs.do_transform_pose(true_pose, transform)
        est_pose = tf2_geometry_msgs.do_transform_pose(est_pose, transform)

        return (
            (true_pose.pose.position.x - est_pose.pose.position.x) ** 2
            + (true_pose.pose.position.y - est_pose.pose.position.y) ** 2
            + (true_pose.pose.position.z - est_pose.pose.position.z) ** 2
        ) ** 0.5


if __name__ == "__main__":
    rospy.init_node("safety_switch", anonymous=True)
    ss = SafetySwitch()
    rospy.spin()
