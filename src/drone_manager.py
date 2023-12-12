#!/usr/bin/env python3

import re

import rospy
import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PoseStamped, PointStamped
from sesame_navigation.msg import PointId, PointIdArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from colorsys import hsv_to_rgb


class DroneManager:
    """
    This class is responsible for managing the obstacles detected by the vision stack.
    It receives the obstacles from one or more UAVs and publishes the obstacles that
    are active in the map frame."""

    def __init__(self):
        self.publish_rate_ = rospy.get_param("~publish_rate", 10.0)
        self.output_frame_ = rospy.get_param("~output_frame", "map")
        self.drone_to_publish_ = rospy.get_param("~drone_to_publish", "uav_1")
        self.disabled_drones_ = []

        self.pose_regex_ = rospy.get_param(
            "~pose_regex", "^/uav_[1-9].*/local_position/pose_stamped$"
        )
        self.color_map_ = eval(
            rospy.get_param(
                "~drones_color_map",
                "{\
                'uav_1': hsv_to_rgb(0.0, 1, 1),\
                'uav_2': hsv_to_rgb(0.5, 1, 1),\
                'uav_3': hsv_to_rgb(0.1, 1, 1),\
                'uav_4': hsv_to_rgb(0.6, 1, 1),\
                'uav_5': hsv_to_rgb(0.2, 1, 1),\
                'uav_6': hsv_to_rgb(0.7, 1, 1),\
                'uav_7': hsv_to_rgb(0.3, 1, 1),\
                'uav_8': hsv_to_rgb(0.8, 1, 1),\
                'uav_9': hsv_to_rgb(0.4, 1, 1),\
            }",
            )
        )
        self.marker_size_ = rospy.get_param("~marker_size", 0.25)
        self.use_visualization_ = rospy.get_param("~use_visualization", False)
        self.drones_to_track_ = eval(
            rospy.get_param(
                "~drones_to_track",
                "[\
                'uav_1',\
                'uav_2',\
                'uav_3',\
                'uav_4',\
                'uav_5',\
                'uav_6',\
                'uav_7',\
                'uav_8',\
                'uav_9',\
            ]",
            )
        )
        self.drones_id_map_ = eval(
            rospy.get_param(
                "~drones_id_map",
                "{\
                'uav_1': 1,\
                'uav_2': 2,\
                'uav_3': 3,\
                'uav_4': 4,\
                'uav_5': 5,\
                'uav_6': 6,\
                'uav_7': 7,\
                'uav_8': 8,\
                'uav_9': 9,\
            }",
            )
        )
        self.inv_drones_id_map_ = {v: k for k, v in self.drones_id_map_.items()}
        self.tracked_drones_ = {drone: None for drone in self.drones_to_track_}

        self.tf_buffer_ = tf2_ros.Buffer(rospy.Duration(100))
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_)

        rospy.loginfo("Waiting for pose topics to be published...")
        rospy.sleep(10.0)
        selected_topics = []
        topics = rospy.get_published_topics()
        for topic in topics:
            res = re.match(self.pose_regex_, topic[0])
            if res:
                selected_topics.append(topic[0])
                rospy.Subscriber(topic[0], PoseStamped, self.poseCallback)
        rospy.loginfo("Done!")
        rospy.loginfo("found topics: {}".format(selected_topics))

        self.drones_sub = rospy.Subscriber(
            "input_drones", PointIdArray, self.dronesCallback
        )
        self.disable_drone_sub = rospy.Subscriber(
            "disable_drone", String, self.disableDroneCallback
        )
        self.enable_drone_sub = rospy.Subscriber(
            "enable_drone", String, self.enableDroneCallback
        )
        self.request_drone_id_sub = rospy.Subscriber(
            "request_drone_id", String, self.requestDroneIdCallback
        )

        self.single_drone_point_pub_ = rospy.Publisher(
            "single_drone_point", PointStamped, queue_size=1
        )
        self.drones_pub_ = rospy.Publisher("output_drones", PointIdArray, queue_size=1)
        self.viz_pub_ = rospy.Publisher("visualization", MarkerArray, queue_size=1)

    def disableDroneCallback(self, msg: String) -> None:
        """
        Callback function for the disable_drone subscriber. It disables the drone
        with the given ID.

        Args:
            msg (Int32): Int32 message containing the ID of the drone to disable.
        """

        if not msg.data in self.drones_to_track_:
            rospy.logwarn("Drone {} not in list of drones to track".format(msg.data))
            return
        if not msg.data in self.disabled_drones_:
            self.disabled_drones_.append(msg.data)
            rospy.loginfo("Disabling drone {}".format(msg.data))
        else:
            rospy.logwarn("Drone {} already disabled".format(msg.data))

    def enableDroneCallback(self, msg: String) -> None:
        """
        Callback function for the enable_drone subscriber. It enables the drone
        with the given ID.

        Args:
            msg (Int32): Int32 message containing the ID of the drone to enable.
        """

        if not msg.data in self.drones_to_track_:
            rospy.logwarn("Drone {} not in list of drones to track".format(msg.data))
            return
        if msg.data in self.disabled_drones_:
            self.disabled_drones_.remove(msg.data)
            rospy.loginfo("Enabling drone {}".format(msg.data))
        else:
            rospy.logwarn("Drone {} already enabled".format(msg.data))

    def requestDroneIdCallback(self, msg: String) -> None:
        """
        Callback function for the request_drone_id subscriber. It tells the node
        which drone to output as a point stamped message.

        Args:
            msg (Int32): Int32 message containing the ID of the drone to disable.
        """

        if msg.data in self.drones_to_track_:
            self.drone_to_publish_ = msg.data
            rospy.loginfo("Now outputing the position of drone {}.".format(msg.data))
        else:
            rospy.logwarn(
                "Drone {} is not in the list of trackable drones. Ignoring.".format(
                    msg.data
                )
            )

    def publishSingleDronePoint(self) -> None:
        """
        Publishes the position of the drone as a point stamped message.
        """

        if (self.drone_to_publish_ in self.tracked_drones_.keys()) and (
            not self.tracked_drones_[self.drone_to_publish_] is None
        ):
            point = PointStamped()
            point.header.stamp = rospy.Time.now()
            point.header.frame_id = self.output_frame_
            point.point = self.tracked_drones_[self.drone_to_publish_]["position"]
            self.single_drone_point_pub_.publish(point)
        else:
            rospy.logwarn(
                "Requested position of {}, but no position is available yet for this target.".format(
                    self.drone_to_publish_
                )
            )

    def dronesCallback(self, msg: PointIdArray):
        """
        Callback function for the drones_sub subscriber. Receives the pose of the drones
        detected by the vision stack and updates the pose of the drones being tracked.

        Args:
            msg (PoseArray): PoseArray message containing the poses of the drones detected
                             by the vision stack.
        """

        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header

        transform = self.tf_buffer_.lookup_transform(
            self.output_frame_,
            msg.header.frame_id,
            msg.header.stamp,
            rospy.Duration(1.0),
        )

        for point_id in msg.points:
            pose_stamped.pose.position = point_id.position
            local_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)

            # Check if the drone is in the list of drones to track
            if point_id.id not in self.inv_drones_id_map_:
                continue
            drone = self.inv_drones_id_map_[point_id.id]
            # Update the position of the drone if the drone has never been seen / updated
            if self.tracked_drones_[drone] is None:
                self.tracked_drones_[drone] = {
                    "vision": True,
                    "position": local_pose.pose.position,
                    "timestamp": msg.header.stamp,
                }
            # If the drone has been seen before, check how long ago it was seen
            else:
                last_seen = self.tracked_drones_[drone]["timestamp"]
                time_diff = msg.header.stamp - last_seen
                # If the drone was seen using vision, update the position
                # Skip if the drone is disabled
                if (self.tracked_drones_[drone]["vision"]) and (
                    not drone in self.disabled_drones_
                ):
                    self.tracked_drones_[drone] = {
                        "vision": True,
                        "position": local_pose.pose.position,
                        "timestamp": msg.header.stamp,
                    }
                # If the last time the drone was seen is older than 1 second, update the position
                # Use the position from vision if the drone is disabled
                elif time_diff.to_sec() > 1.0:
                    self.tracked_drones_[drone] = {
                        "vision": True,
                        "position": local_pose.pose.position,
                        "timestamp": msg.header.stamp,
                    }
                else:
                    pass

    def poseCallback(self, msg: PoseStamped) -> None:
        """
        Callback function for the pose subscriber. It receives the pose of the UAV.

        Args:
            msg (PoseStamped): PoseStamped message containing the pose of the UAV.
        """

        transform = self.tf_buffer_.lookup_transform(
            self.output_frame_,
            msg.header.frame_id,
            rospy.Time(0),
            rospy.Duration(1.0),
        )

        local_pose = tf2_geometry_msgs.do_transform_pose(msg, transform)

        drone_name = msg.header.frame_id.split("/")[0]
        # Always update the position of the drone
        if (drone_name in self.drones_to_track_) and (
            not drone_name in self.disabled_drones_
        ):
            self.tracked_drones_[drone_name] = {
                "vision": False,
                "position": local_pose.pose.position,
                "timestamp": msg.header.stamp,
            }

    def publishVisualization(self) -> None:
        if self.use_visualization_:
            marker_array = MarkerArray()
            markers = []
            for drone, data in self.tracked_drones_.items():
                if data is not None:
                    marker = Marker()
                    marker.header.frame_id = self.output_frame_
                    marker.header.stamp = rospy.Time(0)
                    marker.ns = "managed_drones"
                    marker.id = self.drones_id_map_[drone] * 3
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position = data["position"]
                    marker.pose.position.z = 0
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = self.marker_size_
                    marker.scale.y = self.marker_size_
                    marker.scale.z = self.marker_size_
                    marker.color.a = 1.0
                    marker.color.r = self.color_map_[drone][0]
                    marker.color.g = self.color_map_[drone][1]
                    marker.color.b = self.color_map_[drone][2]
                    marker.lifetime = rospy.Duration(1)
                    markers.append(marker)
                    marker = Marker()
                    marker.header.frame_id = self.output_frame_
                    marker.header.stamp = rospy.Time(0)
                    marker.ns = "managed_drones"
                    marker.id = self.drones_id_map_[drone] * 3 + 1
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position = data["position"]
                    marker.pose.position.z = 0
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = self.marker_size_ * 1.25
                    marker.scale.y = self.marker_size_ * 1.25
                    marker.scale.z = self.marker_size_ * 1.25
                    marker.color.a = 0.333
                    marker.color.r = self.color_map_[drone][0]
                    marker.color.g = self.color_map_[drone][1]
                    marker.color.b = self.color_map_[drone][2]
                    marker.lifetime = rospy.Duration(1)
                    markers.append(marker)
                    marker = Marker()
                    marker.header.frame_id = self.output_frame_
                    marker.header.stamp = rospy.Time(0)
                    marker.ns = "managed_drones"
                    marker.id = self.drones_id_map_[drone] * 3 + 2
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position = data["position"]
                    marker.pose.position.z = 0
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = self.marker_size_ * 1.5
                    marker.scale.y = self.marker_size_ * 1.5
                    marker.scale.z = self.marker_size_ * 1.5
                    marker.color.a = 0.333
                    marker.color.r = self.color_map_[drone][0]
                    marker.color.g = self.color_map_[drone][1]
                    marker.color.b = self.color_map_[drone][2]
                    marker.lifetime = rospy.Duration(1)
                    markers.append(marker)
            marker_array.markers = markers
            self.viz_pub_.publish(marker_array)

    def run(self) -> None:
        """
        Publishes the known drone position at a fixed rate.
        """

        rate = rospy.Rate(self.publish_rate_)
        while not rospy.is_shutdown():
            drones = []
            for drone, data in self.tracked_drones_.items():
                if data is not None:
                    point_id = PointId()
                    point_id.id = self.drones_id_map_[drone]
                    point_id.position = data["position"]
                    drones.append(point_id)
            drones_msg = PointIdArray()
            drones_msg.header.stamp = rospy.Time.now()
            drones_msg.header.frame_id = self.output_frame_
            drones_msg.points = drones
            self.drones_pub_.publish(drones_msg)
            self.publishVisualization()
            self.publishSingleDronePoint()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("obstacle_manager", anonymous=True)
    OM = DroneManager()
    OM.run()
