#!/usr/bin/env python3

import numpy as np
import copy

import rospy
import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray


class ObstacleManager:
    """
    This class is responsible for managing the obstacles detected by the vision stack.
    It receives the obstacles from one or more UAVs and publishes the obstacles that
    are active in the map frame."""

    def __init__(self):
        self.obstacle_radius_ = rospy.get_param("~obstacle_radius", 2.5)
        self.obstacle_epsilon_ = rospy.get_param("~obstacle_epsilon", 0.25)
        self.output_frame_ = rospy.get_param("~output_frame", "map")
        self.marker_color_ = rospy.get_param("~marker_color", [1.0, 0.0, 0.0])
        self.publish_visualization_ = rospy.get_param("~publish_visualization", False)

        self.observed_obstacles_ = None
        self.active_obstacles_ = None

        self.tf_buffer_ = tf2_ros.Buffer(rospy.Duration(100))
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_)

        self.obstacles_sub = rospy.Subscriber(
            "input_obstacles", PoseArray, self.obstaclesCallback
        )

        self.obstacles_pub = rospy.Publisher(
            "output_obstacles", PoseArray, queue_size=1
        )

        self.obstacles_viz_pub_ = rospy.Publisher(
            "obstacles_viz", MarkerArray, queue_size=1
        )

    def projectToFrame(self, pose_array: PoseArray) -> np.ndarray:
        """
        Projects the poses from the PoseArray message to the frame specified as output_frame_.
        From these poses, it extracts the xyz coordinates and returns them as a numpy array.

        Args:
            pose_array (PoseArray): PoseArray message containing the poses to be projected.

        Returns:
            np.ndarray: Numpy array containing the xyz coordinates of the poses projected
                        to the output_frame_.
        """
        pose_stamped = PoseStamped()
        pose_stamped.header = pose_array.header

        transform = self.tf_buffer_.lookup_transform(
            self.output_frame_,
            pose_array.header.frame_id,
            pose_array.header.stamp,
            rospy.Duration(1.0),
        )

        xyz_coords = []
        for pose in pose_array.poses:
            pose_stamped.pose = pose
            local_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
            xyz_coords.append(
                [
                    local_pose.pose.position.x,
                    local_pose.pose.position.y,
                    local_pose.pose.position.z,
                ]
            )
        return np.array(xyz_coords)

    def check4Updates(self, xyz_coords: np.ndarray) -> np.ndarray:
        """
        Checks if the obstacles detected by the vision stack are new obstacles or
        if they are already being tracked. If they are new obstacles, they are
        added to the list of observed obstacles.

        Args:
            xyz_coords (np.ndarray): Numpy array containing the xyz coordinates of the
                                     obstacles detected by the vision stack.

        Returns:
            np.ndarray: Numpy array containing the xyz coordinates of the new obstacles.
        """

        new_obstacles = []
        if self.observed_obstacles_ is None:
            new_obstacles = xyz_coords
            self.observed_obstacles_ = copy.copy(new_obstacles)
        elif xyz_coords.size == 0:
            new_obstacles = np.array([])
        else:
            for xyz_coord in xyz_coords:
                # We are discarding the z coordinate as we are solving
                # this problem in the xy plane
                d = self.observed_obstacles_ - xyz_coord
                norm = np.linalg.norm(d[:, :2], axis=1)
                if (
                    sum(norm > self.obstacle_radius_)
                    == self.observed_obstacles_.shape[0]
                ):
                    new_obstacles.append(xyz_coord)
            new_obstacles = np.array(new_obstacles)
            if new_obstacles.size != 0:
                new_obstacles = np.array(new_obstacles)
                self.observed_obstacles_ = np.concatenate(
                    [self.observed_obstacles_, new_obstacles], axis=0
                )
        return new_obstacles

    def aggregate(self, dct: dict, new: int, parent: list = []) -> list:
        """
        Recursive function that returns the ids of the obstacles that are connected.
        The dictionary is structured as follows:
        {0: [1, 2, 3], 1: [0, 2], 2: [0, 1], 3: [0]}
        This means that obstacle 0 is connected to obstacles 1, 2 and 3.
        Obstacle 1 is connected to obstacles 0 and 2.
        Obstacle 2 is connected to obstacles 0 and 1.
        Obstacle 3 is connected to obstacle 0.

        Args:
            dct (dict): Dictionary containing the ids of the obstacles that are connected.
            new (int): Id of the new obstacle.
            parent (list): List containing the ids of the obstacles that are connected.

        Returns:
            list: List containing the ids of the obstacles that are connected.
        """

        childs = dct[new]
        parent.append(new)
        if childs:
            for child in childs:
                if not (child in parent):
                    self.aggregate(dct, child, parent)
        return parent

    def addObstacles(self, new_obstacles: np.ndarray) -> None:
        """
        Adds the new obstacles to the list of observed obstacles and
        aggregates the obstacles that are too close for the UAVs to
        pass between them.

        Args:
            new_obstacles (np.ndarray): Numpy array containing the xyz coordinates of the
                                        new obstacles.
        """
        if new_obstacles.size != 0:
            self.active_obstacles_ = []
            ids = {k: [] for k in range(len(self.observed_obstacles_))}
            for i, obs_obs_a in enumerate(self.observed_obstacles_):
                for j, obs_obs_b in enumerate(self.observed_obstacles_):
                    if i != j:
                        d = obs_obs_a - obs_obs_b
                        d[2] = 0
                        norm = np.linalg.norm(d)
                        if norm < self.obstacle_radius_ * (2 + self.obstacle_epsilon_):
                            ids[i].append(j)
                            ids[j].append(i)
            ids_aggregated = []
            for i in ids.keys():
                ids_aggregated.append(self.aggregate(ids, i, parent=[]))
            ids_aggregated = list(set([tuple(sorted(i)) for i in ids_aggregated]))
            for ids in ids_aggregated:
                selected_obstacles = [self.observed_obstacles_[j] for j in ids]
                center = np.mean(selected_obstacles, axis=0)
                radius = (
                    np.linalg.norm(center - selected_obstacles, axis=1).max()
                    + self.obstacle_radius_
                )
                self.active_obstacles_.append([center, radius])

    def publishObstacles(self) -> None:
        """
        Publishes the active obstacles in the map frame.
        """

        if not self.active_obstacles_ is None:
            pose_array = PoseArray()
            pose_array.header.frame_id = self.output_frame_
            pose_array.header.stamp = rospy.Time.now()
            for obstacle in self.active_obstacles_:
                pose = Pose()
                pose.position.x = obstacle[0][0]  # x
                pose.position.y = obstacle[0][1]  # y
                pose.position.z = obstacle[1]  # radius
                pose.orientation.w = 1.0
                pose_array.poses.append(pose)
            self.obstacles_pub.publish(pose_array)

    def obstaclesCallback(self, msg: PoseArray) -> None:
        """
        Callback function that receives the obstacles detected by the vision stack.
        It projects the obstacles to the map frame, checks if they are new obstacles
        and publishes the active obstacles.

        Args:
            msg (PoseArray): PoseArray message containing the obstacles detected by the
                             vision stack.
        """

        xyz_coords = self.projectToFrame(msg)
        new_obstacles = self.check4Updates(xyz_coords)
        self.addObstacles(new_obstacles)
        if new_obstacles.size != 0:
            self.publishObstacles()
            if self.publish_visualization_:
                self.publishVisualization()

    def publishVisualization(self) -> None:
        """
        Publishes the visualization of the obstacles.
        """

        marker_array = MarkerArray()
        for i, obstacle in enumerate(self.active_obstacles_):
            marker = Marker()
            marker.header.frame_id = self.output_frame_
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = obstacle[0][0]
            marker.pose.position.y = obstacle[0][1]
            marker.pose.position.z = 0
            marker.pose.orientation.w = 1.0
            marker.scale.x = obstacle[1] * 2
            marker.scale.y = obstacle[1] * 2
            marker.scale.z = obstacle[1] * 2
            marker.color.r = self.marker_color_[0]
            marker.color.g = self.marker_color_[1]
            marker.color.b = self.marker_color_[2]
            marker.color.a = 0.25
            marker_array.markers.append(marker)
        self.obstacles_viz_pub_.publish(marker_array)


if __name__ == "__main__":
    rospy.init_node("obstacle_manager", anonymous=True)
    OM = ObstacleManager()
    rospy.spin()
