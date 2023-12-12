#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import List, Tuple

import rospy
import cv2

from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from sesame_navigation.msg import PointId, PointIdArray
from visualization_msgs.msg import Marker, MarkerArray
from colorsys import hsv_to_rgb

import copy

import tf2_ros
import tf2_geometry_msgs

R = 6365842  # Radius of the earth in meters at this location
RAD = np.pi / 180.0  # Conversion from degrees to radians
DEG = 180.0 / np.pi  # Conversion from radians to degrees


class DetectArucoDrones:
    """
    This class is responsible for detecting the ArUco markers in the image from the
    camera. If a marker is detected, it computes the pose of the marker in the camera
    frame and transforms it to the local frame.
    """

    def __init__(self):
        self.body_frame_ = rospy.get_param("~body_frame", "uav_2/base_link")
        self.local_frame_ = rospy.get_param("~local_frame", "uav_2/odom")
        self.marker_size_ = rospy.get_param("~marker_size", 0.15)
        self.marker_id_ = rospy.get_param("~markers_id", [1, 2])
        self.aruco_size_ = rospy.get_param("~aruco_size", 5)

        self.use_visualization_ = rospy.get_param("~use_visualization", False)
        self.viz_marker_size_ = rospy.get_param("~viz_marker_size", 0.25)
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

        self.image_ = None
        self.camera_info_ = None
        self.pose_array_ = None
        self.obj_points_ = None
        self.pose_ = None

        self.instantiateMarkerSize()
        if self.aruco_size_ == 4:
            self.aruco_dict_ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        elif self.aruco_size_ == 5:
            self.aruco_dict_ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_params_ = cv2.aruco.DetectorParameters()
        self.detector_ = cv2.aruco_ArucoDetector(self.aruco_dict_, self.aruco_params_)

        self.bridge_ = CvBridge()

        self.tf_buffer_ = tf2_ros.Buffer(rospy.Duration(100.0))
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_)

        self.image_sub_ = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.imageCallback
        )

        self.camera_info_sub_ = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.cameraInfoCallback
        )

        self.drone_pose_pub_ = rospy.Publisher("drones", PointIdArray, queue_size=1)
        self.viz_pub_ = rospy.Publisher("drones_viz", MarkerArray, queue_size=1)

    def instantiateMarkerSize(self) -> None:
        """
        Instantiates the object points array using the marker size
        It tells OpenCV we want to measure the center of the marker.
        """

        self.obj_points_ = np.zeros((4, 1, 3), dtype=np.float32)
        self.obj_points_[0, 0] = np.array(
            [-self.marker_size_ / 2, self.marker_size_ / 2, 0]
        )
        self.obj_points_[1, 0] = np.array(
            [self.marker_size_ / 2, self.marker_size_ / 2, 0]
        )
        self.obj_points_[2, 0] = np.array(
            [self.marker_size_ / 2, -self.marker_size_ / 2, 0]
        )
        self.obj_points_[3, 0] = np.array(
            [-self.marker_size_ / 2, -self.marker_size_ / 2, 0]
        )

    def imageCallback(self, msg: Image) -> None:
        """
        Callback function for the image subscriber. It receives the image from the camera
        and detects the ArUco markers. If the marker is detected, it computes the pose of
        the marker in the camera frame and transforms it to the local frame.

        Args:
            msg (Image): Image message containing the image from the camera.
        """

        if self.camera_info_ is None:
            rospy.logwarn("No camera info received yet, skipping frame.")
            return

        self.image_ = msg
        rgb = self.bridge_.imgmsg_to_cv2(self.image_, "bgr8")
        (corners, ids, rejected) = self.detector_.detectMarkers(rgb)

        if len(corners) > 0:
            self.pose_array = PointIdArray()
            self.pose_array.header = copy.copy(self.image_.header)
            self.pose_array.header.frame_id = self.local_frame_
            for i, corner in enumerate(corners):
                if ids[i] in self.marker_id_:
                    cv_pose = self.getPoseFromCorners(corner)
                    local_point_id, keep = self.transformPoseToLocalFrame(cv_pose)
                    if keep:
                        local_point_id.id = ids[i][0]
                        self.pose_array.points.append(local_point_id)
            if self.pose_array.points:
                self.drone_pose_pub_.publish(self.pose_array)
                if self.use_visualization_:
                    self.visualizeDrones()

    def visualizeDrones(self) -> None:
        if self.pose_array.points:
            marker_array = MarkerArray()
            for i, point in enumerate(self.pose_array.points):
                marker = Marker()
                marker.header = copy.copy(self.image_.header)
                marker.header.frame_id = self.local_frame_
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position = point.position
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = self.viz_marker_size_
                marker.scale.y = self.viz_marker_size_
                marker.scale.z = self.viz_marker_size_
                marker.color.r = self.color_map_[self.inv_drones_id_map_[point.id]][0]
                marker.color.g = self.color_map_[self.inv_drones_id_map_[point.id]][1]
                marker.color.b = self.color_map_[self.inv_drones_id_map_[point.id]][2]
                marker.color.a = 1.0
                marker.lifetime = rospy.Duration(1.0)
                marker_array.markers.append(marker)
            self.viz_pub_.publish(marker_array)

    def getPoseFromCorners(
        self, corners: List[List[float]]
    ) -> Tuple[float, List[float], List[float]]:
        """
        Computes the pose of the marker in the camera frame.

        Args:
            corners (np.ndarray): Numpy array containing the corners of the marker.

        Returns:
            float: retval.
            List[float]: Rotation vector.
            List[float]: Translation vector.
        """

        out = cv2.solvePnP(
            self.obj_points_,
            corners[0],
            self.camera_matrix_,
            self.dist_coeffs_,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        return out

    def transformPoseToLocalFrame(self, out: list) -> Tuple[PointId, bool]:
        """
        Transforms the pose of the marker in the camera frame to the local frame.
        Checks if the marker is within a given distance threshold from the UAV.

        Args:
            out (list): List containing the pose of the marker in the camera frame.

        Returns:
            PoseStamped: Pose of the marker in the local frame.
            bool: True if the marker is within the distance threshold, False otherwise.
        """

        pose = PoseStamped()
        pose.header = self.image_.header
        pose.pose.position.x = out[2][0]
        pose.pose.position.y = out[2][1]
        pose.pose.position.z = out[2][2]
        pose.pose.orientation.w = 1.0
        # Pose of the object in the body frame
        transform = self.tf_buffer_.lookup_transform(
            self.local_frame_,
            self.image_.header.frame_id,
            self.image_.header.stamp,
            rospy.Duration(1.0),
        )

        pose_local = tf2_geometry_msgs.do_transform_pose(pose, transform)

        pose_id = PointId()
        pose_id.position = pose_local.pose.position

        p = np.array(
            [
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            ]
        )
        keep = (
            np.sqrt(np.sum(p * p)) < 12.0
        )  # param for thresholding obstacle detection
        return pose_id, keep

    def cameraInfoCallback(self, msg: CameraInfo) -> None:
        """
        Callback function for the camera info subscriber. It receives the camera info
        and computes the camera matrix and the distortion coefficients.

        Args:
            msg (CameraInfo): Camera info message containing the camera matrix and the
                              distortion coefficients.
        """

        self.camera_info_ = msg
        self.camera_matrix_ = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs_ = np.array(msg.D)


if __name__ == "__main__":
    rospy.init_node("detect_aruco_drones", anonymous=True)
    detect_aruco_drones = DetectArucoDrones()
    rospy.spin()
