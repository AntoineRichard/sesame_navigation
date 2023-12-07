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

        self.image_ = None
        self.camera_info_ = None
        self.pose_array_ = None
        self.obj_points_ = None
        self.pose_ = None

        self.instantiateMarkerSize()

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
                        local_point_id.id = ids[i]
                        self.pose_array.points.append(local_point_id)
            if self.pose_array.points:
                self.drone_pose_pub_.publish(self.pose_array)

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
        keep = np.sqrt(np.sum(p * p)) < 8.0  # param for thresholding obstacle detection
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
