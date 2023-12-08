#!/usr/bin/env python3

import pandas as pd
import numpy as np
import threading
import copy

import rospy
import cv2

from geographic_msgs.msg import GeoPoseStamped, GeoPointStamped
from std_msgs.msg import Bool

from mavros_msgs.srv import (
    CommandBool,
    CommandBoolRequest,
    SetMode,
    SetModeRequest,
    CommandTOL,
    CommandTOLRequest,
)
from mavros_msgs.msg import State, Altitude
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import (
    PoseStamped,
    TwistStamped,
    PoseArray,
    Quaternion,
    PointStamped,
)

from sprayer_path_planner.msg import PointId, PointIdArray

import tf2_ros
import tf2_geometry_msgs

R = 6365842  # Radius of the earth in meters at this location
RAD = np.pi / 180.0  # Conversion from degrees to radians
DEG = 180.0 / np.pi  # Conversion from radians to degrees


def quat2yaw(q: Quaternion) -> float:
    """
    Converts a quaternion to a yaw angle.

    Args:
        q (Quaternion): Quaternion to be converted.

    Returns:
        float: Yaw angle of the quaternion.
    """

    return np.arctan2(
        2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
    )  # yaw (z-axis rotation)


class Controller:
    """
    Exponential smoothing to provide more natural control outputs.
    """

    def __init__(self, k1: float, sat: float = None):
        assert k1 < 1.0, "k1 must be less than 1.0"
        assert k1 > 0.0, "k1 must be greater than 0.0"

        self.k1 = k1
        self.k2 = 1 - k1
        self.sat = sat
        self.u = 0

    def control(self, d: float) -> float:
        """
        Exponential smoothing control law.

        Args:
            d (float): Error signal.

        Returns:
            float: Control output.
        """

        d = self.saturate(d)
        self.u = self.k1 * d + self.k2 * self.u
        self.u = self.saturate(self.u)
        return self.u

    def saturate(self, u: float) -> float:
        """
        Saturates the control output if a saturation value is provided.

        Args:
            u (float): Control output.

        Returns:
            float: Saturated control output.
        """

        if self.sat is not None:
            u = np.sign(u) * np.min(np.abs(u), self.sat)
        return u


class SprayingUAVPlanner:
    def __init__(self):
        # Heightmap parameters
        self.heightmap_path_ = rospy.get_param(
            "heightmap_dem_path",
            "/home/antoine/Digital_Vineyard_Rosenberg/20230914_RemichRosenberg_DEM_10cm_safety.tif",
        )
        self.heightmap_mpp_ = rospy.get_param("~heightmap_mpp", 0.1)
        self.heightmap_min_altitude_ = rospy.get_param("~heightmap_min_altitude", 150.0)
        long = rospy.get_param("~heightmap_origin_long", 49.5381173342964)
        lat = rospy.get_param("~heightmap_origin_lat", 6.352911227620768)
        self.heightmap_bottom_left_corner_long_lat_ = [long, lat]
        self.local_frame_ = rospy.get_param("~local_frame", "uav_2/odom")
        self.tracked_drone = rospy.get_param("~tracked_drone", 1)

        # Planner parameters
        self.target_dist_ = rospy.get_param("~target_distance", 7.0)
        self.max_target_dist_ = rospy.get_param("~max_target_distance", 10.0)
        self.camera_angle_ = rospy.get_param("~camera_angle", 25.0 * np.pi / 180)
        self.travel_dist_ = rospy.get_param("~travel_distance", 1.0)
        self.w_goal_ = rospy.get_param("~goal_weight", 5.0)
        self.w_obstacle_ = rospy.get_param("~obstacle_weight", -7.5)
        self.altitude_offset_ = rospy.get_param("~altitude_offset", 0.5)

        # Controllers
        self.use_position_control_ = rospy.get_param("~use_position_control", True)

        # Offsets and home coordinates
        self.offset_ = None
        self.offset_hmap_ = None
        self.zero_llh = np.array([0, 0, 0])

        # Shared variables
        self.global_position_ = None
        self.obstacles_position_ = np.array([])
        self.obstacles_radius_ = np.array([])
        self.heightmap_ = None
        self.altitude_ = None
        self.local_pose_ = None
        self.drone_state_ = State()
        self.lock_ = threading.Lock()

        # Tfs handlers
        self.tf_buffer_ = tf2_ros.Buffer(rospy.Duration(100.0))
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_)

        self.pose_pub = rospy.Publisher(
            "/mavros/setpoint_position/local", PoseStamped, queue_size=1
        )
        self.vel_pub = rospy.Publisher(
            "/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1
        )
        self.set_origin = rospy.Publisher(
            "/mavros/global_position/gp_origin",
            GeoPointStamped,
            queue_size=1,
            latch=True,
        )

        self.global_position_sub = rospy.Subscriber(
            "/mavros/global_position/global",
            NavSatFix,
            self.globalPositionCallback,
        )
        self.altitude_sub = rospy.Subscriber(
            "/mavros/altitude", Altitude, self.altitudeCallback
        )
        self.local_pose_sub = rospy.Subscriber(
            "/mavros/local_position/pose", PoseStamped, self.localPoseCallback
        )
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.stateCallback)
        self.obstacles_sub = rospy.Subscriber(
            "obstacles", PoseArray, self.obstaclesCallback
        )

        self.drones_state_sub = rospy.Subscriber(
            "drones", PointIdArray, self.dronesCallback
        )

        self.buildHeightmap()

    # =============================================================================
    #        Services and basic functions
    # =============================================================================

    def initializeUAV(self) -> None:
        """
        Sets the origin of the global frame to the current position of the drone.
        It removes the anoying warning message that appears when the origin is no set
        in the simulation.
        It is unclear if this would be needed for the real drone.
        """

        self.set_origin.publish(GeoPointStamped())
        self.setOffset()

    def armUAV(self) -> None:
        """
        Arms the UAV.
        """

        rospy.wait_for_service("/uav_2/mavros/cmd/arming")
        arm = rospy.ServiceProxy("/uav_2/mavros/cmd/arming", CommandBool)
        arm_req = CommandBoolRequest()
        arm_req.value = True
        arm(arm_req)
        rospy.loginfo("Armed UAV")

    def disarmUAV(self) -> None:
        """
        Disarms the UAV.
        """

        rospy.wait_for_service("/uav_2/mavros/cmd/arming")
        arm = rospy.ServiceProxy("/uav_2/mavros/cmd/arming", CommandBool)
        arm_req = CommandBoolRequest()
        arm_req.value = False
        arm(arm_req)
        rospy.loginfo("Disarmed UAV")

    def setOffboardMode(self) -> None:
        """
        Sets the UAV in offboard mode.
        It allows us to send position and velocity commands to the UAV.
        """

        rospy.wait_for_service("/uav_2/mavros/set_mode")
        mode = rospy.ServiceProxy("/uav_2/mavros/set_mode", SetMode)
        mode_req = SetModeRequest()
        mode_req.custom_mode = "OFFBOARD"
        mode(mode_req)
        rospy.loginfo("Set OFFBOARD mode")

    def requestLanding(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        """
        Requests the UAV to land at the specified coordinates.
        The coordinates are in the UAV's local frame.

        Args:
            x (float, optional): x coordinate of the landing position. Defaults to 0.
            y (float, optional): y coordinate of the landing position. Defaults to 0.
            z (float, optional): z coordinate of the landing position. Defaults to 0.
        """

        rospy.wait_for_service("/uav_2/mavros/cmd/land")
        land = rospy.ServiceProxy("/uav_2/mavros/cmd/land", CommandTOL)
        land_req = CommandTOLRequest()
        land_req.latitude = x
        land_req.longitude = y
        land_req.altitude = z
        land(land_req)
        rospy.loginfo("Landing")

    def landAtHome(self, height: float = 4.0) -> None:
        """
        Lands the UAV at the home position.

        Args:
            height (float, optional): Height the UAV should reach before landing.
                                      Defaults to 4.0 meters above it's home position.
        """

        rate = rospy.Rate(20)
        alt_reached = False
        pose = PoseStamped()
        pose.pose.position.z = height
        rospy.loginfo("Going back home")
        while (not rospy.is_shutdown()) and (alt_reached == False):
            self.pose_pub.publish(pose)
            xyz_uav = np.array(
                [
                    self.local_pose_.pose.position.x,
                    self.local_pose_.pose.position.y,
                    self.local_pose_.pose.position.z,
                ]
            )
            alt_reached = np.linalg.norm(xyz_uav - np.array([0, 0, height])) < 0.25
            rate.sleep()
        self.requestLanding()

    def takeOff(self, height: float = 4.0) -> None:
        """
        Takes off to the specified height.

        Args:
            height (float, optional): Height the UAV should reach. Defaults to 4.0.
        """

        rate = rospy.Rate(20)
        alt_reached = False
        pose = PoseStamped()
        pose.pose.position.z = height
        rospy.loginfo("Taking off")
        while (not rospy.is_shutdown()) and (alt_reached == False):
            self.pose_pub.publish(pose)
            alt_reached = np.abs(self.local_pose_.pose.position.z - height) < 0.25
            rate.sleep()
        rospy.loginfo("Take off complete")

    # =============================================================================
    #        Callback functions
    # =============================================================================

    def globalPositionCallback(self, msg: NavSatFix) -> None:
        """
        Callback function for the global position subscriber.

        Args:
            msg (NavSatFix): NavSatFix message containing the global position of the UAV.
        """

        self.global_position_ = msg

    def localPoseCallback(self, msg: PoseStamped) -> None:
        """
        Callback function for the local pose subscriber.

        Args:
            msg (PoseStamped): PoseStamped message containing the pose of the UAV.
        """

        self.local_pose_ = msg

    def stateCallback(self, msg: State) -> None:
        """
        Callback function for the state subscriber.

        Args:
            msg (State): State message containing the state of the UAV.
        """

        self.drone_state_ = msg

    def altitudeCallback(self, msg: Altitude) -> None:
        """
        Callback function for the altitude subscriber.

        Args:
            msg (Altitude): Altitude message containing the altitude of the UAV.
        """

        self.altitude_ = msg.amsl

    def obstaclesCallback(self, msg: PoseArray) -> None:
        """
        Callback function for the obstacle subscriber.

        Args:
            msg (PoseArray): PoseArray message containing the position and radius of the
                             detected obstacles.
        """
        obstacles = []
        radius = []

        # Project the obstacles position into the uav local frame
        if msg.poses:
            pose = PoseStamped()
            pose.header = msg.header
            for obstacle in msg.poses:
                pose.pose = obstacle
                transform = self.tf_buffer_.lookup_transform(
                    self.local_frame_,
                    pose.header.frame_id,
                    pose.header.stamp,
                    rospy.Duration(1.0),
                )

                obstacle_local = tf2_geometry_msgs.do_transform_pose(pose, transform)

                # We avoid obstacles in a 2D plane, z is not needed.
                xyz = np.array(
                    [
                        obstacle_local.pose.position.x,
                        obstacle_local.pose.position.y,
                        0,
                    ]
                )
                # We use z as a proxy to store the radius of the obstacles.
                # We use a mutex to prevent the code from accessing the variables
                # if they are being updated.
                obstacles.append(xyz)
                radius.append(obstacle.position.z)

        self.lock_.acquire()
        self.obstacles_position_ = np.array(obstacles)
        self.obstacles_radius_ = np.array(radius)
        self.lock_.release()

    def dronesCallback(self, msg: PointIdArray) -> None:
        if msg.points:
            pose = PoseStamped()
            pose.header = msg.header
            transform = self.tf_buffer_.lookup_transform(
                self.local_frame_,
                pose.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            for point_id in msg.points:
                if point_id.id == self.tracked_drone:
                    pose.pose.position = point_id.position
                    local_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
                    self.target_position_ = local_pose.pose.position

    # =============================================================================
    #        Waypoints & Heightmap
    # =============================================================================

    def readHeightMap(self) -> None:
        """
        Reads the heightmap image.
        """

        self.heightmap_ = cv2.imread(self.heightmap_path_, -1)

    def setOffset(self) -> None:
        """
        Sets the offset between the local drone frame and the global frame.
        To do so, it relies on the current gps coordinates and altitude of the drone.
        """

        if (self.global_position_ is not None) and (self.altitude_ is not None):
            rospy.loginfo("Global position and altitude aquired, setting offset.")
            self.offset_ = self.castLLHToXYZ(
                self.global_position_.latitude,
                self.global_position_.longitude,
                self.altitude_,
            )
        else:
            rospy.logwarn("Global position or altitude not available")
            rospy.logwarn("Waiting for global position and altitude")
            rospy.sleep(1)
            if not rospy.is_shutdown():
                self.setOffset()

    def getHMOffset(self) -> None:
        """
        Sets the offset between the Heightmap and the global frame.
        """

        self.offset_hmap_ = self.castLLHToXYZ(
            self.heightmap_bottom_left_corner_long_lat_[0],
            self.heightmap_bottom_left_corner_long_lat_[1],
            self.heightmap_min_altitude_,
        )
        self.offset_hmap_[:2] = self.offset_hmap_[:2] - self.offset_[:2]
        self.heightmap_ -= (
            self.heightmap_min_altitude_
            - self.offset_hmap_[2]
            + self.offset_[2]
            - self.altitude_offset_
        )

    def getAltitudeFromHeightMap(self, x: float, y: float) -> float:
        try:
            x = int((x - self.offset_hmap_[0]) / self.heightmap_mpp_)
            y = int((y - self.offset_hmap_[1]) / self.heightmap_mpp_)
            return self.heightmap_[-y, x]
        except:
            return np.max(self.heightmap_)

    def buildHeightmap(self) -> None:
        """
        Builds the heightmap.
        """

        # Convert to local coordinates (instead of GPS coordinates)
        self.initializeUAV()
        # Get the Heightmap and apply the offsets
        self.readHeightMap()
        self.getHMOffset()

    # =============================================================================
    #        Castings and conversions
    # =============================================================================

    def castLLHToXYZ(self, lat: float, long: float, alt: float) -> np.ndarray:
        """
        Casts the gps coordinates to local coordinates.

        Args:
            lat (float): Latitude of the point.
            long (float): Longitude of the point.
            alt (float): Altitude of             rate.sleep()the point.

        Returns:
            np.ndarray: Numpy array containing the local coordinates of the point."""

        dlat = lat - self.zero_llh[0]
        dlong = long - self.zero_llh[1]
        dalt = alt - self.zero_llh[2]

        xyz = np.zeros(3)
        xyz[0] = R * (dlong * RAD) * np.cos(self.zero_llh[0] * RAD)
        xyz[1] = R * dlat * RAD
        xyz[2] = dalt
        return xyz

    # =============================================================================
    #        Obstacle avoidance & Planner
    # =============================================================================

    def compute_obstacles_cost(self, drone_position):
        """
        Computes the cost of the obstacle.
        """
        if self.obstacles_position_.shape[0] == 0:
            # If there is no obstacle, set the penalties to 0
            dx = dy = 0
        else:
            # Computes the cost of going through obstacles
            d = (drone_position - self.obstacles_position_)[:, :2]
            norm = np.sqrt(np.sum(d * d, axis=1))
            theta = np.arctan2(d[:, 1], d[:, 0])
            in_rad = norm < self.obstacles_radius_
            in_2rad = np.invert(in_rad) * (norm < (self.obstacles_radius_ * 2))
            # Take full penalty if inside the obstacle
            dx = np.sum(in_rad * np.cos(theta))
            dy = np.sum(in_rad * np.sin(theta))
            n = np.sqrt(dx * dx + dy * dy)
            if n:
                dx = dx * self.w_obstacle_ / n
                dy = dy * self.w_obstacle_ / n
            # Take 1/3 penalty if inside the 2*radius with linear decrease
            dx2 = np.sum(
                in_2rad
                * np.cos(theta)
                * ((2 * self.obstacles_radius_ - norm) / self.obstacles_radius_)
            )
            dy2 = np.sum(
                in_2rad
                * np.sin(theta)
                * ((2 * self.obstacles_radius_ - norm) / self.obstacles_radius_)
            )
            n2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
            if n2:
                dx2 = dx2 * self.w_obstacle_ / n2 / 3
                dy2 = dy2 * self.w_obstacle_ / n2 / 3
            # r = (norm - self.obstacles_radius_) / self.obstacles_radius_
            # r = np.square(np.tanh(r * 3))
            # dx2 = np.sum(in_2rad * np.cos(theta) * r)
            # dy2 = np.sum(in_2rad * np.sin(theta) * r)
            # n2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
            # if n2:
            #    dx2 = dx2 * self.w_obstacle_ / n2  # / 3
            #    dy2 = dy2 * self.w_obstacle_ / n2  # / 3
            dx += dx2
            dy += dy2
        return dx, dy

    def followPosition(self):
        # WP coordinates
        xyz_target = np.array(
            [
                self.target_position_.x,
                self.target_position_.y,
                self.target_position_.z,
            ]
        )
        # UAV coordinates
        xyz_uav = np.array(
            [
                self.local_pose_.pose.position.x,
                self.local_pose_.pose.position.y,
                self.local_pose_.pose.position.z,
            ]
        )
        # Compute distance and angle
        d = xyz_uav - xyz_target
        norm = np.sqrt(np.sum(d * d))
        theta = np.arctan2(d[1], d[0])
        desired_z = self.target_position_.z + norm * np.tan(self.camera_angle_)
        error = norm - self.target_dist_
        n_error = np.abs(error)
        # Compute the distance to travel
        if n_error < self.travel_dist_:
            travel_dist = n_error
        else:
            travel_dist = self.travel_dist_

        # Computes the weight of the goal
        e = np.min([error, self.max_target_dist_])
        e = np.max([e, -self.max_target_dist_])
        w_goal = e * e * e * self.w_goal_
        w_goal_x = travel_dist * np.cos(theta) * w_goal
        w_goal_y = travel_dist * np.sin(theta) * w_goal
        # Computes the weight of the obstacles
        w_obstacle_x = 0
        w_obstacle_y = 0
        w_obstacle_x, w_obstacle_y = self.compute_obstacles_cost(xyz_uav)
        # Computes the final weight
        dx = w_goal_x + w_obstacle_x
        dy = w_goal_y + w_obstacle_y
        # Get the direction of the travel
        norm = np.sqrt(dx * dx + dy * dy)
        dx = dx / norm
        dy = dy / norm
        # Compute the new position
        x = xyz_uav[0] - dx * travel_dist
        y = xyz_uav[1] - dy * travel_dist
        # Compute the yaw
        yaw = theta + np.pi  # np.arctan2(dy, dx) + np.pi
        # make the pose message
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = np.max([self.getAltitudeFromHeightMap(x, y), desired_z])
        pose.pose.orientation.x = 0
        pose.pose.orientation.y = 0
        pose.pose.orientation.z = np.sin(yaw * 0.5)
        pose.pose.orientation.w = np.cos(yaw * 0.5)

        return pose

    def followUAV(self):
        rate = rospy.Rate(20)
        self.armUAV()
        for i in range(100):
            self.pose_pub.publish(PoseStamped())
            rate.sleep()
        self.setOffboardMode()
        self.takeOff()
        rospy.loginfo(
            "Tracking UAV %d at coordinates %.2f, %.2f"
            % (
                self.tracked_drone,
                self.target_position_.x,
                self.target_position_.y,
            )
        )
        while not rospy.is_shutdown():
            self.lock_.acquire()
            # print(self.followPosition())
            self.pose_pub.publish(self.followPosition())
            self.lock_.release()
            rate.sleep()
        self.landAtHome()


if __name__ == "__main__":
    rospy.init_node("waypoint_planner", anonymous=True)
    SUP = SprayingUAVPlanner()
    SUP.followUAV()
