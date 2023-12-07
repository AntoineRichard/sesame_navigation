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
from geometry_msgs.msg import PoseStamped, TwistStamped, PoseArray, Quaternion, Point

from visualization_msgs.msg import Marker, MarkerArray

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
        self.local_frame_ = rospy.get_param("~local_frame", "uav_1/odom")

        # Planner parameters
        self.csv_data_path_ = rospy.get_param(
            "~waypoints_csv_path", "/home/antoine/Digital_Vineyard_Rosenberg/vines.csv"
        )
        self.d_threshold_ = rospy.get_param("~waypoint_distance_threshold", 0.25)
        self.travel_dist_ = rospy.get_param("~travel_distance", 1.0)
        self.w_goal_ = rospy.get_param("~goal_weight", 5.0)
        self.w_obstacle_ = rospy.get_param("~obstacle_weight", -7.5)
        self.altitude_offset_ = rospy.get_param("~altitude_offset", 0.5)

        # Controllers
        self.use_position_control_ = rospy.get_param("~use_position_control", True)
        if self.use_position_control_:
            self.x_controller_ = Controller(0.75)
            self.y_controller_ = Controller(0.75)
            self.z_controller_ = Controller(0.75)
            self.yaw_controller_ = Controller(0.75)
            self.goToWayPoint = self.goToWayPointPositionControl
        else:
            self.vlin_max_ = rospy.get_param("~maximum_linear_velocity", 1.0)
            self.vang_max_ = rospy.get_param("~maximum_angular_velocity", 0.5)
            self.x_controller_ = Controller(0.75, self.vlin_max_)
            self.y_controller_ = Controller(0.75, self.vlin_max_)
            self.z_controller_ = Controller(0.75, self.vlin_max_)
            self.yaw_controller_ = Controller(0.75, self.vang_max_)
            self.goToWayPoint = self.goToWayPointVelocityControl

        # Visualization parameters
        self.publish_visualization_ = rospy.get_param("~publish_visualization", False)
        self.waypoints_color_ = rospy.get_param("~waypoints_color", [0.11, 0.43, 0.92])
        self.waypoints_color_visited_ = rospy.get_param(
            "~waypoints_color_visited", [0.0, 1.0, 0.4]
        )
        self.waypoints_color_current_ = rospy.get_param(
            "~waypoints_color_current", [0.808, 0.0, 1.0]
        )
        self.waypoints_color_unreachable_ = rospy.get_param(
            "~waypoints_color_unreachable", [1.0, 0.0, 0.0]
        )
        self.waypoints_size_ = rospy.get_param("~waypoints_size", 0.25)
        self.done_ids_ = []
        self.unreachable_ids_ = []
        self.current_id_ = 0
        self.waypoints_points_ = []

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
        self.waypoints_viz_pub_ = rospy.Publisher(
            "waypoints_viz", MarkerArray, queue_size=1
        )

        self.buildWayPoints()

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

        rospy.wait_for_service("/uav_1/mavros/cmd/arming")
        arm = rospy.ServiceProxy("/uav_1/mavros/cmd/arming", CommandBool)
        arm_req = CommandBoolRequest()
        arm_req.value = True
        arm(arm_req)
        rospy.loginfo("Armed UAV")

    def disarmUAV(self) -> None:
        """
        Disarms the UAV.
        """

        rospy.wait_for_service("/uav_1/mavros/cmd/arming")
        arm = rospy.ServiceProxy("/uav_1/mavros/cmd/arming", CommandBool)
        arm_req = CommandBoolRequest()
        arm_req.value = False
        arm(arm_req)
        rospy.loginfo("Disarmed UAV")

    def setOffboardMode(self) -> None:
        """
        Sets the UAV in offboard mode.
        It allows us to send position and velocity commands to the UAV.
        """

        rospy.wait_for_service("/uav_1/mavros/set_mode")
        mode = rospy.ServiceProxy("/uav_1/mavros/set_mode", SetMode)
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

        rospy.wait_for_service("/uav_1/mavros/cmd/land")
        land = rospy.ServiceProxy("/uav_1/mavros/cmd/land", CommandTOL)
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

    # =============================================================================
    #        Waypoints & Heightmap
    # =============================================================================

    def readCSV(self) -> None:
        """
        Reads the csv file containing the coordinates of the plants to be treated.
        """

        self.csv_data_ = pd.read_csv(self.csv_data_path_)

    def readHeightMap(self) -> None:
        """
        Reads the heightmap image.
        """

        self.heightmap_ = cv2.imread(self.heightmap_path_, -1)

    def getCoordsFromCSVData(self) -> None:
        """
        Reads the csv file and extracts the coordinates of the plants.
        It also sorts the plants in the order they should be visited.
        """

        # Extracts the coordinates of the plants as gps coordinates
        self.gps_coords_ = self.csv_data_[self.csv_data_["treatment"] == 1][
            ["longitude", "latitude", "height"]
        ]
        # Extracts the coordinates of the plants as row and plant id
        self.field_coords_ = self.csv_data_[self.csv_data_["treatment"] == 1][
            ["row_number", "plant_id"]
        ]

        # Gets the rows and sorts them from the first to the last
        row_number, plant_count = np.unique(
            np.array((self.field_coords_["row_number"])), return_counts=True
        )
        row_number.sort()

        # Initializes the dictionnaries
        order = {}
        order_flipped = {}

        # Sorts the plants in the order they should be visited
        for i in row_number:
            plant_id_by_row = np.array(
                self.field_coords_[self.field_coords_["row_number"] == i]["plant_id"]
            )
            plant_id_by_row.sort()
            for j, plant_id in enumerate(plant_id_by_row):
                index = self.field_coords_[
                    (self.field_coords_["row_number"] == i)
                    * (self.field_coords_["plant_id"] == plant_id)
                ].index[0]
                if i % 2 == 0:
                    row_count = plant_count[i - 1] - j
                    order[index] = np.sum(plant_count[: i - 1]) + row_count - 1
                else:
                    order[index] = np.sum(plant_count[: i - 1]) + j

        # Flip the dictionnary to get the order from the index
        for key, value in order.items():
            order_flipped[value] = key
        new_indices = list(order_flipped.keys())
        new_indices.sort()

        index = [order_flipped[i] for i in new_indices]
        # Reindex the dataframes
        self.field_coords_ = self.field_coords_.reindex(index=np.array(index))
        self.gps_coords_ = self.gps_coords_.reindex(index=np.array(index))

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

    def getAltitudeFromHeightMap(self, x, y):
        x = int((x - self.offset_hmap_[0]) / self.heightmap_mpp_)
        y = int((y - self.offset_hmap_[1]) / self.heightmap_mpp_)
        return self.heightmap_[-y, x]

    def buildWayPoints(self) -> None:
        """
        Builds an ordered list of waypoints from the csv file,
        and instantiates the heightmap.
        """

        # Get the waypoints
        self.readCSV()
        self.getCoordsFromCSVData()

        # Convert to local coordinates (instead of GPS coordinates)
        lat = np.array(self.gps_coords_["latitude"])
        long = np.array(self.gps_coords_["longitude"])
        height = np.array(self.gps_coords_["height"])
        self.way_points = {}
        self.way_points_xyz = self.convertWaypointsToMapFrame(lat, long, height)
        self.initializeUAV()
        self.way_points_xyz = self.castWaypointsToLocalFrame(self.way_points_xyz)
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

    def castWaypointsToLocalFrame(self, xyz) -> np.ndarray:
        """
        Casts the waypoints from the map coordinates, that is the center of the field,
        to the local coordinates of the uav.
        """

        return xyz - self.offset_

    def convertWaypointsToMapFrame(
        self, lat: np.ndarray, long: np.ndarray, alt: np.ndarray
    ) -> np.ndarray:
        """
        Converts the waypoints from gps coordinates to local coordinates.

        Args:
            lat (np.ndarray): Numpy array containing the latitudes of the waypoints.
            long (np.ndarray): Numpy array containing the longitudes of the waypoints.
            alt (np.ndarray): Numpy array containing the altitudes of the waypoints.

        Returns:
            np.ndarray: Numpy array containing the local coordinates of the waypoints.
        """

        mlat = np.mean(lat)
        mlong = np.mean(long)
        malt = np.mean(alt)
        self.zero_llh = np.array([mlat, mlong, malt])

        xyz = np.zeros((len(lat), 3))
        for i, lla in enumerate(zip(lat, long, alt)):
            xyz[i] = self.castLLHToXYZ(lla[0], lla[1], lla[2] + self.altitude_offset_)
        return xyz

    # =============================================================================
    #        Waypoints visualization
    # =============================================================================

    def buildWayPointsViz(self):
        self.waypoints_points_ = []
        for i, waypoint in enumerate(self.way_points_xyz):
            point = Point()
            point.x = waypoint[0]
            point.y = waypoint[1]
            point.z = 0
            self.waypoints_points_.append(point)

        waypoints_viz_ = MarkerArray()
        for i, waypoint in enumerate(self.waypoints_points_):
            marker = Marker()
            marker.header.frame_id = self.local_frame_
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = waypoint
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.waypoints_size_
            marker.scale.y = self.waypoints_size_
            marker.scale.z = self.waypoints_size_
            marker.color.a = 1.0
            marker.color.r = self.waypoints_color_[0]
            marker.color.g = self.waypoints_color_[1]
            marker.color.b = self.waypoints_color_[2]
            marker.lifetime = rospy.Duration(0)
            waypoints_viz_.markers.append(marker)
        self.waypoints_viz_pub_.publish(waypoints_viz_)

    def updateCurrent(self, id: int) -> None:
        if self.publish_visualization_:
            waypoints_viz_ = MarkerArray()
            marker = Marker()
            marker.header.frame_id = self.local_frame_
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = self.waypoints_points_[id]
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.waypoints_size_
            marker.scale.y = self.waypoints_size_
            marker.scale.z = self.waypoints_size_
            marker.color.a = 1.0
            marker.color.r = self.waypoints_color_current_[0]
            marker.color.g = self.waypoints_color_current_[1]
            marker.color.b = self.waypoints_color_current_[2]
            marker.lifetime = rospy.Duration(0)
            waypoints_viz_.markers.append(marker)
        self.waypoints_viz_pub_.publish(waypoints_viz_)

    def updateUnreachable(self, id: int) -> None:
        if self.publish_visualization_:
            waypoints_viz_ = MarkerArray()
            marker = Marker()
            marker.header.frame_id = self.local_frame_
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = self.waypoints_points_[id]
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.waypoints_size_
            marker.scale.y = self.waypoints_size_
            marker.scale.z = self.waypoints_size_
            marker.color.a = 1.0
            marker.color.r = self.waypoints_color_unreachable_[0]
            marker.color.g = self.waypoints_color_unreachable_[1]
            marker.color.b = self.waypoints_color_unreachable_[2]
            marker.lifetime = rospy.Duration(0)
            waypoints_viz_.markers.append(marker)
        self.waypoints_viz_pub_.publish(waypoints_viz_)

    def updateReached(self, id: int) -> None:
        if self.publish_visualization_:
            waypoints_viz_ = MarkerArray()
            marker = Marker()
            marker.header.frame_id = self.local_frame_
            marker.header.stamp = rospy.Time(0)
            marker.ns = "waypoints"
            marker.id = id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = self.waypoints_points_[id]
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.waypoints_size_
            marker.scale.y = self.waypoints_size_
            marker.scale.z = self.waypoints_size_
            marker.color.a = 1.0
            marker.color.r = self.waypoints_color_visited_[0]
            marker.color.g = self.waypoints_color_visited_[1]
            marker.color.b = self.waypoints_color_visited_[2]
            marker.lifetime = rospy.Duration(0)
            waypoints_viz_.markers.append(marker)
        self.waypoints_viz_pub_.publish(waypoints_viz_)

    # =============================================================================
    #        Obstacle avoidance & Planner
    # =============================================================================

    def getDistanceToWayPoint(self, way_point_idx: int) -> float:
        """
        Computes the distance between the UAV and the specified waypoint.

        Args:
            way_point_idx (int): Index of the waypoint in the list of waypoints.

        Returns:
            float: Distance between the UAV and the waypoint."""

        xyz_uav = np.array(
            [
                self.local_pose_.pose.position.x,
                self.local_pose_.pose.position.y,
                self.local_pose_.pose.position.z,
            ]
        )
        d = xyz_uav - self.way_points_xyz[way_point_idx]
        d[2] = (
            self.getAltitudeFromHeightMap(
                self.way_points_xyz[way_point_idx][0],
                self.way_points_xyz[way_point_idx][1],
            )
            - xyz_uav[2]
        )

        return np.sqrt(np.sum(d * d))

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
            # r = (norm - self.obstacles_radius_) / self.obstacles_radius_
            # r = np.square(np.tanh(r * 3))
            # dx2 = np.sum(in_2rad * np.cos(theta) * r)
            # dy2 = np.sum(in_2rad * np.sin(theta) * r)
            # n2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
            # if n2:
            #    dx2 = dx2 * self.w_obstacle_ / n2  # / 3
            #    dy2 = dy2 * self.w_obstacle_ / n2  # / 3
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
            dx += dx2
            dy += dy2
        return dx, dy

    def goToWayPointPositionControl(self, way_point_idx):
        # WP coordinates
        xyz = self.way_points_xyz[way_point_idx]
        # UAV coordinates
        xyz_uav = np.array(
            [
                self.local_pose_.pose.position.x,
                self.local_pose_.pose.position.y,
                self.local_pose_.pose.position.z,
            ]
        )
        # Compute distance and angle
        d = xyz_uav - xyz
        norm = np.sqrt(np.sum(d * d))
        theta = np.arctan2(d[1], d[0])
        # Compute the distance to travel
        if norm < self.travel_dist_:
            travel_dist = norm
        else:
            travel_dist = self.travel_dist_
        # Computes the weight of the goal
        w_goal_x = travel_dist * np.cos(theta) * self.w_goal_
        w_goal_y = travel_dist * np.sin(theta) * self.w_goal_
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
        pose.pose.position.z = self.getAltitudeFromHeightMap(x, y)
        pose.pose.orientation.x = 0
        pose.pose.orientation.y = 0
        pose.pose.orientation.z = np.sin(yaw * 0.5)
        pose.pose.orientation.w = np.cos(yaw * 0.5)

        return pose

    def goToWayPointVelocityControl(self, way_point_idx):
        # WP coordinates
        xyz = self.way_points_xyz[way_point_idx]
        # UAV coordinates
        xyz_uav = np.array(
            [
                self.local_pose_.pose.position.x,
                self.local_pose_.pose.position.y,
                self.local_pose_.pose.position.z,
            ]
        )
        # Compute distance and angle
        d = xyz_uav - xyz
        norm = np.sqrt(np.sum(d * d))
        theta = np.arctan2(d[1], d[0])
        # Compute the distance to travel
        if norm < self.travel_dist_:
            travel_dist = norm
        else:
            travel_dist = self.travel_dist_
        # Computes the weight of the goal
        w_goal_x = travel_dist * np.cos(theta) * self.w_goal_
        w_goal_y = travel_dist * np.sin(theta) * self.w_goal_
        # Computes the weight of the obstacles
        w_obstacle_x = 0
        w_obstacle_y = 0
        w_obstacle_x, w_obstacle_y = self.compute_obstacles_cost(xyz_uav)
        # Computes the final weight
        dx = w_goal_x + w_obstacle_x
        dy = w_goal_y + w_obstacle_y
        # Get the direction of the travel
        norm_dxdy = np.sqrt(dx * dx + dy * dy)
        dx = dx / norm_dxdy
        dy = dy / norm_dxdy
        dz = self.getAltitudeFromHeightMap(xyz_uav[0], xyz_uav[1]) - xyz_uav[2]
        # Compute the delta yaw
        yaw = theta + np.pi
        dyaw = yaw - quat2yaw(self.local_pose_.pose.orientation)
        # Compute the velocities
        vx = dx * norm
        vy = dy * norm
        vz = dz
        vyaw = dyaw

        # make the velocity message
        vel = TwistStamped()

        return vel

    def isWayPointReachable(self, way_point_id):
        xyz = self.way_points_xyz[way_point_id]
        reachable = True

        if self.obstacles_position_.shape[0] == 0:
            # No obstacles, the goal is always reachable
            reachable = True
        else:
            # Check if the goal is reachable
            d = self.obstacles_position_[:, :2] - xyz[:2]
            norm = np.sqrt(np.sum(d * d, axis=1))
            reachable = np.sum(norm < self.obstacles_radius_) == 0
            if (not reachable) and (way_point_id not in self.unreachable_ids_):
                self.unreachable_ids_.append(way_point_id)
        return reachable

    def followWayPoints(self):
        rate = rospy.Rate(20)
        way_point_id = 0
        self.armUAV()
        for i in range(100):
            self.pose_pub.publish(PoseStamped())
            rate.sleep()
        self.setOffboardMode()
        self.takeOff()
        rospy.loginfo(
            "Going to way point %.2f, %.2f"
            % (
                self.way_points_xyz[way_point_id][0],
                self.way_points_xyz[way_point_id][1],
            )
        )
        self.current_id_ = way_point_id
        for i in range(3):
            self.buildWayPointsViz()
            rate.sleep()
        while (not rospy.is_shutdown()) and (way_point_id < len(self.way_points_xyz)):
            self.lock_.acquire()
            if self.getDistanceToWayPoint(way_point_id) < self.d_threshold_:
                rospy.loginfo("Way point reached")
                self.updateReached(way_point_id)
                way_point_id += 1
                rospy.loginfo(
                    "Going to way point %.2f, %.2f"
                    % (
                        self.way_points_xyz[way_point_id][0],
                        self.way_points_xyz[way_point_id][1],
                    )
                )
                self.current_id_ = way_point_id
            if not self.isWayPointReachable(way_point_id):
                rospy.logwarn("Way point is obstructed. Jumping to the next.")
                self.updateUnreachable(way_point_id)
                way_point_id += 1
                self.current_id_ = way_point_id
            else:
                self.updateCurrent(way_point_id)
            self.pose_pub.publish(self.goToWayPoint(way_point_id))
            self.lock_.release()
            rate.sleep()
        self.landAtHome()


if __name__ == "__main__":
    rospy.init_node("waypoint_planner", anonymous=True)
    SUP = SprayingUAVPlanner()
    SUP.followWayPoints()
