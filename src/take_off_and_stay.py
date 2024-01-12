#!/usr/bin/env python3

import rospy
import os
import numpy as np
import threading

from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State

from std_srvs.srv import Trigger, TriggerRequest


class SimpleTakeOffAndStay:
    def __init__(self):
        self.take_off_altitude = rospy.get_param("~take_off_altitude", 1.0)
        self.rate = rospy.get_param("~rate", 30)
        self.timeout = rospy.get_param("~timeout", 10)
        self.uav_name = rospy.get_param("~uav_name", "/uav_a")

        self.landing_requested = False
        self.coordinates_received = False
        self.state = None
        self.initial_altitude = 0.0

        self.local_pose_sub = rospy.Subscriber(
            "/uav_a/mavros/local_position/pose", PoseStamped, self.localPoseCallback
        )
        self.state_sub = rospy.Subscriber(
            "/uav_a/mavros/state", State, self.stateCallback
        )
        self.take_off_srv = rospy.Service(
            "/uav_a/take_off", Trigger, self.takeOffRequest
        )
        self.land_srv = rospy.Service("/uav_a/land", Trigger, self.landRequest)

        self.pose_pub = rospy.Publisher(
            "/uav_a/mavros/setpoint_position/local", PoseStamped, queue_size=1
        )

        self.get_initial_coordinates()
        self.get_services()

    def get_services(self) -> None:
        rospy.wait_for_service(self.uav_name + "/mavros/set_mode")
        self.mode = rospy.ServiceProxy(self.uav_name + "/mavros/set_mode", SetMode)
        rospy.wait_for_service(self.uav_name + "/mavros/cmd/arming")
        self.arm = rospy.ServiceProxy(self.uav_name + "/mavros/cmd/arming", CommandBool)

    def get_initial_coordinates(self) -> None:
        """
        Gets the initial coordinates of the UAV.
        """

        rospy.loginfo("Waiting for initial coordinates")
        while (not rospy.is_shutdown()) and (self.coordinates_received == False):
            rospy.loginfo("Waiting for initial coordinates")
            rospy.sleep(2.0)
        rospy.loginfo("Initial coordinates received")

        self.initial_altitude = self.local_pose_.pose.position.z
        self.cutoff_altitude = self.initial_altitude + 0.05
        self.take_off_altitude = self.initial_altitude + self.take_off_altitude

        self.stay_coordinates = [
            self.local_pose_.pose.position.x,
            self.local_pose_.pose.position.y,
            self.take_off_altitude,
        ]

        rospy.loginfo("Stay coordinates: {}".format(self.stay_coordinates))
        rospy.loginfo("Take off altitude: {}".format(self.take_off_altitude))
        rospy.loginfo("Initial altitude: {}".format(self.initial_altitude))
        rospy.loginfo("Cutoff altitude: {}".format(self.cutoff_altitude))

        while (not rospy.is_shutdown()) and (self.state is None):
            rospy.loginfo("Waiting for MAVROS state to be received")
            rospy.sleep(2.0)

        rospy.loginfo("MAVROS state received!")

    def armUAV(self) -> None:
        """
        Arms the UAV.
        """

        arm_req = CommandBoolRequest()
        arm_req.value = True
        rospy.loginfo("Arming UAV")
        while (not rospy.is_shutdown()) and (self.state.armed == False):
            resp = self.arm(arm_req)
            if resp:
                rospy.loginfo("Successfully armed UAV")
            else:
                rospy.logerr("Failed to arm UAV")
            rospy.sleep(0.25)

    def threadedARM(self) -> None:
        """
        Arms the UAV in a separate thread.
        """

        thread = threading.Thread(target=self.armUAV)
        thread.start()

    def disarmUAV(self) -> None:
        """
        Disarms the UAV.
        """

        arm_req = CommandBoolRequest()
        arm_req.value = False
        resp = self.arm(arm_req)
        if resp:
            rospy.loginfo("Successfully disarmed UAV")
        else:
            rospy.logerr("Failed to disarm UAV")

    def threadedDISARM(self) -> None:
        """
        Disarms the UAV in a separate thread.
        """

        thread = threading.Thread(target=self.disarmUAV)
        thread.start()

    def setOffboardMode(self) -> None:
        """
        Sets the UAV in offboard mode.
        It allows us to send position and velocity commands to the UAV.
        """

        mode_req = SetModeRequest()
        mode_req.custom_mode = "OFFBOARD"
        rospy.loginfo("Switching to OFFBOARD mode")
        while (not rospy.is_shutdown()) and (self.state.mode != "OFFBOARD"):
            resp = self.mode(mode_req)
            if resp:
                rospy.loginfo("Set OFFBOARD mode")
            else:
                rospy.logerr("Failed to set OFFBOARD mode")
            rospy.sleep(0.25)

    def threadedOFFBOARD(self) -> None:
        """
        Sets the UAV in offboard mode in a separate thread.
        """

        thread = threading.Thread(target=self.setOffboardMode)
        thread.start()

    def takeOff(self, height: float = 4.0) -> None:
        """
        Takes off to the specified height.

        Args:
            height (float, optional): Height the UAV should reach. Defaults to 4.0.
        """

        rate = rospy.Rate(self.rate)
        alt_reached = False
        pose = PoseStamped()
        pose.pose.position.x = self.stay_coordinates[0]
        pose.pose.position.y = self.stay_coordinates[1]
        pose.pose.position.z = height
        pose.pose.orientation.w = 1.0
        rospy.loginfo("Taking off")
        starting_time = rospy.Time.now()
        while (not rospy.is_shutdown()) and (alt_reached == False):
            if self.state.mode != "OFFBOARD":
                delta = rospy.Time.now() - starting_time
                if delta.to_sec() > self.timeout:
                    rospy.loginfo("Timeout reached")
                    rospy.loginfo("OFFBOARD mode not set")
                    rospy.loginfo("Current state: {}".format(self.state.mode))
                    break
            if self.state.armed == False:
                delta = rospy.Time.now() - starting_time
                if delta.to_sec() > self.timeout:
                    rospy.loginfo("Timeout reached")
                    rospy.loginfo("UAV not armed")
                    rospy.loginfo("Current state: {}".format(self.state.mode))
                    break
            self.pose_pub.publish(pose)
            alt_reached = np.abs(self.local_pose_.pose.position.z - height) < 0.25
            rate.sleep()
        rospy.loginfo("Take off complete")

    def land(self) -> None:
        """
        Lands the UAV.
        """

        rospy.loginfo("Landing")
        rate = rospy.Rate(self.rate)
        pose = PoseStamped()
        pose.pose.position.x = self.stay_coordinates[0]
        pose.pose.position.y = self.stay_coordinates[1]
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        while (not rospy.is_shutdown()) and (
            self.local_pose_.pose.position.z > self.cutoff_altitude
        ):
            self.pose_pub.publish(pose)
            rate.sleep()
        rospy.loginfo("Landing complete")

    def stayAtPosition(self) -> None:
        """
        Stays at the specified coordinates.
        """

        rate = rospy.Rate(self.rate)
        pose = PoseStamped()
        pose.pose.position.x = self.stay_coordinates[0]
        pose.pose.position.y = self.stay_coordinates[1]
        pose.pose.position.z = self.stay_coordinates[2]
        pose.pose.orientation.w = 1.0
        starting_time = rospy.Time.now()
        rospy.loginfo("Staying at position")
        while (not rospy.is_shutdown()) and (self.landing_requested == False):
            if self.landing_requested:
                break
            if self.state.mode != "OFFBOARD":
                delta = rospy.Time.now() - starting_time
                if delta.to_sec() > self.timeout:
                    rospy.loginfo("Timeout reached")
                    rospy.loginfo("OFFBOARD mode not set")
                    rospy.loginfo("Current state: {}".format(self.state.mode))
                    break
            if self.state.armed == False:
                delta = rospy.Time.now() - starting_time
                if delta.to_sec() > self.timeout:
                    rospy.loginfo("Timeout reached")
                    rospy.loginfo("UAV not armed")
                    rospy.loginfo("Current state: {}".format(self.state.mode))
                    break
            pose.header.stamp = rospy.Time.now()
            self.pose_pub.publish(pose)
            rate.sleep()
        rospy.loginfo("Stay complete")

    def localPoseCallback(self, msg: PoseStamped) -> None:
        """
        Callback function for the local pose subscriber.

        Args:
            msg (PoseStamped): PoseStamped message containing the pose of the UAV.
        """
        self.coordinates_received = True
        self.local_pose_ = msg

    def stateCallback(self, msg: State) -> None:
        """
        Callback function for the state subscriber.

        Args:
            msg (State): State message containing the state of the UAV.
        """
        self.state = msg

    def takeOffRequest(self, srv: Trigger) -> None:
        """
        Callback function for the take off service.
        """

        rate = rospy.Rate(20)
        pose = PoseStamped()
        pose.pose.position.x = self.stay_coordinates[0]
        pose.pose.position.y = self.stay_coordinates[1]
        pose.pose.position.z = self.stay_coordinates[2]
        pose.pose.orientation.w = 1.0
        for i in range(50):
            self.pose_pub.publish(pose)
            rate.sleep()
        self.threadedOFFBOARD()
        self.threadedARM()
        self.takeOff(self.take_off_altitude)
        self.stayAtPosition()

    def landRequest(self, srv: Trigger) -> None:
        """
        Callback function for the land service.
        """
        self.landing_requested = True
        self.land()
        self.disarmUAV()


if __name__ == "__main__":
    rospy.init_node("waypoint_planner", anonymous=True)
    take_off_and_stay = SimpleTakeOffAndStay()
    rospy.spin()
