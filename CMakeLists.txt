cmake_minimum_required(VERSION 3.0.2)
project(sesame_navigation)

find_package(catkin REQUIRED COMPONENTS
	geometry_msgs
	mavros_msgs
	mavros
	tf2_ros
	tf2_msgs
	tf2_geometry_msgs
	nav_msgs
	rospy
	std_msgs
	geographic_msgs
	visualization_msgs
	message_generation
)

#catkin_python_setup()

add_message_files(
	FILES
	PointId.msg
	PointIdArray.msg
)

generate_messages(
	DEPENDENCIES
	std_msgs
	geometry_msgs
)

catkin_package(
)

include_directories(
	${catkin_INCLUDE_DIRS}
)

install(DIRECTORY launch/
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

catkin_install_python(PROGRAMS
	src/detect_drones.py
	src/detect_obstacles.py
	src/drone_manager.py
	src/follower_uav_planner.py
	src/gps_command_node.py
	src/navsatfix_to_nmea.py
	src/obstacle_manager.py
	src/relay_topic_with_switch.py
	src/safety_pose_switch.py
	src/sprayer_uav_planner.py
	src/tf_broadcaster_map.py
	src/tf_broadcaster_odom.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#install(FILES
#)
