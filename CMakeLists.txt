cmake_minimum_required(VERSION 3.0.2)
project(depth_to_obj)

find_package(catkin REQUIRED COMPONENTS
    rospy
    message_generation
    std_msgs
    sensor_msgs
    geometry_msgs
    moveit_msgs
    cv_bridge
    tf2_ros
    tf2_sensor_msgs
    actionlib_msgs
    message_runtime
    # ros_numpy
    # gazebo_msgs
    )

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS
  LIBRARIES opencv
  LIBRARIES trimesh
  LIBRARIES coacd
  # LIBRARIES pyassimp
  LIBRARIES moveit_commander
  LIBRARIES ultralytics
  # LIBRARIES depth_to_obj
  CATKIN_DEPENDS cv_bridge rospy sensor_msgs std_msgs geometry_msgs moveit_msgs actionlib_msgs
#  DEPENDS system_lib
)

include_directories(
# include
#  ${depth_to_obj_INCLUDE_DIRS}
 ${catkin_INCLUDE_DIRS}
 ${OpenCV_INCLUDE_DIRS}
 ${coacd_INCLUDE_DIRS}
 ${trimesh_INCLUDE_DIRS}
 ${ultralytics_INCLUDE_DIRS}
#  ${pyassimp_INCLUDE_DIRS}
 ${moveit_commander_INCLUDE_DIRS}
)

 catkin_install_python(PROGRAMS
   src/depth_to_obj.py
   src/stereo_sam2.py
   src/yolov8_stereo_test.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

