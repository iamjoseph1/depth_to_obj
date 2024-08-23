import sys
import os
import numpy as np
import math
import time
import rospy
import message_filters
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix, quaternion_from_matrix, quaternion_multiply

from sensor_msgs.msg import Image, JointState, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from actionlib_msgs.msg import GoalStatusArray
# from geometry_msgs.msg import PoseStamped, Pose, Point
# from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject
# from gazebo_msgs.msg import LinkStates
# from shape_msgs.msg import SolidPrimitive, Plane, Mesh, MeshTriangle
# import pyassimp
from cv_bridge import CvBridge

USERNAME = os.getlogin()
sys.path.append('/home/{}/tocabi_ws/src/suhan_robot_model_tools'.format(USERNAME))
from srmt.planning_scene import PlanningSceneLight
# from suhan_robot_model_tools.suhan_robot_model_tools_wrapper_cpp import PlanningSceneCollisionCheck
# from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
# import moveit_commander

import trimesh
import coacd

count = 0
last_time = 0
last_status = 0

# robot = moveit_commander.move_group.MoveGroupCommander(group_name)
# planning_scene_colission = PlanningSceneCollisionCheck(topic_name)
# planning_scene_colission.set_frame_id('/world')
# planning_scene_collision.set_frame_id(robot.get_planning_frame())
# planning_scene_interface = moveit_commander.planning_scene_interface.PlanningSceneInterface()
planning_scene_collision_world = PlanningSceneLight(topic_name='/planning_scene', base_link='world')

print("**********import finished**********")

def synk_callback(image_msg, joint_msg, moving_msg):

    global planning_scene_collision_world
    global last_status

    q_current = joint_msg.position
    q_current = np.array([*q_current])

    for status in moving_msg.status_list:

        if status.status == 3 and status.status-last_status != 0:

            # create object from depth image
            H, V = int(87), int(58)
            objPath = create_object(image_msg, H, V)
            object_simplification(objPath)

            # # if you need pointcloud2 msg...
            # centroid = point_cloud_centroid(pointcloud2_msg)

            camera_frame = 'camera_realsense_link_gazebo'
            camera_ref_euler = [np.deg2rad(-90), np.deg2rad(90), 0]

            # set pose & orientation of object
            object_id, trans_obj, quat_final = get_pose(camera_frame, camera_ref_euler)

            # quat_final = quaternion_from_euler(0,0,0)
            # quat_final = quaternion_from_euler(np.deg2rad(0),np.deg2rad(-90),np.deg2rad(-90), 'rxyz')

            # add object to planning scene
            addPlanningScene(object_id, trans_obj, quat_final, planning_scene_collision_world)

    last_status = status.status
    
    # display current joint state & object to planning scene
    planning_scene_collision_world.display(group_name = 'panda_arm', q = q_current)

def vete(v, vt):
    return str(v)+"/"+str(vt)

def create_object(image_msg, H, V):

    create_object_start_time = time.time()
    # obj_pub = rospy.Publisher('/depth_to_obj/object_path', String, queue_size=10)
    bridge = CvBridge()
    cv_depth_image = bridge.imgmsg_to_cv2(image_msg,"32FC1")
    # print(cv_depth_image.shape)

    # depthpath = "/home/dyros/tocabi_ws/src/DepthData/DepthData" + str(count)+".png"
    # cv2.imwrite(depthpath, cv_depth_image)
    # print(os.path.abspath(depthpath))
    
    img = cv_depth_image

    if len(img.shape) > 2 and img.shape[2] > 1:
       rospy.loginfo("Expecting a 1D map, but depth map through /realsense/depth/image_new has shape %r"% img.shape)
       return

    # img = 1.0 - img

    h = img.shape[0]    # height : 480
    w = img.shape[1]    # width : 680

    horizon_FOV = math.radians(H)
    vertical_FOV = math.radians(V)

    # Get Distance based on horizon_FOV : width of the scene is commonly more relevant
    D = (w/2)/math.tan(horizon_FOV/2)

    objPath = '/home/dyros/tocabi_ws/src/ObjData/realsense' + str(count)+'.obj'

    if max(objPath.find('\\'), objPath.find('/')) > -1:
        os.makedirs(os.path.dirname(objPath), exist_ok=True)
    
    with open(objPath,"w") as f:

        ids = np.zeros((img.shape[1], img.shape[0]), int)
        vid = 1

        for u in range(0, w):
            for v in range(h-1, -1, -1):

                d = img[v, u]

                ids[u,v] = vid
                if d == 0.0:
                    ids[u,v] = 0
                vid += 1

                x = u - w/2
                y = v - h/2
                z = -D

                norm = 1 / math.sqrt(x*x + y*y + z*z)

                t = d/(z*norm)

                x = -t*x*norm
                y = t*y*norm
                z = -t*z*norm        

                f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")

        for u in range(0, img.shape[1]):
            for v in range(0, img.shape[0]):
                f.write("vt " + str(u/img.shape[1]) + " " + str(v/img.shape[0]) + "\n")

        for u in range(0, img.shape[1]-1):
            for v in range(0, img.shape[0]-1):

                v1 = ids[u,v]; v2 = ids[u+1,v]; v3 = ids[u,v+1]; v4 = ids[u+1,v+1];

                if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                    continue

                f.write("f " + vete(v1,v1) + " " + vete(v2,v2) + " " + vete(v3,v3) + "\n")
                f.write("f " + vete(v3,v3) + " " + vete(v2,v2) + " " + vete(v4,v4) + "\n")
    create_object_end_time = time.time()
    print("object file created!")
    print("time for creating object : ", create_object_end_time-create_object_start_time)

    return objPath

def object_simplification(objPath):

    coacd_start_time = time.time()

    global count
    mesh = trimesh.load(objPath, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    result = coacd.run_coacd(
        mesh,
        threshold=0.05, # depend on user's choice _ default : 0.05
        max_convex_hull=100
    )
    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    scene = trimesh.Scene()
    np.random.seed(0)
    for p in mesh_parts:
        # p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(p)
    
    output_path = '/home/dyros/tocabi_ws/src/depth_to_obj/coacd_data/realsense' + str(count)+'.obj'
    scene.export(output_path)
    # print(os.path.abspath(output_path))

    coacd_end_time = time.time()
    print("CoACD object file created!")
    print("time for simplifying object : ", coacd_end_time-coacd_start_time)

    # return output_path

def point_cloud_centroid(pointcloud2_msg):

    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    total_point = 0

    global_frame = 'world'

    # Initialize a buffer and listener for transformations
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    try:
        # Get the transformation from the original frame to the target frame
        transform = tf_buffer.lookup_transform(global_frame, pointcloud2_msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
        
        # Transform the point cloud to the target frame
        transformed_cloud = do_transform_cloud(pointcloud2_msg, transform)

        # Process the transformed point cloud
        # Iterate through points in the PointCloud2 message
        for point in pc2.read_points(transformed_cloud, skip_nans=True):
            sum_x += point[0]
            sum_y += point[1]
            sum_z += point[2]
            total_point += 1

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(e)

    # Calculate the centroid
    if total_point > 0:
        centroid_x = sum_x / total_point
        centroid_y = sum_y / total_point
        centroid_z = sum_z / total_point
        centroid = [centroid_x, centroid_y, centroid_z]

        return centroid
    else:
        return None, None, None

def get_pose(camera_frame, camera_ref_euler):

    get_pose_start_time = time.time()

    # Give each of the scene objects a unique name        
    object_id = 'object'
    global_frame = 'world'

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # Get the transformation from the original frame to the target frame
    transform_ee = tf_buffer.lookup_transform(global_frame, camera_frame, rospy.Time(0), rospy.Duration(1.0))
    trans_obj = [transform_ee.transform.translation.x, 
           transform_ee.transform.translation.y, 
           transform_ee.transform.translation.z]

    rotation = [transform_ee.transform.rotation.x, 
                transform_ee.transform.rotation.y,
                transform_ee.transform.rotation.z,
                transform_ee.transform.rotation.w]
    euler_rotation = euler_from_quaternion(rotation)
    
    # Set reference orientation of camera
    # **********rotate at local frame**********
    R_ref_camera = quaternion_matrix(quaternion_from_euler(np.deg2rad(-90),np.deg2rad(90),0, 'rxyz'))
    R_camera = quaternion_matrix(rotation)
    R_relative_camera = np.dot(np.linalg.inv(R_ref_camera), R_camera)
    relative_rotation = quaternion_from_matrix(R_relative_camera)
    relative_euler_rotation = euler_from_quaternion(relative_rotation)

    print("Relative orientation of camera : (%f, %f, %f)" % (relative_euler_rotation[0]*180/np.pi,
                                                    relative_euler_rotation[1]*180/np.pi,
                                                    relative_euler_rotation[2]*180/np.pi))
    
    # Set reference orientation of object : object's z-axis points panda
    # **********rotate at local frame**********
    quat_0 = quaternion_from_euler(0,np.deg2rad(-90),np.deg2rad(-90), 'rxyz')

    # # Calculate orientation offset of object -> Get final orientation of object

    quat_pitch = quaternion_from_euler(0, -relative_euler_rotation[0], 0, 'rxyz')
    quat_1 = quaternion_multiply(quat_pitch, quat_0)
    quat_yaw = quaternion_from_euler(0, 0, -relative_euler_rotation[1], 'rxyz')
    quat_2 = quaternion_multiply(quat_yaw, quat_1)
    quat_roll = quaternion_from_euler(relative_euler_rotation[2], 0, 0, 'rxyz')
    quat_final = quaternion_multiply(quat_roll, quat_2)

    get_pose_end_time = time.time()
    print("time for setting pose of object : ", get_pose_end_time-get_pose_start_time)

    return object_id, trans_obj, quat_final

def addPlanningScene(object_id, trans_obj, quat_final, planning_scene_collision_world):

    add_scene_start_time = time.time()

    global count
    
    file_path = 'file:///home/' + USERNAME + '/tocabi_ws/src/depth_to_obj/coacd_data/realsense' + str(count) + '.obj'
    planning_scene_collision_world.add_mesh(object_id, file_path, trans_obj, quat_final)
    count += 1

    add_scene_end_time = time.time()
    print("Visualize object!")
    print("time for uploading object : ", add_scene_end_time-add_scene_start_time)

def main():

    # global last_time
    # last_time = time.time()
    # rospy.init_node('depth_to_obj', anonymous=True)
    # moveit_commander.roscpp_initialize(sys.argv)
    image_sub = message_filters.Subscriber("/realsense/depth/image_rect_raw", Image)
    joint_sub = message_filters.Subscriber("/franka_state_controller/joint_states", JointState)
    moving_sub = message_filters.Subscriber("/move_group/status", GoalStatusArray)
    # pointcloud_sub = message_filters.Subscriber("/realsense/depth/color/points", PointCloud2)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, joint_sub, moving_sub], 3, 0.05)
    ts.registerCallback(synk_callback)

    # euler_orientation = rospy.Subscriber("/gazebo/link_states", LinkStates, link_state_callback)

    rospy.spin()

if __name__ == '__main__':
    main()

    # # Calculate orientation of object at once
    # quat_2 = quaternion_from_euler(relative_euler_rotation[2], -relative_euler_rotation[0], -relative_euler_rotation[1], 'rxyz')

    # # Calculate final pose of object
    # pos_final_z = centroid[2] + (centroid[0]-pos[0])*np.tan(-relative_euler_rotation[0]) #adjust object's pos with pitch


    # print("Orientation of camera_world frame : (%f, %f, %f)" % (euler_rotation[0]*180/np.pi,
    #                                                 euler_rotation[1]*180/np.pi,
    #                                                 euler_rotation[2]*180/np.pi))