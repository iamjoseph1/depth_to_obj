import sys
import os
import numpy as np
import rospy
import ros_numpy
import message_filters
import cv2
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# from ultralytics import SAM2
# from ultralytics import YOLOv10
from ultralytics import YOLO

# ***** you have to update your ultralytics : pip install -U ultralytics *****
# predictor = models.sam2.SAM2Predictor()
# model_sam2 = SAM2("sam2_b.pt")
# print("*********SAM2 model loaded*********")
# model_yolo = YOLOv10(f'/home/dyros/tocabi_ws/src/depth_to_obj/src/best_v10.pt')
model_yolo = YOLO('/home/dyros/tocabi_ws/src/depth_to_obj/src/best_v8_seg_e50.pt')
print("*********Yolo model loaded*********")
# model.info()

# count = 0
last_time = 0
USERNAME = os.getlogin()

seg_image_pub = rospy.Publisher("/yolov8_segmentation/labeled/image", Image, queue_size=5)
new_depth_pub = rospy.Publisher("mujoco_sim_ros/depth/image_new", Image, queue_size=5)

# def initial_image_callback(color_img_msg):
#     img_array = ros_numpy.numpify(color_img_msg)
#     seg_result = model(img_array, points=[300,240])
#     name = seg_result[0].names
#     rospy.signal_shutdown("Received first message.")

# for yolov10
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

# for yolov10
def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    bbox_center = []

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)

    return img, results

# for yolov8_seg
def get_mask_coords_yolov8(color_img_msg):

    img_array = ros_numpy.numpify(color_img_msg)
    mask_coords_list = []
    seg_annotated = np.zeros((680,480,3),dtype=np.uint8)

    seg_result = model_yolo(img_array)
    # print(seg_result[0])
    masks = seg_result[0].masks
    
    if masks.data.is_cuda:
        masks_array = masks.data.cpu().numpy()
    else:
        masks_array = masks.data.numpy()

    for mask in masks_array:

        # Get the coordinates of all non-zero pixels (i.e., mask pixels)
        coords = np.column_stack(np.where(mask > 0))
        mask_coords_list.append(coords.tolist())

    # print(mask_coords_list)    
    seg_annotated = seg_result[0].plot()

    return mask_coords_list, seg_annotated

def store_image(img_msg):

    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(img_msg,"bgr8")
    img_path = '/home/' + USERNAME + '/tocabi_ws/src/depth_to_obj/realsense_img/realsense_img'+str(count)+'.jpg'
    cv2.imwrite(img_path, img)

    return img_path
    
# for SAM2    
def get_mask_coords(color_img_msg):

    img_array = ros_numpy.numpify(color_img_msg)
    mask_coords_list = []
    seg_annotated = np.zeros((680,480,3),dtype=np.uint8)

    # # assuming initial condition (when panda see cabinet's face directly)
    # seg_result = model(img_array, bboxes=[220,160,460,320])
    seg_result = model_sam2(img_array, points=[300,240])
    print(seg_result[0])
    masks = seg_result[0].masks
    
    if masks.data.is_cuda:
        masks_array = masks.data.cpu().numpy()
    else:
        masks_array = masks.data.numpy()

    for mask in masks_array:

        # Get the coordinates of all non-zero pixels (i.e., mask pixels)
        coords = np.column_stack(np.where(mask > 0))
        mask_coords_list.append(coords.tolist())

    # print(mask_coords_list)    
    seg_annotated = seg_result[0].plot(show=False)

    return mask_coords_list, seg_annotated

def create_new_depth(depth_img_msg, mask_coords_list):

    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(depth_img_msg,"32FC1")
    for mask in mask_coords_list:
        for coord in mask:
            x = coord[0]
            y = coord[1]
            depth_image[x,y] = depth_image[mask[0][0]-10, mask[0][1]]
            # depth_image[x,y] = 255
    
    return depth_image

def synk_callback(color_img_msg, depth_img_msg):
    
    # global count

    # global last_time
    # current_time = time.time()

    # if current_time - last_time > 7:
    mask_coords_list, seg_annotated = get_mask_coords_yolov8(color_img_msg)   
    depth_image = create_new_depth(depth_img_msg, mask_coords_list)

    seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))
    print('segmented image published')
    new_depth_pub.publish(ros_numpy.msgify(Image, depth_image, encoding="32FC1"))
    print('new depth image published')


    # img_array = ros_numpy.numpify(color_img_msg)
    # img, results = predict_and_detect(model_yolo, img_array)
    # seg_image_pub.publish(ros_numpy.msgify(Image, img, encoding="rgb8"))
    # print('segmented image published')

    # count =1
    # last_time = time.time()

def main():

    # global last_time
    # last_time = time.time()

    color_img_sub = message_filters.Subscriber("/mujoco_ros_interface/camera/image", Image)
    depth_raw_sub = message_filters.Subscriber("/mujoco_ros_interface/camera/depth", Image)
    ts = message_filters.ApproximateTimeSynchronizer([color_img_sub, depth_raw_sub], 3, 0.01)
    ts.registerCallback(synk_callback)
    
    rospy.spin()

if __name__ == '__main__':

    rospy.init_node('stereo_sam2', anonymous=True)
    # rospy.subscriber("/realsense/color/image_raw", Image, initial_image_callback)
    main()