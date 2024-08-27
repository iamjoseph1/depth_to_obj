from ultralytics import YOLO
import random
import cv2
import numpy as np

model_yolo = YOLO('/home/dyros/tocabi_ws/src/depth_to_obj/src/best_v8_seg_real_e50.pt')
print("*********Yolo model loaded*********")

conf = 0.9

for num in range(355,356):
    img_left = cv2.imread('/home/dyros/tocabi_ws/src/tocabi/data/stereo/left/'+str(num)+'.jpeg')
    img_right = cv2.imread('/home/dyros/tocabi_ws/src/tocabi/data/stereo/right/'+str(num)+'.jpeg')
    print(img_left.shape)

    # # yolo_classes = list(model_yolo.names.values())
    # # classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
    # mask_coords_list = []
    # seg_result = model_yolo(img_left)
    # # print(seg_result[0])
    # masks = seg_result[0].masks
    
    # if masks.data.is_cuda:
    #     masks_array = masks.data.cpu().numpy()
    # else:
    #     masks_array = masks.data.numpy()

    # for mask in masks_array:

    #     # Get the coordinates of all non-zero pixels (i.e., mask pixels)
    #     coords = np.column_stack(np.where(mask > 0))
    #     mask_coords_list.append(coords.tolist())

    # # print(mask_coords_list)    
    # seg_annotated = seg_result[0].plot()

    # for mask in mask_coords_list:
    #     for coord in mask:
    #         x = coord[0]
    #         y = coord[1]
    #         img_left[x,y] = img_left[mask[0][0]-10, mask[0][1]]

    results_left = model_yolo.predict(img_left, conf=conf)

    # colors = [random.choices(range(256), k=3) for _ in classes_ids]
    # print(results)
    for result in results_left:
        if type(result.masks) == type(None):
            continue
        for mask, box in zip(result.masks.xy, result.boxes):
            points = np.int32([mask])
            # print(points)
            # cv2.polylines(img_seg, points, True, (255, 0, 0), 1)
            # color_number = classes_ids.index(int(box.cls[0]))
            # for point_group in points:
            # print(points.shape)
            # for point in points:
            #     for coord in point:
            #         x = coord[1]
            #         y = coord[0]
            #         img_left[x,y] = img_left[points[0][0][1], points[0][0][0]-70].tolist()
            # cv2.fillPoly(img_left, points, color=img_left[points[0][0][1], points[0][0][0]-70].tolist())
            cv2.fillPoly(img_left, points, color=(0, 0, 255))
    
    cv2.imwrite('/home/dyros/tocabi_ws/src/tocabi/data/stereo_seg/left/seg_left_real_e50_conf75_'+str(num)+'.jpg', img_left)
    print('segmented image(left)' + str(num)+ ' saved!')

    results_right = model_yolo.predict(img_right, conf=conf)

    # colors = [random.choices(range(256), k=3) for _ in classes_ids]
    # print(results)
    for result in results_right:
        if type(result.masks) == type(None):
            continue
        for mask, box in zip(result.masks.xy, result.boxes):
            points = np.int32([mask])
            # print(points)
            # cv2.polylines(img_seg, points, True, (255, 0, 0), 1)
            # color_number = classes_ids.index(int(box.cls[0]))
            # for point_group in points:
            # print(points.shape)
            # cv2.fillPoly(img_right, points, color=img_right[points[0][0][1], points[0][0][0]-250].tolist())
            cv2.fillPoly(img_right, points, color=(0, 0, 255))

    cv2.imwrite('/home/dyros/tocabi_ws/src/tocabi/data/stereo_seg/right/seg_right_real_e50_conf75_'+str(num)+'.jpg', img_right)
    print('segmented image(right)' + str(num)+ ' saved!')