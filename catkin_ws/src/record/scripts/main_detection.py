#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
import os
import argparse
import numpy as np
import pyrealsense2 as rs2
import message_filters
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

class ImageListener:
    def __init__(self, depth_image_topic, depth_info_topic, color_image_topic):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--embedder',
            default='mobilenet',
            help='type of feature extractor to use',
            choices=[
                "mobilenet",
                "torchreid",
                "clip_RN50",
                "clip_RN101",
                "clip_RN50x4",
                "clip_RN50x16",
                "clip_ViT-B/32",
                "clip_ViT-B/16"
            ]
        )
        args = parser.parse_args()
        self.bridge = CvBridge()
        
        self.depth_sub = message_filters.Subscriber(depth_image_topic, msg_Image)
        self.color_sub = message_filters.Subscriber(color_image_topic, msg_Image)
        self.sub_info = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)

        self.depth_pub = rospy.Publisher('/bazant/depth', msg_Image, queue_size=10)
        self.color_pub = rospy.Publisher('/bazant/color', msg_Image, queue_size=10)
       
        self.pix = None
        self.pix_grade = None
        self.intrinsics = None

        self.tracker = DeepSort(max_age=30, embedder=args.embedder)

    def synchronize_callback(self, color_image, depth_image):

        start_time = rospy.Time.now()

        """
        Input:
        - color_image: Image with the fruits <class 'sensor_msgs.msg._Image.Image'>
        - MIN_AREA: Minimum area for the fruit <class 'int'>
        - MAX_AREA: Maximum area for the fruit <class 'int'>
        Return: 
        - stats: x1, y1, x2, y2 as a bounding box for each fruit <class 'list'>
        - ids of the fruit <class 'list'>
        """
        stats, labels, scores, bin_image = self.detectObjectBinarize(color_image, MIN_AREA=200, MAX_AREA=70000)

        """" In this place will be also DeepSORT algorithm """
        """
        Input:
        - image: Image with the fruits
        - stats: x1, y1, x2, y2 as a bounding box for each fruit
        - labels: labels of the fruit
        - scores: scores of the fruit
        Return:
        - detections: ????
        - ids of the fruit
        """
        detections, ids, cls = self.deepsort_algorithm(color_image, stats, labels, scores)

        """
        Input:
        - results: x1, y1, x2, y2 as a bounding box for each fruit
        - ids of the fruit
        - depth_image: Image with the depth information
        - class
        Return: 
        - Location of the fruits in 3D
        - ids of the fruit
        """
        fruit_pos, ids = self.position_fruits(depth_image, detections, ids, cls)

        end_time = rospy.Time.now()
        fps = 1.0 / (end_time - start_time).to_sec()

        frame = cv2.cvtColor(bin_image, cv2.COLOR_GRAY2BGR)
        # frame = self.bridge.imgmsg_to_cv2(color_image, "bgr8")
        for i, stat in enumerate(detections):
            cv2.circle(frame, (int(stat[0]), int(stat[1])), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"id: {ids[i]}", (int(stat[0]),int(stat[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.depth_pub.publish(image_message)


    def deepsort_algorithm(self, color_image, stats, labels, scores):
        frame = self.bridge.imgmsg_to_cv2(color_image, "bgr8")

        detections = []
        for i, box in enumerate(stats):
            # Append ([x, y, w, h], score, label_string).
            detections.append(
                (
                    [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    scores[i],
                    str(labels[i])
                )
            )

        tracks = self.tracker.update_tracks(detections, frame=frame)
        tracks_ids = []
        track_detections = []
        track_classes = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tracks_ids.append(track.track_id)
            track_detections.append(track.to_ltrb())
            track_classes = track.det_class

        return track_detections, tracks_ids, track_classes
    
    def publish_binarized_image(self, binarized_image):
        binarized_image = cv2.cvtColor(binarized_image, cv2.COLOR_GRAY2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Filtered Components"
        org = (10, 30)
        font_scale = 1
        color = (0, 255, 0)  # Green color
        thickness = 2
        binarized_image = cv2.putText(binarized_image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

        self.color_pub.publish(self.bridge.cv2_to_imgmsg(binarized_image, "bgr8"))

    def position_fruits(self, depth_image, stats, ids, cls):
        """
        - Y axis increasing in [mm] from up to down, up from the middle of the camera will be negative
        - X axis increasing in [mm] from left to right, left from the middle of the camera will be negative
        - Z axis increasing in [mm] from near to far
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(depth_image, depth_image.encoding)

            if not self.intrinsics:
                cv_image_visualization = cv_image.copy()
                cv_image_visualization_8u = (cv_image_visualization / 16).astype(np.uint8)
    
                cv_image_visualization_bgr = cv2.cvtColor(cv_image_visualization_8u, cv2.COLOR_GRAY2BGR)
                image_message = self.bridge.cv2_to_imgmsg(cv_image_visualization_bgr, encoding="bgr8")

                self.depth_pub.publish(image_message)
                return
            
            fruit_positions_list = []

            for i, stat in enumerate(stats):
                center = (int((stat[0] + stat[2]) / 2), int((stat[1] + stat[3]) / 2))
                if 0 <= center[0] < cv_image.shape[1] and 0 <= center[1] < cv_image.shape[0]:
                    depth = cv_image[center[1], center[0]]
                else:
                    continue
                    
                result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [center[0], center[1]], depth)

                fruit_positions_list.append(result)

            return fruit_positions_list, ids

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return
        

    def detectObjectBinarize(self, data, MIN_AREA=20, MAX_AREA=70000):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        color_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LUV)

        # Define the range for red color
        lower_L_bound = np.array([0, 0, 0])
        upper_L_bound = np.array([255, 255, 255])

        # Define the range for green color
        lower_U_bound = np.array([0, 119, 0])
        upper_U_bound = np.array([255, 255, 255])

        # Define the range for yellow color
        lower_V_bound = np.array([0, 0, 138])
        upper_V_bound = np.array([255, 255, 255])

        # Create masks for the colors
        L_mask = cv2.inRange(color_image, lower_L_bound, upper_L_bound)
        U_mask = cv2.inRange(color_image, lower_U_bound, upper_U_bound)
        V_mask = cv2.inRange(color_image, lower_V_bound, upper_V_bound)

        # Combine the masks
        combined_mask = cv2.bitwise_and(L_mask, U_mask)
        combined_mask = cv2.bitwise_and(combined_mask, V_mask)

        # Apply the mask to get the binarized image
        binarized_image = cv2.bitwise_and(self.image, self.image, mask=combined_mask)
        # Binarize the image to have only zeros and ones
        _, binarized_image = cv2.threshold(combined_mask, 1, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_image, connectivity=8)

        bin_image = np.zeros_like(binarized_image)
        detections = []
        cls = []
        scores = []

        # Filter out components that are smaller than 20 pixels
        for i in range(1, num_labels):
            # print("Stats: ", stats[i, cv2.CC_STAT_AREA])
            if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA and stats[i, cv2.CC_STAT_AREA] <= MAX_AREA:
                bin_image[labels == i] = 255
                detections.append([int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH] + stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_HEIGHT] + stats[i, cv2.CC_STAT_TOP])])
                cls.append("Red")
                scores.append(80)

        self.publish_binarized_image(bin_image)

        return detections, cls, scores, bin_image

        
    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return

def main():
    depth_image_topic = '/camera/aligned_depth_to_color/image_raw'
    depth_info_topic = '/camera/aligned_depth_to_color/camera_info'
    color_image_topic = '/camera/color/image_raw'
    
    listener = ImageListener(depth_image_topic, depth_info_topic, color_image_topic)

    ts = message_filters.TimeSynchronizer([listener.color_sub, listener.depth_sub], 10)
    ts.registerCallback(listener.synchronize_callback)
    rospy.spin()

if __name__ == '__main__':
    node_name = os.path.basename(sys.argv[0]).split('.')[0]
    rospy.init_node(node_name)
    main()
