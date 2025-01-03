#!/usr/bin/env python

import os, sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import threading
import time
from scripts.utils import tracker_lib
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from detection.msg import DetectionObjects, BoundingBox, BoundingBoxes
from ultralytics import YOLO


class BOTSortTracker:
    def __init__(self):
        # Load YOLO model and tracker
        self.trt_model = YOLO("/home/jetson/bazant_project/YOLO/models/fuse_19_12_s/yolo11s_half.engine")
        # self.trt_model = YOLO("/home/jetson/Downloads/yolo11n_int.engine")

        # Publishers
        self.yolo_frame = rospy.Publisher("/yolo_frame", Image, queue_size=10)
        
        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        self.bridge = CvBridge()  # Bridge to convert ROS Image to OpenCV


    def run_yolo(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

        results = self.trt_model.track(frame, persist=True, conf=0.45)
        annotated_frame = results[0].plot()
        # annotated_frame = frame.copy()

        # Calculate FPS
        if not hasattr(self, 'last_time'):
            self.last_time = time.time()
            self.fps = 0
        else:
            current_time = time.time()
            self.fps = 1.0 / (current_time - self.last_time)
            self.last_time = current_time

        # Calculate average FPS over the last 10 frames
        if not hasattr(self, 'fps_list'):
            self.fps_list = []
        
        self.fps_list.append(self.fps)
        if len(self.fps_list) > 10:
            self.fps_list.pop(0)
        
        avg_fps = sum(self.fps_list) / len(self.fps_list)

        # Draw average FPS on the frame
        cv2.putText(annotated_frame, f"Avg FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.yolo_frame.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))


    def image_callback(self, data):
        self.run_yolo(data)


if __name__ == "__main__":
    rospy.init_node('botsort_node', anonymous=True)

    yolo = BOTSortTracker()    

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
