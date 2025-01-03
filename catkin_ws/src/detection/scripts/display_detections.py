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
from detection.msg import DetectionObjects
from ultralytics import YOLO

print(cv2.__version__)

SCALE = 1

class SORT_display:
    def __init__(self):
        # Load YOLO model and tracker
        self.trt_model = YOLO("/home/jetson/bazant_project/YOLO/models/fuse_19_12_s/yolo11s_half.engine")
        # self.trt_model = YOLO("/home/jetson/Downloads/yolo11n_half.engine")

        # Publishers
        self.detected_object_pub = rospy.Publisher("/detected_object", DetectionObjects, queue_size=10)
        self.tracking_image_pub = rospy.Publisher("/camera/tracking_objects", Image, queue_size=10)
        
        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        self.tracker = tracker_lib.Tracker()
        self.frame = None
        self.bridge = CvBridge()  # Bridge to convert ROS Image to OpenCV

        # Start YOLO thread for detection
        threading.Thread(target=self.thread_yolo, daemon=True).start()


    def thread_yolo(self):
        rate = rospy.Rate(5)  # 10 Hz
        while not rospy.is_shutdown():
            if self.frame is not None:
                frame_yolo = self.frame.copy()
                results = self.trt_model(frame_yolo)  # Run inference on the frame

                # Initialize lists to store detection data
                labels = []
                boxes = []
                confidences = []

                # Process the results
                for result in results:
                    if result.boxes:
                        for box in result.boxes:
                            coordinates = box.xyxy.tolist()[0]
                            x1, y1, x2, y2 = map(int, coordinates)
                            confidence = int(box.conf[0]*100)
                            area = (x2 - x1) * (y2 - y1)

                            if confidence < 0.45:
                                continue
                            
                            if area > 1000:
                                continue


                            class_id = int(box.cls[0])
                            label = self.trt_model.names[class_id]
                            
                            boxes.append([int(x1/SCALE), int(y1/SCALE), int(x2/SCALE), int(y2/SCALE)])
                            confidences.append(confidence)
                            labels.append(label)

                # Perform tracking if there are valid detections
                if len(labels) == len(boxes) == len(confidences):
                    scaled_frame = cv2.resize(frame_yolo, None, fx=1/SCALE, fy=1/SCALE)
                    self.tracker.match_detections(scaled_frame, boxes, labels, confidences)

            rate.sleep()

    def image_callback(self, data):
        self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.display_frame(self.frame, data.header)

    def display_frame(self, frame, frame_stamp):
        scaled_frame = cv2.resize(frame, None, fx=1/SCALE, fy=1/SCALE)
        publish_data = DetectionObjects()
        ids = []
        labels = []
        bboxes = []

        detections = list(self.tracker.get_detections(scaled_frame))
        if len(detections) != 0:
            for id, bbox, label in detections:
                x0, y0, x1, y1 = [b * SCALE for b in bbox]
                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2, 1)
                text = "{}:{}".format(id, label)
                labelSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                p1_ = (x0 + 2, y0 + labelSize[1] + 2)
                cv2.putText(frame, text, p1_, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                
                ids.append(int(id))
                labels.append(label)
                bboxes.append([x0, y0, x1, y1])

            bboxes = [item for sublist in bboxes for item in sublist]
            bboxes_data = self.create_2d_array(bboxes, len(bboxes), 4)
            publish_data.header = frame_stamp
            publish_data.ids = ids
            # publish_data.labels = labels
            publish_data.bboxes = bboxes_data
        
            self.detected_object_pub.publish(publish_data)
            
        self.tracking_image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))


    def create_2d_array(self, data, rows, cols):
        """Helper function to create a Float32MultiArray for a 2D array."""
        array = Float32MultiArray()
        array.layout.dim = [
            MultiArrayDimension(label="rows", size=rows, stride=rows * cols),
            MultiArrayDimension(label="cols", size=cols, stride=cols)
        ]
        array.data = data
        return array

if __name__ == "__main__":
    rospy.init_node('sort_display_node', anonymous=True)

    sort = SORT_display()
    
    

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
