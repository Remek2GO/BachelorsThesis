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


class YOLODetection:
    def __init__(self):
        # Load YOLO model and tracker
        self.trt_model = YOLO("/home/jetson/bazant_project/YOLO/models/fuse_19_12_s/yolo11s_half.engine")
        # self.trt_model = YOLO("/home/jetson/Downloads/yolo11n_int.engine")

        # Publishers
        self.detected_objects = rospy.Publisher("/detected_objects", BoundingBoxes, queue_size=10)
        self.yolo_frame = rospy.Publisher("/yolo_frame", Image, queue_size=10)
        
        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        self.tracker = tracker_lib.Tracker()
        self.frame = None
        self.bridge = CvBridge()  # Bridge to convert ROS Image to OpenCV

        self.frame_counter = 0 

    def run_yolo(self, data):
        if self.frame is not None:
            frame_yolo = self.frame.copy()
            results = self.trt_model(self.frame)  # Run inference on the frame

            # Initialize lists to store detection data
            labels = []
            boxes = []
            confidences = []
            bbox_results = BoundingBoxes()
            bbox_results.header = data.header

            # Process the results
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        coordinates = box.xyxy.tolist()[0]
                        x1, y1, x2, y2 = map(int, coordinates)
                        confidence = float(box.conf[0])
                        area = (x2 - x1) * (y2 - y1)

                        if confidence < 0.35:
                            continue
                        
                        if area < 1000:
                            continue


                        class_id = int(box.cls[0])
                        label = self.trt_model.names[class_id]

                        bbox_det = BoundingBox()

                        bbox_det.xmin = x1
                        bbox_det.ymin = y1
                        bbox_det.xmax = x2
                        bbox_det.ymax = y2
                        bbox_det.probability = confidence

                        bbox_results.bounding_boxes.append(bbox_det)

                        boxes.append([x1, y1, x2, y2])
                        confidences.append(confidence)
                        labels.append(label)
                        # Draw bounding box
                        cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw label and confidence
                        text = f"{label}: {confidence:.2f}"
                        cv2.putText(frame_yolo, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Perform tracking if there are valid detections
            if len(labels) == len(boxes) == len(confidences) > 0:
                self.detected_objects.publish(bbox_results)
                self.yolo_frame.publish(self.bridge.cv2_to_imgmsg(frame_yolo, "bgr8"))


    def image_callback(self, data):
        self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.frame_counter += 1

        if self.frame_counter % 5 == 0:
            self.run_yolo(data)

    def display_frame(self, frame, frame_stamp):
        publish_data = DetectionObjects()
        ids = []
        labels = []
        bboxes = []

        detections = self.tracker.get_detections(frame)
        if len(list(detections)) != 0:
            for id, bbox, label in self.tracker.get_detections(frame):
                x0, y0, x1, y1 = [b  for b in bbox]
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
        
            self.detected_object.publish(publish_data)
            

        key = cv2.waitKey(1)
        if key == ord("q"):
            rospy.signal_shutdown("User requested shutdown")


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
    rospy.init_node('detection_node', anonymous=True)

    yolo = YOLODetection()    

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
