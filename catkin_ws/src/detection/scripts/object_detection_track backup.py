#!/usr/bin/env python

import os
import sys
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from detection.msg import DetectionObjects, BoundingBox, BoundingBoxes
from ultralytics import YOLO
from collections import deque

class TrackerObject:
    def __init__(self, tracker, bbox, label):
        self.tracker = tracker
        self.bbox = bbox
        self.label = label
        self.frames_since_update = 0

class YOLODetection:
    def __init__(self):
        self.trt_model = YOLO("/home/jetson/bazant_project/YOLO/models/fuse_19_12_s/yolo11s_half.engine")
        self.detected_objects = rospy.Publisher("/detected_objects", BoundingBoxes, queue_size=10)
        self.yolo_frame = rospy.Publisher("/yolo_frame", Image, queue_size=10)
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        self.bridge = CvBridge()
        self.frame = None
        self.trackers = {}  # Dictionary of active trackers
        self.next_tracker_id = 0
        self.iou_threshold = 0.001
        self.max_frames_without_update = 30

        self.frame_counter = 0

    def image_callback(self, data):
        self.frame_counter += 1

        self.run_tracking()

        if self.frame_counter % 5 == 0:
            self.run_yolo()

    def run_yolo(self):
        if self.frame is None:
            return

        results = self.trt_model(self.frame)

        detections = []
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    coordinates = box.xyxy.tolist()[0]
                    x1, y1, x2, y2 = map(int, coordinates)
                    confidence = float(box.conf[0])
                    if confidence < 0.35:
                        continue

                    class_id = int(box.cls[0])
                    label = self.trt_model.names[class_id]
                    detections.append((x1, y1, x2, y2, label))

        self.update_trackers(detections)

    def update_trackers(self, detections):
        for x1, y1, x2, y2, label in detections:
            matched = False
            for tracker_id, tracker_obj in self.trackers.items():
                iou = self.calculate_iou((x1, y1, x2, y2), tracker_obj.bbox)
                # size_similarity = self.calculate_size_similarity((x1, y1, x2, y2), tracker_obj.bbox)
                print(f"X1, Y1, X2, Y2: {x1}, {y1}, {x2}, {y2}")
                print(f"Tracker Bbox: {tracker_obj.bbox}")
                print(f"Tracker ID: {tracker_id}, IOU: {iou}")
                print("_____________")

                if iou > self.iou_threshold:  #and size_similarity > 0.8:
                    tracker_obj.frames_since_update = 0
                    tracker_obj.bbox = (x1, y1, x2, y2)
                    tracker = cv2.legacy.TrackerMedianFlow_create()
                    tracker.init(self.frame, (x1, y1, x2 - x1, y2 - y1))
                    tracker_obj.tracker = tracker
                    matched = True
                    break

            if not matched:
                tracker = cv2.legacy.TrackerMedianFlow_create()
                tracker.init(self.frame, (x1, y1, x2 - x1, y2 - y1))
                self.trackers[self.next_tracker_id] = TrackerObject(tracker, (x1, y1, x2, y2), label)
                self.next_tracker_id += 1

    def run_tracking(self):
        frame_tracking = self.frame.copy()
        bbox_results = BoundingBoxes()

        print(f"Number of trackers: {len(self.trackers)}")
        for tracker_id, tracker_obj in list(self.trackers.items()):
            success, bbox = tracker_obj.tracker.update(frame_tracking)
            if not success or tracker_obj.frames_since_update > self.max_frames_without_update:
                del self.trackers[tracker_id]
                continue

            # Convert tracker bbox (x, y, w, h) -> (x1, y1, x2, y2)
            x1, y1, w, h = map(int, bbox)
            x2, y2 = x1 + w, y1 + h

            tracker_obj.bbox = (x1, y1, x2, y2)  # Update bbox in consistent format
            tracker_obj.frames_since_update += 1

            bbox_msg = BoundingBox(
                xmin=x1, ymin=y1, xmax=x2, ymax=y2, probability=1.0
            )
            bbox_results.bounding_boxes.append(bbox_msg)

            cv2.rectangle(frame_tracking, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame_tracking, f"{tracker_obj.label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        self.detected_objects.publish(bbox_results)
        self.yolo_frame.publish(self.bridge.cv2_to_imgmsg(frame_tracking, "bgr8"))


    @staticmethod
    def calculate_iou(boxA, boxB):
        # Ensure the format is (x1, y1, x2, y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


    @staticmethod
    def calculate_size_similarity(boxA, boxB):
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return min(areaA, areaB) / max(areaA, areaB)

if __name__ == "__main__":
    rospy.init_node('detection_node', anonymous=True)
    yolo = YOLODetection()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
