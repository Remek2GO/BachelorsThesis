#!/usr/bin/env python3

import cv2
import rospy
from sensor_msgs.msg import Image as msg_Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import time

class YOLODetection:
    def __init__(self):
        self.model_path = "/home/jetson/Downloads/best.engine"
        self.color_image_topic = '/camera/color/image_raw'
        self.trt_model = YOLO(self.model_path)
        
        self.bridge = CvBridge()
        self.last_time = time.time()
        self.yolo_execution_time = time.time()
        
        self.frame_count = 0
        self.opencv_tracker = "medianflow"
        self.trackers = cv2.legacy.MultiTracker_create()
        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.legacy.TrackerCSRT_create,
            "kcf": cv2.legacy.TrackerKCF_create,
            "boosting": cv2.legacy.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.legacy.TrackerTLD_create,
            "medianflow": cv2.legacy.TrackerMedianFlow_create,
            "mosse": cv2.legacy.TrackerMOSSE_create
        }

        rospy.Subscriber(self.color_image_topic, msg_Image, self.image_callback)
        self.img_pub = rospy.Publisher('/yolo_detection', msg_Image, queue_size=10)

        

    def image_callback(self, img_msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        frame = cv_image.copy()

        results = None
        if time.time() - self.yolo_execution_time >= 3:
            _, results = self.yolo_detect(cv_image, 0.5)

        frame = self.update_tracker(frame, results)
        
        elapsed_time = time.time() - self.last_time
        self.last_time = time.time()
        fps = 1 / elapsed_time
        # print(f"FPS: {fps:.2f}")

        # Draw FPS on the image
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        
    def update_tracker(self, frame, results):
        success, bbox_tracker = self.trackers.update(frame)
        cv_image = frame.copy()
        # print("BBBOX: ", bbox_tracker)
        if success:
            for i, bbox in enumerate(bbox_tracker):
                x1, y1, w, h = map(int, bbox)
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if results != None:
            self.trackers = cv2.legacy.MultiTracker_create()
            for box in results:
                for box_t in bbox_tracker:
                    # if box do not overlaps with box_t then add it to tracker
                    if not (box.xyxy[0][0] > box_t[0] + box_t[2] or box.xyxy[0][0] + box.xyxy[0][2] < box_t[0] or box.xyxy[0][1] > box_t[1] + box_t[3] or box.xyxy[0][1] + box.xyxy[0][3] < box_t[1]):
                        break
                coordinates = box.xyxy.tolist()[0]
                x1, y1, x2, y2 = map(int, coordinates)
                tracker = self.OPENCV_OBJECT_TRACKERS[self.opencv_tracker]()
                self.trackers.add(tracker, frame, (x1, y1, x2 - x1, y2 - y1))
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return cv_image

    def yolo_detect(self, frame, threshold=0.5):
        # Perform YOLO detection
        results = self.trt_model(frame)
        bbox_filt = []
        self.yolo_execution_time = time.time()
        print("Before: ", len(results))
        print("Results: ", results)

        # Print detection results
        for result in results:
            print("Processing result...")
            if result.boxes:  # Check if there are any detections
                for box in result.boxes:
                    try:
                        # Extract bounding box coordinates
                        coordinates = box.xyxy.tolist()[0]  # Extract the first (and only) sub-list
                        x1, y1, x2, y2 = map(int, coordinates)
                        area = (x2 - x1) * (y2 - y1)
                        for bbox in result.boxes:
                            if not (bbox.xyxy[0][0] > x2 or bbox.xyxy[0][0] + bbox.xyxy[0][2] < x1 or bbox.xyxy[0][1] > y2 or bbox.xyxy[0][1] + bbox.xyxy[0][3] < y1):
                                continue
                        if area > 2000:
                            continue

                        # Extract confidence score and class ID
                        confidence = float(box.conf[0])  # Confidence score
                        if confidence < threshold:
                            continue
                        bbox_filt.append(box)
                        
                        class_id = int(box.cls[0])  # Class ID
                        label = self.trt_model.names[class_id]  # Class name from the model's `names`

                        # Print detection info
                        print(f"Detected: {label}, Confidence: {confidence:.2f}, "
                            f"Bounding Box: ({x1}, {y1}, {x2}, {y2})")

                        # Optional: Draw detections on the image
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {confidence:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Error processing box: {e}")
            else:
                print("No objects detected in this result.")
        print("After: ", len(results))
        return frame, bbox_filt
    
if __name__ == '__main__':
    rospy.init_node('object_sort_node')
    yolo_detection = YOLODetection()
    rospy.spin()