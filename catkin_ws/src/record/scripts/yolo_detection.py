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
        self.trt_model = YOLO(self.model_path)
        self.color_image_topic = '/camera/color/image_raw'
        self.bridge = CvBridge()
        self.last_time = time.time()
        self.frame_count = 0

        rospy.Subscriber(self.color_image_topic, msg_Image, self.image_callback)
        self.img_pub = rospy.Publisher('/yolo_detection', msg_Image, queue_size=10)

    def image_callback(self, img_msg):
        print("Received image")
        # Calculate and display FPS
        self.last_time = time.time()
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        # Perform YOLO detection
        results = self.trt_model(cv_image)

        # Print detection results
        for result in results:
            print("Processing result...")
            if result.boxes:  # Check if there are any detections
                for box in result.boxes:
                    try:
                        # Extract bounding box coordinates
                        coordinates = box.xyxy.tolist()[0]  # Extract the first (and only) sub-list
                        x1, y1, x2, y2 = map(int, coordinates)

                        # Extract confidence score and class ID
                        confidence = float(box.conf[0])  # Confidence score
                        if confidence < 0.5:
                            continue
                        class_id = int(box.cls[0])  # Class ID
                        label = self.trt_model.names[class_id]  # Class name from the model's `names`

                        # Print detection info
                        print(f"Detected: {label}, Confidence: {confidence:.2f}, "
                            f"Bounding Box: ({x1}, {y1}, {x2}, {y2})")

                        # Optional: Draw detections on the image
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(cv_image, f"{label} {confidence:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Error processing box: {e}")
            else:
                print("No objects detected in this result.")
        
        elapsed_time = time.time() - self.last_time
        fps = 1 / elapsed_time
        print(f"FPS: {fps:.2f}")

        # Draw FPS on the image
        cv2.putText(cv_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Display the image

        self.img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        # cv2.imshow("YOLO Detection", cv_image)
        # cv2.waitKey(1)
        

if __name__ == '__main__':
    rospy.init_node('yolo_detection_node')
    yolo_detection = YOLODetection()
    rospy.spin()