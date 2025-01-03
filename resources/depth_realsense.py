"""
This script uses ROS to process synchronized depth and color images from a Intel Realsense.
It detects objects in the color image and computes the depth at selected points using camera intrinsics.
Results are visualized and published to ROS topics.
"""

#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image as MsgImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import numpy as np
import cv2
import pyrealsense2 as rs2
import message_filters

if not hasattr(rs2, "intrinsics"):
    import pyrealsense2.pyrealsense2 as rs2


class ImageProcessor:
    def __init__(self, depth_topic, depth_info_topic, color_topic):
        """
        Initializes the image processor, setting up ROS subscribers and publishers.

        :param depth_topic: ROS topic for depth images.
        :param depth_info_topic: ROS topic for depth camera info.
        :param color_topic: ROS topic for color images.
        """
        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber(
            depth_topic, MsgImage, self.depth_image_callback
        )
        self.depth_info_sub = rospy.Subscriber(
            depth_info_topic, CameraInfo, self.depth_info_callback
        )
        self.color_sub = rospy.Subscriber(
            color_topic, MsgImage, self.color_image_callback
        )
        self.depth_pub = rospy.Publisher("/bazant/depth", MsgImage, queue_size=10)
        self.color_pub = rospy.Publisher("/bazant/color", MsgImage, queue_size=10)

        self.intrinsics = None
        self.pixel_coords = None

    def synchronized_callback(self, color_image, depth_image):
        """
        Handles synchronized depth and color image data.

        :param color_image: ROS Image message for the color image.
        :param depth_image: ROS Image message for the depth image.
        """
        print("Synchronized callback triggered.")

    def color_image_callback(self, data):
        """
        Processes color image to detect objects and publish results.

        :param data: ROS Image message.
        """
        try:
            processed_image = self.detect_objects(data)
            self.color_pub.publish(self.bridge.cv2_to_imgmsg(processed_image, "bgr8"))
        except CvBridgeError as e:
            print(f"Error processing color image: {e}")

    def depth_image_callback(self, data):
        """
        Processes depth image and visualizes depth at selected points.

        :param data: ROS Image message.
        """
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            pixel = self.pixel_coords

            if not self.intrinsics or not pixel:
                self.visualize_depth_image(depth_image)
                return

            depth = depth_image[pixel[1], pixel[0]]
            coordinates = rs2.rs2_deproject_pixel_to_point(
                self.intrinsics, [pixel[0], pixel[1]], depth
            )
            self.visualize_depth_image(depth_image, pixel, coordinates)
        except (CvBridgeError, ValueError) as e:
            print(f"Error processing depth image: {e}")

    def visualize_depth_image(self, depth_image, pixel=None, coordinates=None):
        """
        Visualizes depth image and overlays information if provided.

        :param depth_image: Depth image in CV2 format.
        :param pixel: Tuple of (x, y) pixel coordinates.
        :param coordinates: 3D coordinates corresponding to the pixel.
        """
        vis_image = (depth_image / 16).astype(np.uint8)
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        if pixel and coordinates:
            cv2.circle(vis_image_bgr, (pixel[0], pixel[1]), 10, (255, 0, 0), -1)
            cv2.putText(
                vis_image_bgr,
                f"X: {coordinates[0]}, Y: {coordinates[1]}, Z: {coordinates[2]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        self.depth_pub.publish(self.bridge.cv2_to_imgmsg(vis_image_bgr, "bgr8"))

    def detect_objects(self, data):
        """
        Detects objects in a color image based on color thresholds and publishes the results.

        Y axis increasing in [mm] from up to down, up from the middle of the camera will be negative
        X axis increasing in [mm] from left to right, left from the middle of the camera will be negative
        Z axis increasing in [mm] from near to far

        :param data: ROS Image message for the color image.
        :return: Processed image with detected objects highlighted.
        """
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        luv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)

        lower_bounds = [
            np.array([0, 0, 0]),
            np.array([0, 119, 0]),
            np.array([0, 0, 138]),
        ]
        upper_bounds = [np.array([255, 255, 255])] * 3

        masks = [
            cv2.inRange(luv_image, lower_bounds[i], upper_bounds[i]) for i in range(3)
        ]
        combined_mask = cv2.bitwise_and(masks[0], cv2.bitwise_and(masks[1], masks[2]))

        _, binarized_image = cv2.threshold(combined_mask, 1, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binarized_image, connectivity=8
        )

        filtered_image = np.zeros_like(binarized_image)
        for i in range(1, num_labels):
            if 300 <= stats[i, cv2.CC_STAT_AREA] <= 70000:
                filtered_image[labels == i] = 255
                self.pixel_coords = (int(centroids[i][0]), int(centroids[i][1]))

        filtered_image_bgr = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            filtered_image_bgr,
            "Filtered Components",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        return filtered_image_bgr

    def depth_info_callback(self, camera_info):
        """
        Sets up camera intrinsics from the provided camera info message.

        :param camera_info: ROS CameraInfo message.
        """
        if self.intrinsics:
            return

        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = camera_info.width
        self.intrinsics.height = camera_info.height
        self.intrinsics.ppx = camera_info.K[2]
        self.intrinsics.ppy = camera_info.K[5]
        self.intrinsics.fx = camera_info.K[0]
        self.intrinsics.fy = camera_info.K[4]
        self.intrinsics.model = (
            rs2.distortion.brown_conrady
            if camera_info.distortion_model == "plumb_bob"
            else rs2.distortion.kannala_brandt4
        )
        self.intrinsics.coeffs = list(camera_info.D)


def main():
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    depth_info_topic = "/camera/aligned_depth_to_color/camera_info"
    color_topic = "/camera/color/image_raw"

    processor = ImageProcessor(depth_topic, depth_info_topic, color_topic)
    ts = message_filters.TimeSynchronizer(
        [processor.color_sub, processor.depth_sub], 10
    )
    ts.registerCallback(processor.synchronized_callback)
    rospy.spin()


if __name__ == "__main__":
    rospy.init_node(os.path.splitext(os.path.basename(sys.argv[0]))[0])
    main()
