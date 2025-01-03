"""
This script saves images from a ROS topic to a specified directory. 
Users can trigger image saving by pressing 's'.
"""

#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os


class ImageSaver:
    def __init__(self):
        """
        Initializes the ImageSaver class, setting up a ROS subscriber and a save directory.
        """
        self.bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_callback
        )
        self.current_image = None
        self.save_directory = "/home/remek2go/bazant_ws/.images/"

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def image_callback(self, data):
        """
        Callback function to process incoming image messages.

        :param data: ROS Image message.
        """
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def save_image(self):
        """
        Saves the latest received image to the specified directory.
        """
        if self.current_image is not None:
            image_path = os.path.join(self.save_directory, "saved_image.jpg")
            cv2.imwrite(image_path, self.current_image)
            rospy.loginfo(f"Image saved to {image_path}")
        else:
            rospy.logwarn("No image received yet!")


if __name__ == "__main__":
    rospy.init_node("image_saver", anonymous=True)
    saver = ImageSaver()
    rospy.loginfo("Press 's' to save the image.")

    while not rospy.is_shutdown():
        key = input("Enter 's' to save an image or 'q' to quit: ")
        if key.lower() == "s":
            saver.save_image()
        elif key.lower() == "q":
            rospy.loginfo("Exiting ImageSaver.")
            break
