#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import time

class ImageToVideoSaver:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('image_to_video_saver', anonymous=True)

        # Subscribe to the /camera/color/image_raw topic
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Video writer parameters
        self.frame_width = 1280      # Adjust based on your camera's resolution
        self.frame_height = 720      # Adjust based on your camera's resolution
        self.fps = 30                # Frames per second (desired)
        self.output_file = 'output.mp4'  # Output video file name

        # Initialize VideoWriter with the MP4 codec
        self.video_writer = cv2.VideoWriter(
            self.output_file,
            cv2.VideoWriter_fourcc(*'avc1'),  # Codec for MP4 format
            self.fps,
            (self.frame_width, self.frame_height)
        )

        # Flag to track if the first frame is received
        self.frame_received = False

        # Track the previous timestamp
        self.previous_time = None

        rospy.loginfo("Video recording started... Press Ctrl+C to stop.")

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Resize the image if it doesn't match the expected frame size
            cv_image_resized = cv2.resize(cv_image, (self.frame_width, self.frame_height))

            # Get the current frame timestamp
            current_time = msg.header.stamp.to_sec()

            if self.previous_time is not None:
                # Calculate the delay needed to maintain the desired frame rate
                expected_interval = 1.0 / self.fps
                actual_interval = current_time - self.previous_time

                # Sleep if the actual interval is shorter than the expected interval
                if actual_interval < expected_interval:
                    time.sleep(expected_interval - actual_interval)

            # Write the frame to the video file
            self.video_writer.write(cv_image_resized)

            self.frame_received = True
            self.previous_time = current_time

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def cleanup(self):
        # Release the video writer when the node is stopped
        self.video_writer.release()
        if self.frame_received:
            rospy.loginfo(f"Video saved as {self.output_file}")
        else:
            rospy.loginfo("No frames received. Video file not created.")

if __name__ == '__main__':
    try:
        image_to_video_saver = ImageToVideoSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        image_to_video_saver.cleanup()
