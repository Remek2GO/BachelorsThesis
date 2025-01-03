#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import cv2
from cv_bridge import CvBridge, CvBridgeError
from detection.msg import BoundingBoxes, BoundingBox, ObjectPose
import numpy as np
import pyrealsense2 as rs2
import message_filters
if not hasattr(rs2, 'intrinsics'):
    import pyrealsense2.pyrealsense2 as rs2


class ProjectionListener:
    def __init__(self):

        # Subscribers
        detections_sub = message_filters.Subscriber("/detected_objects", BoundingBoxes)
        depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.depth_info_callback)

        # Publishers
        self.object_pose = rospy.Publisher("/object_pose", ObjectPose, queue_size=10)

        # Synchronize messages
        ts = message_filters.ApproximateTimeSynchronizer([detections_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.synchronize_callback)

        self.bridge = CvBridge()
        self.intrinsics = None

    def synchronize_callback(self, detected_objects, depth_image):
        object_positions = ObjectPose()
        ids = []
        poses = []

        for bbox in detected_objects.bounding_boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax

            # Compute the fruit position
            fruit_pos = self.position_fruits(depth_image, [x1, y1, x2, y2])
            if fruit_pos:
                ids.append(bbox.id)
                poses.append(fruit_pos)

        # Populate and publish ObjectPose message
        object_positions.header = detected_objects.header
        object_positions.ids = ids
        object_positions.poses = poses
        self.object_pose.publish(object_positions)

    def position_fruits(self, depth_image, bbox):
        """
        - Y axis increasing in [mm] from up to down, up from the middle of the camera will be negative
        - X axis increasing in [mm] from left to right, left from the middle of the camera will be negative
        - Z axis increasing in [mm] from near to far
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(depth_image, depth_image.encoding)

            if not self.intrinsics:
                return None

            # Compute center of the bounding box
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            if 0 <= center[0] < cv_image.shape[1] and 0 <= center[1] < cv_image.shape[0]:
                depth = cv_image[center[1], center[0]]
                # Skip invalid depth values
                if depth == 0 or np.isnan(depth):
                    return None

            # Project pixel to 3D point
            result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [center[0], center[1]], depth)
            return result

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return None
        except ValueError as e:
            rospy.logerr(f"Value Error: {e}")
            return None

    def depth_info_callback(self, camera_info):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = camera_info.width
            self.intrinsics.height = camera_info.height
            self.intrinsics.ppx = camera_info.K[2]
            self.intrinsics.ppy = camera_info.K[5]
            self.intrinsics.fx = camera_info.K[0]
            self.intrinsics.fy = camera_info.K[4]
            if camera_info.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif camera_info.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in camera_info.D]
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

if __name__ == "__main__":
    rospy.init_node('projection_node', anonymous=True)

    projection = ProjectionListener()
 
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
