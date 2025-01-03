import cv2
import numpy as np
import rospy
from detection.msg import ObjectPose
from sensor_msgs.msg import Imu
import message_filters
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry

X_SHIFT = -0.14
Y_SHIFT = 0.0
Z_SHIFT = 0.12


class RPY:
    """
    This class is responsible for storing the roll, pitch and yaw angles
    """

    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def as_array(self):
        return np.array([self.roll, self.pitch, self.yaw])


class GlobalPosition:
    """
    This class is responsible for calculating the global position of the detected objects
    """

    def __init__(self):
        # Subscribers
        object_pose_sub = message_filters.Subscriber("/object_pose", ObjectPose)
        imu_sub = message_filters.Subscriber("/camera/imu", Imu)
        global_pose_sub = message_filters.Subscriber(
            "/mavros/global_position/local ", Odometry
        )

        # Publishers
        self.global_pose = rospy.Publisher("/global_pose", ObjectPose, queue_size=10)

        # Synchronize messages
        ts = message_filters.ApproximateTimeSynchronizer(
            [object_pose_sub, imu_sub, global_pose_sub], 10
        )
        ts.registerCallback(self.synchronize_callback)

    def synchronize_callback(self, object_pose, imu, global_pose):

        # Convert quaternion to euler angles
        rpy = self.quaternion2euler(imu)

        # Get rotation matrix based on IMU orientation
        R = self.get_rotation_matrix(rpy)

        # Rotate the position of the detected objects
        for i, pose in enumerate(object_pose.poses):
            rotated_pose = self.rotate_position(pose, R)
            uav_ned = self.get_uav_ned(rotated_pose)

            global_ned = self.get_global_ned(uav_ned, global_pose)

    def get_global_ned(self, uav_ned, global_pose):
        global_object_pose = ObjectPose()

        global_object_pose.header = global_pose.header
        global_object_pose.position.x = (
            uav_ned.position.x + global_pose.pose.pose.position.x
        )
        global_object_pose.position.y = (
            uav_ned.position.y + global_pose.pose.pose.position.y
        )
        global_object_pose.position.z = (
            uav_ned.position.z + global_pose.pose.pose.position.z
        )
        global_object_pose.id = uav_ned.id

        return global_object_pose

    def get_uav_ned(self, pose):
        uav_ned = Pose()
        uav_ned.position.x = pose.position[0] + X_SHIFT
        uav_ned.position.y = pose.position[1] + Y_SHIFT
        uav_ned.position.z = pose.position[2] + Z_SHIFT

        return uav_ned

    def rotate_position(self, pose, R):
        position = np.array(pose.position)

        rotated_position = np.dot(R, position)

        pose.position = rotated_position.tolist()

        return pose

    def get_rotation_matrix(self, rpy):
        r = rpy.roll
        p = rpy.pitch
        y = rpy.yaw

        r_x = np.array(
            [[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]]
        )

        r_y = np.array(
            [[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]]
        )

        r_z = np.array(
            [[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]]
        )

        rotation = np.dot(r_z, np.dot(r_y, r_x))

        return rotation

    def quaternion2euler(self, imu):
        rpy = RPY()

        x = imu.orientation.x
        y = imu.orientation.y
        z = imu.orientation.z
        w = imu.orientation.w
        rpy.roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        rpy.pitch = np.arcsin(2 * (w * y - z * x))
        rpy.yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return rpy


if __name__ == "__main__":
    rospy.init_node("global_pose_node", anonymous=True)

    projection = GlobalPosition()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
