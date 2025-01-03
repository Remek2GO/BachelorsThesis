import rospy
from mavros_msgs.msg import RCIn
import subprocess
import datetime

class UserInterface:
    def __init__(self):
        rospy.init_node('record_rosbag_node', anonymous=True)
        self.recording_process = None
        self.topics_to_record= [
    		"/camera/color/image_raw",
    		"/camera/color/camera_info",
    		"/camera/imu",
    		"/camera/aligned_depth_to_color/image_raw",
    		"/camera/aligned_depth_to_color/camera_info",
    		"/mavros/imu/data",
		    "/mavros/global_position/local",
		    "/mavros/local_position/pose",
		    "/camera/infra1/image_rect_raw",
		    "/camera/infra1/camera_info",
		    "/camera/infra2/image_rect_raw",
		    "/camera/infra2/camera_info"]
        self.bag_dir = "/home/jetson/bags/"
        rospy.Subscriber('/mavros/rc/in', RCIn, self.rc_in_callback)

    def rc_in_callback(self, msg):
        channel_10_value = msg.channels[6]  # Channel 7 is at index 6
        if channel_10_value >= 2000:
            if self.recording_process is None:
                self.start_recording()
        else:
            if self.recording_process is not None:
                self.stop_recording()

    def start_recording(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_name = f"{self.bag_dir}rosbag_{timestamp}.bag"
        self.recording_process = subprocess.Popen(['rosbag', 'record', '-O', bag_name] + self.topics_to_record)
        # self.recording_process = subprocess.Popen(['rosbag', 'record', '-O', bag_name, '-a'])
        rospy.loginfo(f"Started recording rosbag: {bag_name}")

    def stop_recording(self):
        self.recording_process.terminate()
        self.recording_process.wait()
        self.recording_process = None
        rospy.loginfo("Stopped recording rosbag")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = UserInterface()
        node.run()
    except rospy.ROSInterruptException:
        pass
