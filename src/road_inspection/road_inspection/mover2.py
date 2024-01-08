import rclpy
import cv2
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

save_directory = 'snapshots'
os.makedirs(save_directory, exist_ok=True)

snapshot_counter = 0

def image_callback(msg):
    global snapshot_counter
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    cv2.imshow("Image", cv_image)
    cv2.waitKey(1)

    key = cv2.waitKey(1) &  0xFF

    if key == ord('s'):
        filename = os.path.join(save_directory, f'snapshot{snapshot_counter}.png')
        cv2.imwrite(filename, cv_image)
        print(f"Snapshot savevd as {filename}")
        snapshot_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("image_viewer")
    subscription = node.create_subscription(Image, '/limo/depth_camera_link/image_raw', image_callback, 10)
    subscription
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()