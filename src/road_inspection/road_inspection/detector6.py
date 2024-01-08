import rclpy
from rclpy.node import Node
from rclpy import qos
import numpy as np

# OpenCV
import cv2
from tf2_ros import Buffer, TransformListener

# ROS libraries
import image_geometry
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from tf2_geometry_msgs import do_transform_pose

font = cv2.FONT_HERSHEY_SIMPLEX

class ObjectDetector(Node):

    object_id_counter = 0  # Class variable for persistent IDs
    camera_model = None
    image_depth_ros = None
    visualisation = True
    color2depth_aspect = 1.0  # for a simulated camera
    total_objects_detected = 0  # Class variable for total objects detected

    def __init__(self):    
        super().__init__('image_projection_3')
        self.bridge = CvBridge()

        self.camera_info_sub = self.create_subscription(CameraInfo, '/limo/depth_camera_link/camera_info',
                                                self.camera_info_callback, 
                                                qos_profile=qos.qos_profile_sensor_data)
        
        self.object_location_pub = self.create_publisher(PoseStamped, '/limo/object_location', 10)

        self.image_sub = self.create_subscription(Image, '/limo/depth_camera_link/image_raw', 
                                                  self.image_color_callback, qos_profile=qos.qos_profile_sensor_data)
        
        self.image_sub = self.create_subscription(Image, '/limo/depth_camera_link/depth/image_raw', 
                                                  self.image_depth_callback, qos_profile=qos.qos_profile_sensor_data)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.object_id_set = set()  # Set to keep track of detected object IDs

    def assign_object_id(self):
        # Increment the object ID counter
        self.object_id_counter += 1
        return self.object_id_counter

    def search_contours(self, image_color, image_mask):
        contours, hierarchy = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_count = 0

        # Clear the set to track currently detected objects
        self.object_id_set.clear()

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            # Assign a custom object ID
            object_id = self.assign_object_id()

            # Print and draw the object ID
            print(f'Object ID: {object_id}')
            cv2.drawContours(image_color, [contour], -1, (0, 255, 0), 2)
            cv2.putText(image_color, f'Object ID: {object_id}', (10, 60 + contours_count * 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Add the object ID to the set to mark it as printed
            self.object_id_set.add(object_id)

            # Print the object location
            print(f'Object Location (2D): {M["m10"]/M["m00"]}, {M["m01"]/M["m00"]}')

            contours_count += 1

        # Print the count in the console
        print('Object Count:', contours_count)

        # Add the count to the total_objects_detected
        self.total_objects_detected += contours_count

    def get_tf_transform(self, target_frame, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return transform
        except Exception as e:
            self.get_logger().warning(f"Failed to lookup transform: {str(e)}")
            return None

    def camera_info_callback(self, data):
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(data)

    def image_depth_callback(self, data):
        self.image_depth_ros = data

    def image_color_callback(self, data):
       
        # wait for camera_model and depth image to arrive
        if self.camera_model is None:
            return

        if self.image_depth_ros is None:
            return

        # covert images to open_cv
        try:
            image_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_depth = self.bridge.imgmsg_to_cv2(self.image_depth_ros, "32FC1")
        except CvBridgeError as e:
            print(e)

        # detect a color blob in the color image
        hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([140, 50, 150])
        upper_pink = np.array([180, 255, 255])
        image_mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # Call the search_contours function for counting
        self.search_contours(image_color, image_mask)

        # ... rest of the code
        # ... (previous code)

        # calculate moments of the binary image
        M = cv2.moments(image_mask)

        if M["m00"] == 0:
            print('No object detected.')
            return

        # calculate the y,x centroid
        image_coords = (M["m01"] / M["m00"], M["m10"] / M["m00"])
        # "map" from color to depth image
        depth_coords = (image_depth.shape[0]/2 + (image_coords[0] - image_color.shape[0]/2)*self.color2depth_aspect, 
            image_depth.shape[1]/2 + (image_coords[1] - image_color.shape[1]/2)*self.color2depth_aspect)
        # get the depth reading at the centroid location
        depth_value = image_depth[int(depth_coords[0]), int(depth_coords[1])]  # you might need to do some boundary checking first!

        #print('image coords: ', image_coords)
        #print('depth coords: ', depth_coords)
        #print('depth value: ', depth_value)

        # calculate object's 3d location in camera coords
        camera_coords = self.camera_model.projectPixelTo3dRay((image_coords[1], image_coords[0]))  # project the image coords (x,y) into 3D ray in camera coords 
        camera_coords = [x/camera_coords[2] for x in camera_coords]  # adjust the resulting vector so that z = 1
        camera_coords = [x*depth_value for x in camera_coords]  # multiply the vector by depth

        print('camera coords: ', camera_coords)

        # define a point in camera coordinates
        object_location = PoseStamped()
        object_location.header.frame_id = "depth_link"
        object_location.pose.orientation.w = 1.0
        object_location.pose.position.x = camera_coords[0]
        object_location.pose.position.y = camera_coords[1]
        object_location.pose.position.z = camera_coords[2]

        # publish so we can see that in rviz
        self.object_location_pub.publish(object_location)        

        # print out the coordinates in the odom frame
        transform = self.get_tf_transform('odom', 'depth_link')
        if transform:
            p_odom = do_transform_pose(object_location.pose, transform)
            print('odom coords: ', p_odom.position)

        if self.visualisation:
            # draw circles
            cv2.circle(image_color, (int(image_coords[1]), int(image_coords[0])), 10, 255, -1)
            cv2.circle(image_depth, (int(depth_coords[1]), int(depth_coords[0])), 5, 255, -1)

            # resize and adjust for visualisation
            image_color = cv2.resize(image_color, (0,0), fx=0.5, fy=0.5)
            image_depth *= 1.0/10.0  # scale for visualisation (max range 10.0 m)

            cv2.imshow("image depth", image_depth)
            cv2.imshow("image color", image_color)
            cv2.waitKey(1)

        # Print the sum of total_objects_detected
        print(f'Total Objects Detected: {self.total_objects_detected}')

def main(args=None):
    rclpy.init(args=args)
    image_projection = ObjectDetector()
    rclpy.spin(image_projection)
    image_projection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
