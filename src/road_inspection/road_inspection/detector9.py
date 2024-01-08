import rclpy
from rclpy.node import Node
from rclpy import qos
import cv2
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from tf2_geometry_msgs import do_transform_pose
from image_geometry import PinholeCameraModel  # Import PinholeCameraModel

font = cv2.FONT_HERSHEY_SIMPLEX

class ObjectDetector(Node):

    object_id_counter = 0
    camera_model = None
    image_depth_ros = None
    visualisation = True
    color2depth_aspect = 1.0
    objects_detected_per_frame = 0
    total_objects_detected = 0

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
        self.object_id_set = set()
        self.origin_visited = True
        self.total_objects_detected = 0

        self.camera_model = PinholeCameraModel()  # Initialize PinholeCameraModel

        # Load the Haar Cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.object_cascade = cv2.CascadeClassifier('path/to/haarcascade_object.xml')


    def assign_object_id(self):
        self.object_id_counter += 1
        return self.object_id_counter

    def search_contours(self, image_color, image_mask):
        self.object_id_counter = 0

        # Convert the color image to grayscale for Haar Cascade detection
        gray_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

         # Detect objects using Haar Cascade
        objects = self.object_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

        # Clear the set to track currently detected objects
        self.object_id_set.clear()

        for (x, y, w, h) in objects:
            # Assign a custom object ID
            object_id = self.assign_object_id()

            # Draw rectangle and display object ID
            cv2.rectangle(image_color, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image_color, f'Object ID: {object_id}', (x, y-10), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Add the object ID to the set to mark it as printed
            self.object_id_set.add(object_id)

            # Print the object location
            print(f'Object Location (2D): {x + w / 2}, {y + h / 2}')

            # Print the size of the detected object
            object_size = w * h
            print(f'Object Size: {object_size} pixels')

            # Publish the exact location of the detected object
            self.publish_object_location((x + w / 2, y + h / 2), image_color.shape, object_id)

        # Print the count in the console
        print('Object Count:', len(objects))

        # Add the count to objects_detected_per_frame
        self.objects_detected_per_frame += len(objects)

         # Add the count to the total_objects_detected
        self.total_objects_detected += len(objects)

        # Check if the origin has been visited
        if not self.origin_visited:
            transform = self.get_tf_transform('odom', 'depth_link')
            if transform and transform.transform.translation.x == 0.0 and transform.transform.translation.y == 0.0:
                print('Origin visited. Stopping further processing.')
                self.origin_visited = True

        # Print the number of objects detected in this frame
        print(f'Objects Detected in this Frame: {self.objects_detected_per_frame}')

    def publish_object_location(self, centroid, image_shape, object_id):
        # Map from image to depth
        depth_coords = (
            image_shape[0] / 2 + (centroid[0] - image_shape[0] / 2) * self.color2depth_aspect,
            image_shape[1] / 2 + (centroid[1] - image_shape[1] / 2) * self.color2depth_aspect,
        )

        # Get the depth reading at the centroid location
        depth_value = self.get_depth_at_coords(depth_coords)

        # Calculate object's 3D location in camera coordinates
        camera_coords = self.camera_model.projectPixelTo3dRay((centroid[1], centroid[0]))
        camera_coords = [x / camera_coords[2] for x in camera_coords]
        camera_coords = [x * depth_value for x in camera_coords]

        print('camera coords: ', camera_coords)

        # Define a point in camera coordinates
        object_location = PoseStamped()
        object_location.header.frame_id = "depth_link"
        object_location.pose.orientation.w = 1.0
        object_location.pose.position.x = camera_coords[0]
        object_location.pose.position.y = camera_coords[1]
        object_location.pose.position.z = camera_coords[2]

        # Publish the object location
        self.object_location_pub.publish(object_location)

        # publish so we can see that in rviz
        self.object_location_pub.publish(object_location)

        # print out the coordinates in the odom frame
        transform = self.get_tf_transform('odom', 'depth_link')
        if transform:
            p_odom = do_transform_pose(object_location.pose, transform)
            print('odom coords: ', p_odom.position)

    def get_depth_at_coords(self, coords):
        # Get the depth reading at the specified coordinates
        depth_value = self.image_depth_ros.data[
            int(coords[0]) + int(coords[1]) * self.image_depth_ros.width
        ]
        return depth_value

    def get_tf_transform(self, target_frame, source_frame):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return transform
        except Exception as e:
            self.get_logger().warning(f"Failed to lookup transform: {str(e)}")
            return None

    def camera_info_callback(self, data):
        if not self.camera_model:
            self.camera_model.fromCameraInfo(data)

    def image_depth_callback(self, data):
        self.image_depth_ros = data

    def image_color_callback(self, data):
        # Wait for camera_model and depth image to arrive
        if self.camera_model is None:
            return

        if self.image_depth_ros is None:
            return

        # Convert images to OpenCV
        try:
            image_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_depth = self.bridge.imgmsg_to_cv2(self.image_depth_ros, "32FC1")
        except CvBridgeError as e:
            print(e)

        # Call the search_contours function for counting
        self.search_contours(image_color, None)  # Pass None for image_mask since we're not using it

        # Resize and adjust for visualization
        image_color = cv2.resize(image_color, (0,0), fx=0.5, fy=0.5)
        image_depth *= 1.0/10.0  # scale for visualization (max range 10.0 m)

        # Display images
        cv2.imshow("image depth", image_depth)
        cv2.imshow("image color", image_color)
        cv2.waitKey(1)

        # Reset the count for the next frame
        self.objects_detected_per_frame = 0

        # Print the sum of total_objects_detected
        print(f'Total Objects Detected: {self.total_objects_detected}')

    def display_total_objects_detected(self):
        # Display the total count of objects detected across all frames
        print(f'Total Objects Detected Across All Frames: {self.total_objects_detected}')

def main(args=None):
    rclpy.init(args=args)
    image_projection = ObjectDetector()
    rclpy.spin(image_projection)

    # Display the total count before shutting down
    image_projection.display_total_objects_detected()

    image_projection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
