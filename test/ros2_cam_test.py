import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')
        self.subscription = self.create_subscription(
            Image,
            '/kevin/color/image_raw',
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.image = None

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
    def get_image(self):
        while self.image is None:
            rclpy.spin_once(self, timeout_sec=0.01)
        
def main(args=None):
    rclpy.init(args=args)
    node = ImageListener()
    a = node.get_image()
    print(1)
    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()
    #     cv2.destroyAllWindows()

if __name__ == '__main__':
    main()