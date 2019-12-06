import rospy
from sensor_msgs.msg import Image
import ros_numpy

import numpy as np
import matplotlib.pyplot as plt

import skimage
# import scipy

class ImgProxy:
    def __init__(self):
        self.topic = '/camera/depth/image'
        self.sub = rospy.Subscriber(self.topic, Image, self.callbackImage, queue_size=1)
        self.msg = None
        self.image = None
        self.has_image = True

    def callbackImage(self, msg):
        if not self.has_image:
            self.msg = msg
            self.has_image = True

    def getImage(self):
        self.has_image = False
        while not self.has_image:
            rospy.sleep(0.01)
        self.image = ros_numpy.numpify(self.msg)
        # self.image = self.image[240-100:240+100, 320-100:320+100]
        # self.image[np.isnan(self.image)] = 0
        # self.image = -self.image
        # self.image -= self.image.min()
        return self.image

def main():
    rospy.init_node('image_proxy')
    imgProxy = ImgProxy()
    imgProxy.getImage()

    img = skimage.transform.resize(imgProxy.image, (90, 90))

    plt.imshow(img)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()