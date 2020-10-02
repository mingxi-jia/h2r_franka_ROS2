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

    def getImage(self, iteration=10):
        images = []
        for _ in range(iteration):
            self.has_image = False
            while not self.has_image:
                rospy.sleep(0.01)
            img = ros_numpy.numpify(self.msg)
            mask = np.isnan(img)
            img[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), img[~mask])
            images.append(img)
        self.image = np.mean(images, axis=0)
        # self.image = self.image[240-100:240+100, 320-100:320+100]
        # self.image[np.isnan(self.image)] = 0
        # self.image = -self.image
        # self.image -= self.image.min()
        return self.image

def main():
    rospy.init_node('image_proxy')
    imgProxy = ImgProxy()
    while True:
        obs = imgProxy.getImage()
        obs = obs[240 - 100:240 + 100, 320 - 100:320 + 100]
        obs[np.isnan(obs)] = 0
        obs = -obs
        obs -= obs.min()
        obs = skimage.transform.resize(obs, (90, 90))
        obs = obs[45-5:45+5, 45-5:45+5]
        plt.imshow(obs)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    main()