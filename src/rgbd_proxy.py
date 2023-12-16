import time

import rospy
from sensor_msgs.msg import Image
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
import ros_numpy
from sklearn.impute import SimpleImputer
from skimage.restoration import inpaint
import numpy as np
import matplotlib.pyplot as plt

import skimage.transform
from skimage.transform import rotate
# import scipy

class RGBD_Proxy:
    def __init__(self):
        self.d_topic = '/depth_to_rgb/image_raw'
        self.d_sub = rospy.Subscriber(self.d_topic, Image, self.callbackD, queue_size=1)
        self.rgb_topic = '/rgb/image_raw'
        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.callbackRGB, queue_size=1)
        self.d_img = None
        self.rgb_img = None
        self.has_d = False
        self.has_rgb = False

    def callbackD(self, msg):
        if not self.has_d:
            self.d_img = msg
            self.has_d = True

    def callbackRGB(self, msg):
        if not self.has_rgb:
            self.rgb_img = msg
            self.has_rgb = True

    def getRGBD(self):
        """
        get cloud in camera frame.
        :return: point cloud, cloud frame, cloud time
        """
        self.has_d, self.has_rgb = False, False
        while not self.has_d or not self.has_rgb:
            rospy.sleep(0.01)
        d_img = ros_numpy.numpify(self.d_img)
        rgb_img = ros_numpy.numpify(self.rgb_img)
        return d_img, rgb_img
        # d_img = ros_numpy.image.
        # pc = ros_numpy.point_cloud2.split_rgb_field(pc)
        # height = pc.shape[0]
        # width = pc.shape[1]
        # cloud = np.zeros((height * width, 6), dtype=np.float32)
        # cloud[:, 0] = np.resize(pc['x'], height * width)
        # cloud[:, 1] = np.resize(pc['y'], height * width)
        # cloud[:, 2] = np.resize(pc['z'], height * width)
        # cloud[:, 3] = np.resize(pc['r'], height * width)
        # cloud[:, 4] = np.resize(pc['g'], height * width)
        # cloud[:, 5] = np.resize(pc['b'], height * width)
        # mask = np.logical_not(np.isnan(cloud).any(axis=1))
        # cloud = cloud[mask]
        # # print("Received Structure cloud with {} points.".format(cloud.shape[0]))
        # return cloud

    def getProjectImg(self, target_size, img_size, return_rgb=False):  #ToDo
        d_img, rgb_img = self.getRGBD()
        view_matrix = np.eye(4)
        view_matrix[:3, 3] = [-0.007, -0.013, 0]
        # augment = np.ones((1, cloud.shape[0]))
        # pts = np.concatenate((cloud.T, augment), axis=0)
        pts = cloud.T
        projection_matrix = np.array([
            [1 / (target_size / 2), 0, 0, 0],
            [0, 1 / (target_size / 2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        tran_world_pix = np.matmul(projection_matrix, view_matrix)
        pts[:2] = np.matmul(tran_world_pix[:2, :2], pts[:2])
        # pts[1] = -pts[1]
        pts[0] = (pts[0] + 1) * img_size / 2
        pts[1] = (pts[1] + 1) * img_size / 2

        pts[0] = np.round_(pts[0])
        pts[1] = np.round_(pts[1])
        mask = (pts[0] >= 0) * (pts[0] < img_size) * (pts[1] > 0) * (pts[1] < img_size)
        pts = pts[:, mask]
        # dense pixel index
        mix_xy = (pts[1].astype(int) * img_size + pts[0].astype(int))
        # lexsort point cloud first on dense pixel index, then on z value
        ind = np.lexsort(np.stack((pts[2], mix_xy)))
        # bin count the points that belongs to each pixel
        bincount = np.bincount(mix_xy)
        # cumulative sum of the bin count. the result indicates the cumulative sum of number of points for all previous pixels
        cumsum = np.cumsum(bincount)
        # rolling the cumsum gives the ind of the first point that belongs to each pixel.
        # because of the lexsort, the first point has the smallest z value
        cumsum = np.roll(cumsum, 1)
        cumsum[0] = bincount[0]
        cumsum[cumsum == np.roll(cumsum, -1)] = 0
        # pad for unobserved pixels
        cumsum = np.concatenate((cumsum, -1 * np.ones(img_size * img_size - cumsum.shape[0]))).astype(int)

        depth = pts[2][ind][cumsum]
        depth[cumsum == 0] = np.nan
        depth = depth.reshape(img_size, img_size)
        mask = np.isnan(depth)
        # depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])
        # imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        # depth = imputer.fit_transform(depth)
        depth = inpaint.inpaint_biharmonic(depth, mask)
        depth = rotate(depth, 90)
        assert depth.shape == (img_size, img_size)

        if return_rgb:
            rgb = pts[3:][:, ind][:, cumsum]
            rgb = rgb.T.reshape(img_size, img_size, 3)
            rgb = rotate(rgb, 90)
            rgb = rgb.transpose(2, 0, 1)
            rgb /= 2550
            rgb -= 0.05
            return depth, rgb
        else:
            return depth

def main():
    rospy.init_node('test')
    cloudProxy = RGBD_Proxy()
    while True:
        obs, rgb = cloudProxy.getProjectImg(0.8, 128*2, return_rgb=True)
        obs = -obs
        obs -= obs.min()
        # obs = skimage.transform.resize(obs, (90, 90))
        plt.figure()
        plt.imshow(obs)
        plt.colorbar()
        plt.figure()
        plt.imshow(rgb.transpose(1, 2, 0).astype(int))
        # plt.imshow(rgb.astype(int))
        plt.show()
        print(1)


if __name__ == '__main__':  #ToDo
    main()