import time

import rospy
from sensor_msgs.msg import Image
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
import ros_numpy
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt

import skimage.transform
# import scipy

class CloudProxy:
    def __init__(self):
        self.topic = '/camera/depth/points'
        self.sub = rospy.Subscriber(self.topic, PointCloud2, self.callbackCloud, queue_size=1)
        self.msg = None
        self.image = None
        self.has_cloud = False

    def callbackCloud(self, msg):
        if not self.has_cloud:
            self.msg = msg
            self.has_cloud = True

    def getCloud(self):
        """
        get cloud in camera frame.
        :return: point cloud, cloud frame, cloud time
        """
        self.has_cloud = False
        while not self.has_cloud:
            rospy.sleep(0.01)
        cloudTime = self.msg.header.stamp
        cloudFrame = self.msg.header.frame_id
        # cloud = np.array(list(point_cloud2.read_points(self.msg)))[:, 0:3]
        pc = ros_numpy.numpify(self.msg)
        height = pc.shape[0]
        width = pc.shape[1]
        cloud = np.zeros((height * width, 3), dtype=np.float32)
        cloud[:, 0] = np.resize(pc['x'], height * width)
        cloud[:, 1] = np.resize(pc['y'], height * width)
        cloud[:, 2] = np.resize(pc['z'], height * width)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        # print("Received Structure cloud with {} points.".format(cloud.shape[0]))
        return cloud

    def getProjectImg(self, target_size, img_size):
        cloud = self.getCloud()
        view_matrix = np.eye(4)
        view_matrix[:3, 3] = [-0.007, -0.013, 0]
        augment = np.ones((1, cloud.shape[0]))
        pts = np.concatenate((cloud.T, augment), axis=0)
        projection_matrix = np.array([
            [1 / (target_size / 2), 0, 0, 0],
            [0, 1 / (target_size / 2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        tran_world_pix = np.matmul(projection_matrix, view_matrix)
        pts = np.matmul(tran_world_pix, pts)
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
        # mask = np.isnan(depth)
        # depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        depth = imputer.fit_transform(depth)

        return depth

def main():
    rospy.init_node('test')
    cloudProxy = CloudProxy()
    while True:
        obs = cloudProxy.getProjectImg(0.4, 128)
        obs = -obs
        obs -= obs.min()
        # obs = skimage.transform.resize(obs, (90, 90))
        plt.imshow(obs)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    main()