import numpy as np
import open3d

def get_point_cloud_from_depth(rgb, depth, intrinsics):
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])

    points = np.float32([px, py, depth, rgb[..., 0], rgb[..., 1], rgb[..., 2]]).transpose(1, 2, 0)
    cloud=points.reshape(-1,6)
    return cloud

def visualize_pointcloud(cloud):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(cloud[:,:3])
    pcd.colors = open3d.utility.Vector3dVector(cloud[:, 3:6]/255)
    open3d.visualization.draw_geometries([pcd])