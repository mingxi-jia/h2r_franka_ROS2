import rospy
import tf2_ros
import src.utils.transformation as transformation

class TFProxy:
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)

    def lookup_transform(self, fromFrame, toFrame, lookupTime=rospy.Time(0)):
        """
        Lookup a transform in the TF tree.
        :param fromFrame: the frame from which the transform is calculated
        :type fromFrame: string
        :param toFrame: the frame to which the transform is calculated
        :type toFrame: string
        :return: transformation matrix from fromFrame to toFrame
        :rtype: 4x4 np.array
        """

        transformMsg = self.tfBuffer.lookup_transform(fromFrame, toFrame, lookupTime, rospy.Duration(1.0))
        translation = transformMsg.transform.translation
        pos = [translation.x, translation.y, translation.z]
        rotation = transformMsg.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        T = transformation.quaternion_matrix(quat)
        T[0:3, 3] = pos
        return T
