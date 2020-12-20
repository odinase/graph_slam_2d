import numpy as np
import gtsam
import utils

class FrontEnd:
    def __init__(self, keyframe_rotation = 2.5*np.pi/180, keyframe_distance = 0.5):
        
        self.keyframe_rotation = keyframe_rotation
        self.keyframe_distance = keyframe_distance

        self.latest_poses = []
        self.latest_kf_poses = []
        self.pose_count = 0

    def new_odometry(self, u):
        """
        Takes in an odometry measurement with displacement in x and y in body frame, together with difference in heading.
        Needs to add new pose for association later, in addition to see if this is a keyframe. 
        """
        
        odom = gtsam.Pose2(u[0], u[1], u[2])
        new_pose = latest_poses[-1].compose(odom)

        latest_poses.append(new_pose)

        pose_diff = latest_kf_poses[-1].between(new_pose)
        if (
            np.linalg.norm(pose_diff.translation()) > self.keyframe_distance
            or 
            np.abs(pose_diff.theta()) > keyframe_rotation
        ):
            latest_kf_poses.append(new_pose)


    def new_observation(self, z):
        pass

