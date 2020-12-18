import numpy as np
import gtsam

class FrontEnd:
    def __init__(self, keyframe_rotation = 2.5*np.pi/180, keyframe_distance = 0.5):
        
        self.keyframe_rotation = keyframe_rotation
        self.keyframe_distance = keyframe_distance

        self.latest_odoms = []
        self.latest_kf_odoms = []
        self.pose_count = 0

    def new_odometry(self, u):
        
        



    def new_observation(self, z):
        pass

