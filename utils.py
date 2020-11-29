import numpy as np
import gtsam

def wrapToPi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def rot_mat(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])

def np2pose(a):
    assert a.shape == (3,), "a must be (3,) to make Pose2"
    return gtsam.Pose2(a[0], a[1], a[2])

def pose2np(p):
    return np.array([p.x(), p.y(), p.theta()])
