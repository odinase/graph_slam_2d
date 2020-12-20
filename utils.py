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

def transform(T: gtsam.Pose2, p: np.ndarray):
    R = T.rotation().matrix()
    t = T.translation()
    return p@R.T + t

def polar2cart(p):
    """
    Assumes p is [[range1, bearing1], [range2, bearing2], ...]
    """
    c = np.vstack((
        p[:,0]*np.cos(p[:,1]),
        p[:,0]*np.sin(p[:,1])
    )).T

    return c

def cart2polar(c):
    """
    Assumes c is [[x1, y1], [x2, y2], ...]
    """
    p = np.vstack((
        np.linalg.norm(c, axis=1),
        np.arctan2(c[:,1], c[:,0])
    )).T

    return p