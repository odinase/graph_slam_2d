import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
import utils

class GraphSLAM:
    def __init__(self, p, q, r):
        
        # Add noise models
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(q)
        self.measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(r)

        # Create graph and initilize newest pose
        self.graph = gtsam.NonlinearFactorGraph()
        self.poses = gtsam.Values()

        # To enumerate all poses and landmarks
        self.kx = 1
        self.kl = 1

        self.landmarks = np.empty(0)

        # Initilize graph with prior
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(p)
        self.graph.add(gtsam.PriorFactorPose2(X(self.kx), gtsam.Pose2(0.0, 0.0, 0.0), prior_noise))


    def add_factor(self, factor):
        self.graph.add(factor)


    def f(self, x, u):
        xpred = np.hstack(
            (utils.rot_mat(x[2]) @ u, utils.wrapToPi(x[2] + u[2]))
        )

        return xpred


    def predict(self, x, u):
        """
        Should just predict next pose and add it together with the odometry factor 
        """
        odom_factor = gtsam.BetweenFactorPose2(X(self.kx), X(self.kx+1), gtsam.Pose2(u[0], u[1], u[2]), self.odometry_noise)

        self.add_factor(odom_factor)
        xpred = self.f(x, u)

        self.estimates.insert(X(self.kx + 1), utils.np2pose(xpred))

        self.kx += 1

        return x


    def update(self, z):
        
        
