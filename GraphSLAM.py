import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
import utils

class GraphSLAM:
    def __init__(self, p, q, r, alphas=np.array([0.001, 0.0001]), sensor_offset=np.zeros(2)):
        
        # Add noise models
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(q)
        self.measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(r)
        
        self.alphas = alphas
        self.sensor_offset = sensor_offset

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
        """
        Takes in vector of range-bearing measurements of landmarks. 

        Should do association by JCBB to existing landmarks first. This will allow for adding extra factors 
        """
        # S = marginals.jointMarginalCovariance(gtsam.KeyVector(nodes))


        a = JCBB(z, zpred, S, self.alphas[0], self.alphas[1])
        # Extract associated measurements
        zinds = np.empty_like(z, dtype=bool)
        zinds[::2] = a > -1  # -1 means no association
        zinds[1::2] = zinds[::2]
        zass = z[zinds] # associated measurements

        # extract and rearange predicted measurements and cov
        zbarinds = np.empty_like(zass, dtype=int)
        zbarinds[::2] = 2 * a[a > -1]
        zbarinds[1::2] = 2 * a[a > -1] + 1
        zpredass = zpred[zbarinds]
