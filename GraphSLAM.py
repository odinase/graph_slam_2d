# %% GraphSLAM
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
import utils
import queue
from JCBB import JCBB

class GraphSLAM:
    def __init__(self, p, q, r, alphas=np.array([0.001, 0.0001]), sensor_offset=np.zeros(2), keyframe_rotation = 2.5*np.pi/180, keyframe_distance = 0.5):
        
        # Thresholds for when to add new pose to graph
        self.keyframe_rotation = keyframe_rotation
        self.keyframe_distance = keyframe_distance

        # Add noise models
        self.Q = gtsam.noiseModel.Diagonal.Sigmas(q)
        self.R = gtsam.noiseModel.Diagonal.Sigmas(r)
        
        self.alphas = alphas
        # Position of sensor offset, transform from sensor to body frame
        self.sensor_offset = gtsam.Pose2(sensor_offset[0], sensor_offset[1], 0)

        # Create graph and initilize newest pose
        self.estimates = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()

        # Start in origin
        self.latest_pose = gtsam.Pose2(0.0, 0.0, 0.0)

        # To enumerate all poses and landmarks in the graph
        self.landmark_count = 1
        self.pose_count = 0

        # Needs to store landmarks for associations later
        self.landmarks = np.empty((0, 2))
        self.landmarks_idxs = np.empty((0,), dtype=int)

        # Initilize graph with prior
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(p)
        self.graph.add(gtsam.PriorFactorPose2(X(0), self.latest_pose, prior_noise))
        self.estimates.insert(X(0), self.latest_pose)

    def get_kf_pose(self, kf_idx):
        return self.estimates.atPose2(X(kf_idx))

    def new_odometry(self, u):
        """
        Does dead-reckoning on newest state. Checks if newest pose estimate is sufficiently different from last keyframe to add it to the pose graph.
        """
        # Propagate latest state
        odom = gtsam.Pose2(u[0], u[1], u[2])
        self.latest_pose = self.latest_pose.compose(odom)
        latest_keyframe_id = self.pose_count

        latest_kf_pose = self.get_kf_pose(latest_keyframe_id)

        assert self.estimates.exists(X(latest_keyframe_id)), "new_odometry: latest_keyframe_id doesn't exist in graph??"


        pose_diff = latest_kf_pose.between(self.latest_pose)

        if (
            # We have moved sufficiently far away
            np.linalg.norm(pose_diff.translation()) > self.keyframe_distance
            or 
            # We have rotated enough
            np.abs(pose_diff.theta()) > self.keyframe_rotation
        ):
            # New keyframe, add to pose graph
            self.estimates.insert(X(latest_keyframe_id + 1), self.latest_pose)

            # Connect previous keyframe to new keyframe by "virtual odometry", i.e. the difference between them
            self.graph.add(
                gtsam.BetweenFactorPose2(
                    X(latest_keyframe_id), X(latest_keyframe_id + 1), pose_diff, self.Q
                )
            )

            # Update pose count
            self.pose_count = latest_keyframe_id + 1


    # def meas_in_inertial_cartesian(self, z):
    #     # Transpose here to make the math easier
    #     z = z.T
    #     # Cartesian sensor frame
    #     z_c_s = np.vstack((z[0]*np.cos(z[1]), z[0]*np.sin(z[1])))
    #     # Cartesian body frame
    #     z_c_b = self.sensor_offset.rotation().matrix()@z_c_s + self.sensor_offset.translation()[:,None]
    #     # Cartesian inertial (world) frame
    #     z_c_i = self.latest_pose.rotation().matrix()@z_c_b + self.latest_pose.translation()[:,None]

    #     # Transpose back to make the 
    #     return z_c_i.T


    def new_observation(self, z):
        """
        Receives a bunch of new observations, assumed to be in the same frame of latest pose. 

        We need to do associations as well. We first run JCBB on all measurements against all landmarks in graph. We then connect the associated landmarks to the latest pose keyframe.

        All unassociated measurements are stored as candidates. They should only be validated as actual landmarks with enough confidence (must double check how this is done, but looked simple enough)
        """

        # All received measurements are in range-bearing in newest pose, needs to transform to world

        assert len(self.landmarks_idxs.shape) == 1, "GraphSLAM.new_observation: landmark_idx needs to be 1D"
        assert self.landmarks.shape[0] == self.landmarks_idxs.shape[0], "GraphSLAM.new_observation: Must be equally many landmarks as landmarks_idxs"

        numLmks = self.landmarks_idxs.shape[0]
        z_c_s = utils.polar2cart(z)
        T_bs = self.sensor_offset
        T_ib = self.latest_pose
        T_is = T_ib.compose(T_bs)
        z_c_i = utils.transform(T_is, z_c_s)

        assert z_c_i.shape == z.shape, "GraphSLAM.new_observation: z_c_i and z must have same shape"

        # It doesn't make sense to make associations with no prior landmarks
        if numLmks > 0:
            marginals = gtsam.Marginals(self.graph, self.estimates)
            
            P_ = marginals.jointMarginalCovariance(gtsam.KeyVector(self.landmarks_idxs)).fullMatrix()

            # z = z.ravel()
            # T_bi = self.latest_pose.inverse()
            # landmarks_b = utils.transform(T_bi, self.landmarks)
            # l_ranges = np.linalg.norm(landmarks_b, axis=1)
            # l_bearing = np.arctan2(landmarks_b[:,1], landmarks_b[:,0])
            # z_pred = np.vstack((l_ranges, l_bearing)).ravel("F")

            z = z_c_i.ravel()
            z_pred = self.landmarks.ravel()

            a = JCBB(z, z_pred, S, self.alphas[0], self.alphas[1])

        # We can make no associations, add all measurements to graph
        else:
            # Get latest landmark idx (we start counting on 1)
            lmk_idx = self.landmarks_idxs.shape[0]
            kf_idx = self.pose_count
            latest_kf_pose = self.get_kf_pose(kf_idx)
            
            for l in z_c_i:
                # We add the inertial position of the landmark directly to the graph
                lmk_idx += 1
                self.estimates.insert(L(lmk_idx), l)
                # We need to transform range-bearing to latest keyframe
                l_b = latest_kf_pose.transformTo(l)
                l_range = np.linalg.norm(l_b)
                l_bearing = gtsam.Rot2.relativeBearing(l_b)

                self.graph.add(gtsam.BearingRangeFactor2D(
                    X(kf_idx), L(lmk_idx), l_bearing, l_range, self.R))

                # Add new landmark to buffer for later
                self.landmarks_idxs = np.append(self.landmarks_idxs, L(lmk_idx))

            # Add landmark for later association
            self.landmarks = np.append(self.landmarks, z_c_i, axis=0)

# %% Testing

if __name__ == "__main__":
    from scipy.io import loadmat
    simSLAM_ws = loadmat("simulatedSLAM")

    z = [zk.T for zk in simSLAM_ws["z"].ravel()]

    landmarks = simSLAM_ws["landmarks"].T
    odometry = simSLAM_ws["odometry"].T
    poseGT = simSLAM_ws["poseGT"].T

    K = len(z)
    M = len(landmarks)

    # %% Initilize

    doAsso = True

    JCBBalphas = np.array(
        [0.0500,    0.0111]
    )  # first is for joint compatibility, second is individual

    # For consistency testing
    alpha = 0.05

    q = np.array([0.5, 0.5, 3*np.pi/180])**2;
    r = np.array([0.06, 2*np.pi/180])**2;
    p = np.array([0.01, 0.01, 0.01]) ** 2

    slam = GraphSLAM(p, q, r, JCBBalphas)


    slam.new_observation(z[0])
    slam.new_odometry(odometry[0])

    slam.new_observation(z[1])
    slam.new_odometry(odometry[1])

    print(slam.estimates)
# %%
