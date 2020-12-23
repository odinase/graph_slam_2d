# %% GraphSLAM
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, L
import utils
import queue
from JCBB import JCBB
import profilehooks
import line_profiler
import atexit

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

class GraphSLAM:
    def __init__(self, p, q, r, alphas=np.array([0.001, 0.0001]), sensor_offset=np.zeros(2)):

        # Add noise models
        self.Q = gtsam.noiseModel.Diagonal.Sigmas(q)
        self.R = gtsam.noiseModel.Diagonal.Sigmas(r)
        
        self.alphas = alphas
        # Position of sensor offset, transform from sensor to body frame
        self.sensor_offset = gtsam.Pose2(sensor_offset[0], sensor_offset[1], 0)

        # Create graph and initilize newest pose
        self.estimates = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()

        # To enumerate all poses and landmarks in the graph
        self.landmark_count = 1
        self.pose_count = 0

        # Needs to store landmarks for associations later
        self.landmarks = np.empty((0, 2))
        self.landmarks_idxs = np.empty((0,), dtype=int)

        # Initilize graph with prior
        # Start in origin
        prior_pose = gtsam.Pose2(0.0, 0.0, 0.0)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(p)
        self.graph.add(gtsam.PriorFactorPose2(X(0), prior_pose, prior_noise))
        self.estimates.insert(X(0), prior_pose)

    def get_kf_pose(self, kf_idx):
        return self.estimates.atPose2(X(kf_idx))

    def get_lmk_point(self, lmk_idx):
        return self.estimates.atPoint2(lmk_idx)

    @property
    def latest_pose(self):
        return self.get_kf_pose(self.pose_count)

    # @profilehooks.profile(sort="cumulative")
    def new_odometry(self, u):
        """
        Does dead-reckoning on newest state. Checks if newest pose estimate is sufficiently different from last keyframe to add it to the pose graph.
        """
        # Propagate latest state
        odom = utils.np2pose(u)
        last_pose = self.latest_pose
        last_pose = last_pose.compose(odom)
        latest_keyframe_id = self.pose_count

        assert self.estimates.exists(X(latest_keyframe_id)), "new_odometry: latest_keyframe_id doesn't exist in graph??"

        # New keyframe, add to pose graph
        self.estimates.insert(X(latest_keyframe_id + 1), last_pose)

        # Connect previous keyframe to new keyframe by "virtual odometry", i.e. the difference between them
        self.graph.add(
            gtsam.BetweenFactorPose2(
                X(latest_keyframe_id), X(latest_keyframe_id + 1), odom, self.Q
            )
        )

        # Update pose count
        self.pose_count += 1

    def transform_jacobian(self, T, z):
        r, theta = z
        R = T.rotation().matrix()
        J = np.array([
            [np.cos(theta), -r*np.sin(theta)],
            [np.sin(theta),  r*np.cos(theta)]
        ])
        x, y = r*np.cos(theta), r*np.sin(theta)
        x_trans, y_trans = R@np.array([x, y]) + T.translation()
        L = np.array([
            [x_trans/np.sqrt(x_trans**2 + y_trans**2), y_trans/np.sqrt(x_trans**2 + y_trans**2)],
            [y/(x_trans**2 + y_trans**2), -x/(x_trans**2 + y_trans**2)]
        ])

        Jac = L@R@J

        return Jac

    # @profilehooks.profile(sort="cumulative")
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

        landmarks_to_add = True
        z_nonass = z

        # It doesn't make sense to make associations with no prior landmarks
        if numLmks > 0:
            marginals = gtsam.Marginals(self.graph, self.estimates)
            
            P_margs = marginals.jointMarginalCovariance(gtsam.KeyVector(np.append(X(self.pose_count), self.landmarks_idxs)))
            P = marginals.jointMarginalCovariance(gtsam.KeyVector(np.append(X(self.pose_count), self.landmarks_idxs))).fullMatrix()

            # We need to permute P to move pose from bottom-right to top-left
            P = np.roll(P, (3,3), axis=(0,1))

            latest_kf_pose = self.latest_pose

            H = self.H(latest_kf_pose)
            S = H @ P @ H.T

            idxs = np.arange(numLmks * 2).reshape(numLmks, 2)
            ## block diag indices for 2 x 2 blocks, broadcast R
            S[idxs[..., None], idxs[:, None]] += self.R.covariance()[None]

            z_pred = self.h(latest_kf_pose)
            z = z.ravel()

            a = JCBB(z, z_pred, S, self.alphas[0], self.alphas[1])

            # Needs to reshape back
            z = z.reshape(-1,2)

            # a > -1 gives associated measurements, a[a > -1] gives associated landmarks
            # We need to connect all previously seen landmarks to the last keyframe pose 

            # Needs to tranform measurements to being relative keyframe
            zass = z[a > -1]
            
            # Get two lists for ranges and bearings for convenience
            ranges = zass[:,0]
            bearings = [gtsam.Rot2(b) for b in zass[:,1]]

            # Index of landmarks that have been associated. Must start at 1 as 1-indexed
            associated_lmk_idxs = a[a > -1] + 1

            x_kf = X(self.pose_count)

            associated_lmks = [L(idx) for idx in associated_lmk_idxs]

            # Now to connect existing landmarks to newest keyframe with virtual measurements
            for lmk, z_range, z_bearing, org_meas in zip(associated_lmks, ranges, bearings, z):
                # Likelihood factor between associated landmark and last keyframe
                meas_factor = gtsam.BearingRangeFactor2D(
                    x_kf, lmk, z_bearing, z_range, self.R
                )
                self.graph.add(meas_factor)

            landmarks_to_add = np.any(a == -1)

            z_nonass = z[a == -1]
            
            # Cartesian measurement in sensor frame
            z_nonass_s = utils.polar2cart(z_nonass)
            
            T_ib = self.latest_pose
            T_bs = self.sensor_offset
            T_is = T_ib.compose(T_bs)

            z_c_i = utils.transform(T_is, z_nonass_s)

        else:
            a = np.zeros(z.shape[0]//2) - 1


        # We have measurements to add to graph
        if landmarks_to_add:
            # Get latest landmark idx (we start counting on 1)
            lmk_idx = self.landmarks_idxs.shape[0]
            kf_idx = self.pose_count
            latest_kf_pose = self.latest_pose
            
            for l, (l_range, l_bearing) in zip(z_c_i, z_nonass):
                # We add the inertial position of the landmark directly to the graph
                lmk_idx += 1
                self.estimates.insert(L(lmk_idx), l)

                self.graph.add(gtsam.BearingRangeFactor2D(
                    X(kf_idx), L(lmk_idx), gtsam.Rot2(l_bearing), l_range, self.R)
                )

                # Add new landmark to buffer for later
                self.landmarks_idxs = np.append(self.landmarks_idxs, L(lmk_idx))

            # Add landmark for later association
            self.landmarks = np.append(self.landmarks, z_c_i, axis=0)

        return a + 1

    # @profilehooks.profile(sort="cumulative")
    def h(self, pose):
        T_bi = pose.inverse() # Transform from inertial to body
        T_sb = self.sensor_offset.inverse() # Transform from body to sensor
        T_si = T_sb.compose(T_bi) # Transform from inertial to sensor
        landmarks_s = utils.transform(T_si, self.landmarks)
        l_ranges = np.linalg.norm(landmarks_s, axis=1)
        l_bearing = np.arctan2(landmarks_s[:,1], landmarks_s[:,0])
        z_pred = np.vstack((l_ranges, l_bearing)).ravel("F")

        return z_pred

    # @profilehooks.profile(sort="cumulative")
    def Hm(self, pose) -> np.ndarray:
        """Calculate the jacobian of h.
        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.
        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. eta.
        """
        # extract states and map
        x = self.latest_pose #eta[0:3]
        ## reshape map (2, #landmarks), m[j] is the jth landmark
        m = self.landmarks #eta[3:].reshape((-1, 2)).T

        numM = m.shape[0]

        R_ib = x.rotation().matrix()

        delta_m = m - x.translation() # relative position of landmarks to body frame in inertial frame

        # Arm from body to sensor in body frame
        r_bs_b = self.sensor_offset.translation()

        # Cartesian coordinates of landmarks relative sensor in inertial frame
        zc = delta_m - r_bs_b@R_ib.T

        # Transpose here to make dimensions correct later
        delta_m = delta_m.T
        zc = zc.T

        zpred = self.h(pose).reshape(-1, 2).T
        zr = zpred[0]

        Rpihalf = utils.rot_mat(np.pi / 2)

        # Allocate H and set submatrices as memory views into H
        H = np.zeros((2 * numM, 3 + 2 * numM))
        Hx = H[:, :3]  # slice view, setting elements of Hx will set H as well
        Hm = H[:, 3:]  # slice view, setting elements of Hm will set H as well

        jac_Zcb = -np.eye(2, 3)  # preallocate and update this for speed
        for i in range(numM):  # this whole loop can be vectorized
            ind = 2 * i
            inds = slice(ind, ind + 2)
            jac_Zcb[:, 2] = -Rpihalf @ delta_m[:, i]

            Hx[ind] = (zc[:, i] / zr[i]) @ jac_Zcb
            Hx[ind + 1] = (zc[:, i] / zr[i] ** 2) @ Rpihalf.T @ jac_Zcb

            Hm[inds, inds] = -Hx[inds, :2]

        return Hm

    # @profilehooks.profile(sort="cumulative")
    def Hx(self, pose) -> np.ndarray:
        """Calculate the jacobian of h.
        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.
        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. eta.
        """
        # extract states and map
        x = self.latest_pose #eta[0:3]
        ## reshape map (2, #landmarks), m[j] is the jth landmark
        m = self.landmarks #eta[3:].reshape((-1, 2)).T

        numM = m.shape[0]

        R_ib = x.rotation().matrix()

        delta_m = m - x.translation() # relative position of landmarks to body frame in inertial frame

        # Arm from body to sensor in body frame
        r_bs_b = self.sensor_offset.translation()

        # Cartesian coordinates of landmarks relative sensor in inertial frame
        zc = delta_m - r_bs_b@R_ib.T

        # Transpose here to make dimensions correct later
        delta_m = delta_m.T
        zc = zc.T

        zpred = self.h(pose).reshape(-1, 2).T
        zr = zpred[0]

        Rpihalf = utils.rot_mat(np.pi / 2)

        # Allocate H and set submatrices as memory views into H
        H = np.zeros((2 * numM, 3 + 2 * numM))
        Hx = H[:, :3]  # slice view, setting elements of Hx will set H as well
        Hm = H[:, 3:]  # slice view, setting elements of Hm will set H as well

        jac_Zcb = -np.eye(2, 3)  # preallocate and update this for speed
        for i in range(numM):  # this whole loop can be vectorized
            ind = 2 * i
            inds = slice(ind, ind + 2)
            jac_Zcb[:, 2] = -Rpihalf @ delta_m[:, i]

            Hx[ind] = (zc[:, i] / zr[i]) @ jac_Zcb
            Hx[ind + 1] = (zc[:, i] / zr[i] ** 2) @ Rpihalf.T @ jac_Zcb

            Hm[inds, inds] = -Hx[inds, :2]

        return Hx

    # @profilehooks.profile(sort="cumulative")
    def H(self, pose) -> np.ndarray:
        """Calculate the jacobian of h.
        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.
        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. eta.
        """
        # extract states and map
        x = self.latest_pose #eta[0:3]
        ## reshape map (2, #landmarks), m[j] is the jth landmark
        m = self.landmarks #eta[3:].reshape((-1, 2)).T

        numM = m.shape[0]

        R_ib = x.rotation().matrix()

        delta_m = m - x.translation() # relative position of landmarks to body frame in inertial frame

        # Arm from body to sensor in body frame
        r_bs_b = self.sensor_offset.translation()

        # Cartesian coordinates of landmarks relative sensor in inertial frame
        zc = delta_m - r_bs_b@R_ib.T

        # Transpose here to make dimensions correct later
        delta_m = delta_m.T
        zc = zc.T

        zpred = self.h(pose).reshape(-1, 2).T
        zr = zpred[0]

        Rpihalf = utils.rot_mat(np.pi / 2)

        # Allocate H and set submatrices as memory views into H
        H = np.zeros((2 * numM, 3 + 2 * numM))
        Hx = H[:, :3]  # slice view, setting elements of Hx will set H as well
        Hm = H[:, 3:]  # slice view, setting elements of Hm will set H as well

        jac_Zcb = -np.eye(2, 3)  # preallocate and update this for speed
        for i in range(numM):  # this whole loop can be vectorized
            ind = 2 * i
            inds = slice(ind, ind + 2)
            jac_Zcb[:, 2] = -Rpihalf @ delta_m[:, i]

            Hx[ind] = (zc[:, i] / zr[i]) @ jac_Zcb
            Hx[ind + 1] = (zc[:, i] / zr[i] ** 2) @ Rpihalf.T @ jac_Zcb

            Hm[inds, inds] = -Hx[inds, :2]

        return H

    # Full optimizer of pose graph
    # @profilehooks.profile(sort="cumulative")
    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.estimates, params)
        self.estimates = optimizer.optimize()
# %% Testing

if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError as e:
        print(e)
        print("install tqdm to have progress bar")

        # def tqdm as dummy as it is not available
        def tqdm(*args, **kwargs):
            return args[0]
    from scipy.io import loadmat
    simSLAM_ws = loadmat("simulatedSLAM")

    z = [zk.T for zk in simSLAM_ws["z"].ravel()]

    landmarks = simSLAM_ws["landmarks"].T
    odometry = simSLAM_ws["odometry"].T
    poseGT = simSLAM_ws["poseGT"].T

    K = len(z)

    print(f"K is {K}")

    M = len(landmarks)

    # %% Initilize

    doAsso = True

    JCBBalphas = np.array(
        [0.001, 0.0001]
    )  # first is for joint compatibility, second is individual

    # For consistency testing
    alpha = 0.05

    q = np.array([0.05, 0.05, 1 * np.pi / 180])
    r = np.array([0.05, 1 * np.pi / 180])
    # GTSAM doesn't like zero covariance
    p = np.array([1e-9, 1e-9, 1e-9])

    slam = GraphSLAM(p, q, r, JCBBalphas)

    K = 200

    print(f"K is now {K}")

    for k in tqdm(range(K)):
        slam.new_observation(z[k])
        slam.new_odometry(odometry[k])

        # We shouldn't have to optimize super often... Right?
        if k % 5 == 0:
            slam.optimize()
    
# %%
