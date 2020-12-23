#!/usr/bin/env python3.6

# %% Imports
from typing import List, Optional

from scipy.io import loadmat
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import chi2
import utils
from GraphSLAM import GraphSLAM
import gtsam
from gtsam.symbol_shorthand import X, L

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm to have progress bar")

    # def tqdm as dummy as it is not available
    def tqdm(*args, **kwargs):
        return args[0]

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )


from plotting import ellipse

# %% Load data
simSLAM_ws = loadmat("simulatedSLAM")

## NB: this is a MATLAB cell, so needs to "double index" to get out the measurements of a time step k:
#
# ex:
#
# z_k = z[k][0] # z_k is a (2, m_k) matrix with columns equal to the measurements of time step k
#
##
z = [zk.T for zk in simSLAM_ws["z"].ravel()]

landmarks = simSLAM_ws["landmarks"].T
odometry = simSLAM_ws["odometry"].T
poseGT = simSLAM_ws["poseGT"].T

K = len(z)
M = len(landmarks)

# %% Initilize
doAsso = True

JCBBalphas = np.array(
    [1e-3, 1e-5]
)  # first is for joint compatibility, second is individual

# For consistency testing
alpha = 0.05

q = np.array([0.5, 0.5, 3 * np.pi / 180])
r = np.array([0.06, 2 * np.pi / 180])
# GTSAM doesn't like zero covariance
p = np.array([1e-9, 1e-9, 1e-9])

slam = GraphSLAM(p, q, r, JCBBalphas)

NIS = np.zeros(K)
NISnorm = np.zeros(K)
CI = np.zeros((K, 2))
CInorm = np.zeros((K, 2))
NEESes = np.zeros((K, 3))

# %% Set up plotting
# plotting

doAssoPlot = False
playMovie = True
if doAssoPlot:
    figAsso, axAsso = plt.subplots(num=1, clear=True)

# %% Run simulation
N = 100#K

a = [None] * N

print("starting sim (" + str(N) + " iterations)")

total_num_asso = 0

for k, (z_k, u_k) in tqdm(enumerate(zip(z[:N], odometry[:N]))):
    
    a_k = slam.new_observation(z_k)
    slam.optimize()
    # We shouldn't have to optimize super often... Right?
    # if k % 5 == 0:

    if k < K - 1:
        slam.new_odometry(u_k)

    a[k] = a_k

print(slam.landmarks[slam.landmarks[:,0].argsort()])

for k, a_k in enumerate(a):
    print(f"{k+1}: {a_k}")

    # num_asso = np.count_nonzero(a[k] > -1)

    # CI[k] = chi2.interval(1 - alpha, 2 * num_asso)

    # if num_asso > 0:
    #     NISnorm[k] = NIS[k] / (2 * num_asso)
    #     CInorm[k] = CI[k] / (2 * num_asso)
    #     total_num_asso += num_asso
    # else:
    #     NISnorm[k] = 1
    #     CInorm[k].fill(1)
    
    # NEESes[k] = slam.NEESes(eta_hat[k][:3], P_hat[k][:3, :3], poseGT[k])

    # if doAssoPlot and k > 0:
    #     axAsso.clear()
    #     axAsso.grid()
    #     zpred = slam.h(eta_pred[k]).reshape(-1, 2)
    #     axAsso.scatter(z_k[:, 0], z_k[:, 1], label="z")
    #     axAsso.scatter(zpred[:, 0], zpred[:, 1], label="zpred")
    #     xcoords = np.block([[z_k[a[k] > -1, 0]], [zpred[a[k][a[k] > -1], 0]]]).T
    #     ycoords = np.block([[z_k[a[k] > -1, 1]], [zpred[a[k][a[k] > -1], 1]]]).T
    #     for x, y in zip(xcoords, ycoords):
    #         axAsso.plot(x, y, lw=3, c="r")
    #     axAsso.legend()
    #     axAsso.set_title(f"k = {k}, {np.count_nonzero(a[k] > -1)} associations")
    #     plt.draw()
    #     plt.pause(0.001)


print("sim complete")

pose_est = np.array([utils.pose2np(slam.get_kf_pose(kf)) for kf in range(slam.pose_count+1)])
# lmk_est = [eta_hat_k[3:].reshape(-1, 2) for eta_hat_k in eta_hat]
# lmk_est_final = lmk_est[N - 1]
lmk_est_final = np.array([slam.get_lmk_point(lmk) for lmk in slam.landmarks_idxs])

marginals = gtsam.Marginals(slam.graph, slam.estimates)
P_hat = marginals.jointMarginalCovariance(gtsam.KeyVector(np.append(X(slam.pose_count), slam.landmarks_idxs)))


np.set_printoptions(precision=4, linewidth=100)

# %% Plotting of results
mins = np.amin(landmarks, axis=0)
maxs = np.amax(landmarks, axis=0)

ranges = maxs - mins
offsets = ranges * 0.2

mins -= offsets
maxs += offsets

fig2, ax2 = plt.subplots(num=2, clear=True)
# landmarks
ax2.scatter(*landmarks.T, c="r", marker="^")
ax2.scatter(*lmk_est_final.T, c="b", marker=".")
# Draw covariance ellipsis of measurements
for l, (lmk_l, l_idx) in enumerate(zip(lmk_est_final, slam.landmarks_idxs)):
    idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
    rI = P_hat.at(l_idx, l_idx)
    el = ellipse(lmk_l, rI, 5, 200)
    ax2.plot(*el.T, "b")
    ax2.annotate(f"{l+1}", # this is the text
                (lmk_l[0], lmk_l[1]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center

ax2.plot(*poseGT.T[:2,:N], c="r", label="gt")
ax2.plot(*pose_est.T[:2], c="g", label="est")
ax2.plot(*ellipse(pose_est[-1, :2], P_hat.at(X(slam.pose_count), X(slam.pose_count))[:2,:2], 5, 200).T, c="g")
ax2.set(title="results", xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
ax2.axis("equal")
ax2.grid()

plt.show()
# %% Consistency

# # NIS
# insideCI = (CInorm[:N,0] <= NISnorm[:N]) * (NISnorm[:N] <= CInorm[:N,1])

# fig3, ax3 = plt.subplots(num=3, clear=True)
# ax3.plot(CInorm[:N,0], '--')
# ax3.plot(CInorm[:N,1], '--')
# ax3.plot(NISnorm[:N], lw=0.5)

# ax3.set_title(f'NIS, {insideCI.mean()*100}% inside CI')

# # NEES

# fig4, ax4 = plt.subplots(nrows=3, ncols=1, figsize=(7, 5), num=4, clear=True, sharex=True)
# tags = ['all', 'pos', 'heading']
# dfs = [3, 2, 1]

# for ax, tag, NEES, df in zip(ax4, tags, NEESes.T, dfs):
#     CI_NEES = chi2.interval(1 - alpha, df)
#     ax.plot(np.full(N, CI_NEES[0]), '--')
#     ax.plot(np.full(N, CI_NEES[1]), '--')
#     ax.plot(NEES[:N], lw=0.5)
#     insideCI = (CI_NEES[0] <= NEES) * (NEES <= CI_NEES[1])
#     ax.set_title(f'NEES {tag}: {insideCI.mean()*100}% inside CI')

#     CI_ANEES = np.array(chi2.interval(1 - alpha, df*N)) / N
#     print(f"CI ANEES {tag}: {CI_ANEES}")
#     print(f"ANEES {tag}: {NEES.mean()}")

# fig4.tight_layout()

# # ANIS

# valid_NIS = ~np.isnan(NIS)
# NIS = NIS[valid_NIS]
# CI_ANIS = np.array(chi2.interval(1 - alpha, total_num_asso * 2)) / NIS.size
# ANIS = NIS.mean()
# print(f"ANIS: {ANIS}\nBounds: {CI_ANIS}")

# # %% RMSE

# ylabels = ['m', 'deg']
# scalings = np.array([1, 180/np.pi])

# fig5, ax5 = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), num=5, clear=True, sharex=True)

# pos_err = np.linalg.norm(pose_est[:N,:2] - poseGT[:N,:2], axis=1)
# heading_err = np.abs(utils.wrapToPi(pose_est[:N,2] - poseGT[:N,2]))

# errs = np.vstack((pos_err, heading_err))

# for ax, err, tag, ylabel, scaling in zip(ax5, errs, tags[1:], ylabels, scalings):
#     ax.plot(err*scaling)
#     ax.set_title(f"{tag}: RMSE {np.sqrt((err**2).mean())*scaling} {ylabel}")
#     ax.set_ylabel(f"[{ylabel}]")
#     ax.grid()

# fig5.tight_layout()

# # %% Movie time

# if playMovie:
#     try:
#         print("recording movie...")

#         from celluloid import Camera

#         pauseTime = 0.05
#         fig_movie, ax_movie = plt.subplots(num=6, clear=True)

#         camera = Camera(fig_movie)

#         ax_movie.grid()
#         ax_movie.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
#         camera.snap()

#         for k in tqdm(range(N)):
#             ax_movie.scatter(*landmarks.T, c="r", marker="^")
#             ax_movie.plot(*poseGT[:k, :2].T, "r-")
#             ax_movie.plot(*pose_est[:k, :2].T, "g-")
#             ax_movie.scatter(*lmk_est[k].T, c="b", marker=".")

#             if k > 0:
#                 el = ellipse(pose_est[k, :2], P_hat[k][:2, :2], 5, 200)
#                 ax_movie.plot(*el.T, "g")

#             numLmk = lmk_est[k].shape[0]
#             for l, lmk_l in enumerate(lmk_est[k]):
#                 idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
#                 rI = P_hat[k][idxs, idxs]
#                 el = ellipse(lmk_l, rI, 5, 200)
#                 ax_movie.plot(*el.T, "b")

#             camera.snap()
#         animation = camera.animate(interval=100, blit=True, repeat=False)
#         print("playing movie")

#     except ImportError:
#         print(
#             "Install celluloid module, \n\n$ pip install celluloid\n\nto get fancy animation of EKFSLAM."
#         )

# plt.show()
# %%
