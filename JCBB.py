import numpy as np
from functools import lru_cache
# import profilehooks
from scipy.stats import chi2
import scipy.linalg as la
import utils

chi2isf_cached = lru_cache(maxsize=None)(chi2.isf)

# TODO: make sure a is 0-indexed
def JCBB(z, zbar, S, alpha1, alpha2):
    assert len(z.shape) == 1, "z must be in one row in JCBB"
    assert z.shape[0] % 2 == 0, "z must be equal in x and y"
    m = z.shape[0] // 2

    a = np.full(m, -1, dtype=int)
    abest = np.full(m, -1, dtype=int)

    # ic has measurements rowwise and predicted measurements columnwise
    ic = individualCompatibility(z, zbar, S)
    g2 = chi2.isf(alpha2, 2)
    order = np.argsort(np.amin(ic, axis=1))
    j = 0
    z_order = np.empty(2 * len(order), dtype=int)
    z_order[::2] = 2 * order
    z_order[1::2] = 2 * order + 1
    zo = z[z_order]

    ico = ic[order]

    abesto = JCBBrec(zo, zbar, S, alpha1, g2, j, a, ico, abest)

    abest[order] = abesto

    return abest


def JCBBrec(z, zbar, S, alpha1, g2, j, a, ic, abest):
    m = z.shape[0] // 2
    assert isinstance(m, int), "m in JCBBrec must be int"
    n = num_associations(a)

    if j >= m:  # end of recursion
        if n > num_associations(abest) or (
            (n >= num_associations(abest))
            and (NIS(z, zbar, S, a) < NIS(z, zbar, S, abest))
        ):
            abest = a
        # else abest = previous abest from the input
    else:  # still at least one measurement to associate
        I = np.argsort(ic[j, ic[j, :] < g2])
        # allinds = np.array(range(ic.shape[1]), dtype=int)
        usableinds = np.where(ic[j, :] < g2)[0]  # allinds[ic[j, :] < g2]
        # if np.any(np.where(ic[j, :] < g2)[0] != usableinds):
        #     raise ValueError

        for i in usableinds[I]:
            a[j] = i
            # jointly compatible?
            if NIS(z, zbar, S, a) < chi2isf_cached(alpha1, 2 * (n + 1)):
                # We need to decouple ici from ic, so copy is required
                ici = ic[j:, i].copy()
                ic[j:, i] = np.Inf  # landmark not available any more.

                # Needs to explicitly copy a for recursion to work
                abest = JCBBrec(z, zbar, S, alpha1, g2, j + 1, a.copy(), ic, abest)
                ic[j:, i] = ici  # set landmark available again for next round.

        if n + (m - j - 2) >= num_associations(abest):
            a[j] = -1
            abest = JCBBrec(z, zbar, S, alpha1, g2, j + 1, a, ic, abest)

    return abest


#   @profilehooks.profile(sort="cumulative")
def individualCompatibility(z, zbar, S):
    nz = z.shape[0] // 2
    nz_bar = zbar.shape[0] // 2

    assert z.shape[0] % 2 == 0, "JCBB.individualCompatibility: z must have even lenght"
    assert (
        zbar.shape[0] % 2 == 0
    ), "JCBB.individualCompatibility: zbar must have even length"

    # all innovations from broadcasting
    # extra trailing dimension to avoid problems in solve when z has 2 landmarks
    v_all = z.reshape(-1, 1, 2, 1) - zbar.reshape(1, -1, 2, 1)

    # get the (2, 2) blocks on the diagonal to make the (nz_bar, 2, 2) array of individual S
    ## first idxs get to the start of lmk, second is along the lmk axis
    idxs = np.arange(nz_bar)[:, None] * 2 + np.arange(2)[None]
    ## broadcast lmk axis to two dimesions
    S_all = S[idxs[..., None], idxs[:, None]]

    # broadcast S_all over the measurements by adding leading 1 size axis to match v_all
    # solve nz by nz_bar systems
    # sum over axis 3 to get rid of trailing dim (faster than squeeze?)
    ic = (v_all * np.linalg.solve(S_all[None], v_all)).sum(axis=(2, 3))
    return ic


def NIS(z, zbar, S, a):
    zr = z.reshape(-1, 2).T
    zbarr = zbar.reshape(-1, 2).T

    nis = np.inf

    if (a > -1).any():  # We have associations
        is_ass = a > -1
        ztest = zr[:, is_ass]
        ass_idxs = a[is_ass]  # .astype(np.int)
        zbartest = zbarr[:, ass_idxs]

        inds = np.empty(2 * len(ass_idxs), dtype=int)
        inds[::2] = 2 * ass_idxs
        inds[1::2] = 2 * ass_idxs + 1
        # inds = np.block([[inds], [inds + 1]]).flatten("F")

        Stest = S[inds[:, None], inds]

        v = ztest - zbartest
        v = v.T.flatten()

        v[1::2] = utils.wrapToPi(v[1::2])

        nis = v @ np.linalg.solve(Stest, v)

    return nis


def num_associations(array):
    return np.count_nonzero(array > -1)
