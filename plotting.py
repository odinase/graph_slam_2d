import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la


def ellipse(mu, P, s, n):
    thetas = np.linspace(0, 2*np.pi, n)
    ell = mu + s * (la.cholesky(P).T @ np.array([np.cos(thetas), np.sin(thetas)])).T
    return ell