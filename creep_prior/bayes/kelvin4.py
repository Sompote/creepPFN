"""Identifiable Kelvin-plus-log family ('kelvin4') for the hierarchical prior.

The original Kelvin-plus-log family has retardation times 0.3, 3, 30 and 300 days plus
log(1 + t/10). Over 0-160 days the 300-day term and the log term are nearly collinear, so
their relative weight is not identified and NUTS chains drift along that direction. 'kelvin4'
drops the 300-day term: g(t) = sum_k A_k h_k(t; t_a), k over {0.3, 3, 30 days, log(1+t/10)},
each h_k normalised to 0 at t_a and 1 at day 160, A_k >= 0, M = sum A_k.
Unconstrained vector: z = (log M, log A_1/A_4, log A_2/A_4, log A_3/A_4).
"""
import numpy as np
from scipy.optimize import nnls

HORIZON = 160.0
TAU4 = np.array([0.3, 3.0, 30.0])
NAME = 'kelvin4'


def basis4(t, anchor):
    t = np.asarray(t, float)
    def phi(v):
        v = np.asarray(v, float)
        return np.concatenate([-np.expm1(-v[..., None] / TAU4), np.log1p(v[..., None] / 10)], axis=-1)
    o = phi(anchor)
    return (phi(t) - o) / np.maximum(phi(HORIZON) - o, 1e-12)


def evaluate4(params, t, anchor):
    return basis4(t, anchor) @ np.asarray(params)


def encode4(params):
    p = np.maximum(np.asarray(params, float), 1e-6)
    return np.r_[np.log(p.sum()), np.log(p[:-1] / p[-1])]


def decode4(z):
    z = np.asarray(z, float)
    w = np.exp(np.r_[z[1:], 0.0] - np.max(np.r_[z[1:], 0.0])); w /= w.sum()
    return np.exp(z[0]) * w


def fit4(t, y):
    """Nonnegative least squares with the same weak amplitude penalty as the pipeline fit."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    amp = max(float(np.ptp(y)), 1.0)
    p, _ = nnls(np.vstack([basis4(t, t[0]), np.eye(4) * 0.05]), np.r_[y / amp, np.zeros(4)])
    return np.maximum(p * amp, 1e-6)
