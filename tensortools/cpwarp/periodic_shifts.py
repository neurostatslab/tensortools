import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def tri_sym_circ_matvec(c, a, x, out):
    out[0] = (a * x[-1]) + (c * x[0]) + (a * x[1])
    for i in range(1, x.shape[0] - 1):
        out[i] = (a * x[i - 1]) + (c * x[i]) + (a * x[i + 1])
    out[-1] = (a * x[-2]) + (c * x[-1]) + (a * x[0])


@numba.jit(nopython=True, cache=True)
def rojo_solve(c, a, f, x, z):
    """
    Solves symmetric, tridiagonal circulant system, assuming diagonal
    dominance. Algorithm and notation described in Rojo (1990).
    Parameters
    ----------
    c : float
        Diagonal elements.
    a : float
        Off-diagonal elements. Should satisfy abs(c) > 2 * abs(a).
    f : ndarray
        Right-hand side.
    x : ndarray
        Vector holding solution.
    z : ndarray
        Vector storing intermediate computations
    Reference
    ---------
    Rojo O (1990). A new method for solving symmetric circulant
    tridiagonal systems of linear equations. Computers Math Applic.
    20(12):61-67.
    """

    N = f.size

    for i in range(N):
        z[i] = -f[i] / a

    lam = -c / a

    if lam > 0:
        mu = 0.5 * lam + np.sqrt(0.25 * (lam ** 2) - 1)
    else:
        mu = 0.5 * lam - np.sqrt(0.25 * (lam ** 2) - 1)

    z[0] = z[0] + (z[-1] / lam)
    for i in range(1, N - 2):
        z[i] = z[i] + (z[i - 1] / mu)
    z[-2] = z[-2] + (z[-1] / lam) + (z[-3] / mu)

    z[-2] = z[-2] / mu
    for i in range(N - 2):
        z[-3 - i] = (z[-3 - i] + z[-2 - i]) / mu

    musm1 = ((mu ** 2) - 1)
    d = (1 - (mu ** -N)) * musm1 * mu
    mu1 = mu ** (1 - N)
    mu2 = mu
    mu3 = mu ** (3 - N)
    for i in range(N - 1):
        x[i] = z[i] + (musm1 * mu1 * z[0] + (mu2 + mu3) * z[-2]) / d

        mu1 *= mu
        mu2 /= mu
        mu3 *= mu

    x[-1] = (z[-1] + x[0] + x[-2]) / lam

    return x


@numba.jit(nopython=True, cache=True)
def shift_gram(shift):
    """
    Computes weighted sum over dot(W_k.T, W_k).
    """
    rr = shift % 1
    d = (1 - rr) ** 2 + rr ** 2
    off_d = rr * (1 - rr)
    return d, off_d


@numba.jit(nopython=True, cache=True)
def apply_shift(x, shift, out):

    T = out.shape[0]

    if shift > 0:
        d = int(shift // 1)
        r = shift % 1
        for t in range(T):
            j = t - d
            out[t] = x[j] * (1 - r) + x[j - 1] * r

    elif shift < 0:
        d = int((-shift) // 1)
        r = (-shift) % 1
        for t in range(T):
            j = t - d + 1
            out[-t - 1] = x[-j] * (1 - r) + x[-j + 1] * r

    else:
        out[:] = x

    return out


@numba.jit(nopython=True, cache=True)
def trans_shift(x, shift, out):

    T = out.shape[0]
    out.fill(0.0)

    if shift > 0:
        d = int(shift // 1)
        r = shift % 1
        for t in range(T):
            j = t - d
            out[j] += x[t] * (1 - r)
            out[j - 1] += x[t] * r

    elif shift < 0:
        d = int((-shift) // 1)
        r = (-shift) % 1
        for t in range(T):
            j = t - d + 1
            out[-j] += x[-t - 1] * (1 - r)
            out[-j + 1] += x[-t - 1] * r

    else:
        out[:] = x

    return out
