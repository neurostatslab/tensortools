import numpy as np
import numba


@numba.jit(nopython=True, cache=True)
def sym_bmat_mul(S, x, out):
    """
    Symmetric banded matrix times vector.

    Parameters
    ----------
    S : ndarray
        (b x n) array specifying symmetric banded matrix in
        "upper form".
    x : ndarray
        Vector of length n, multiplying S.
    out : ndarray
        Vector of length n, which is overwritten to store result.
    """

    b, n = S.shape

    for i in range(n):
        out[i] = S[-1, i] * x[i]

    for j in range(1, b):
        for i in range(n - j):
            out[i] += S[(-j - 1), j + i] * x[j + i]
            out[j + i] += S[(-j - 1), j + i] * x[i]

    return out


@numba.jit(nopython=True, cache=True)
def shift_gram(shift, T, out):
    """
    Creates symmetric banded matrix representation of
    dot(W.T, W) in upper form.
    """
    out.fill(0.0)

    if shift > 0:
        d, r = int(shift), (shift % 1)
        out[-1, 0] = d + 1
        z1 = (1 - r) ** 2
        z2 = r ** 2
        z3 = r * (1 - r)
        for i in range(T - d - 1):
            out[-1, i] += z2
            out[-2, i + 1] += z3
        for i in range(1, T - d):
            out[-1, i] += z1

    elif shift < 0:
        d, r = int(-shift), ((-shift) % 1)
        out[-1, -1] = d + 1
        z1 = (1 - r) ** 2
        z2 = r ** 2
        z3 = r * (1 - r)
        for i in range(T - d - 1):
            out[-1, -1 - i] += z2
            out[-2, -1 - i] += z3
        for i in range(1, T - d):
            out[-1, -1 - i] += z1

    else:
        out[-1] += 1.0

    return out


@numba.jit(nopython=True, cache=True)
def apply_shift(x, shift, out):
    """
    Translates elements of `x` along axis=0 by `shift`, using linear
    interpolation for non-integer shifts.

    Parameters
    ----------
    x : ndarray
        Array with ndim >= 1, holding data.
    shift : float
        Shift magnitude.
    out : ndarray
        Array with the same shape as x.

    Returns
    -------
    out : ndarray
    """

    T = len(out)
    if shift > 0:
        d = int(shift // 1)
        r = shift % 1
        for t in range(T):
            j = t - d
            if j <= 0:
                out[t] = x[0]
            else:
                out[t] = x[j] * (1 - r) + x[j - 1] * r
    elif shift < 0:
        d = int((-shift) // 1)
        r = (-shift) % 1
        for t in range(T):
            j = t - d
            if j <= 0:
                out[-t-1] = x[-1]
            else:
                out[-t-1] = x[-j-1] * (1 - r) + x[-j] * r
    else:
        out[:] = x

    return out


@numba.jit(nopython=True, cache=True)
def trans_shift(x, shift, out):
    out.fill(0.0)
    T = len(out)

    if shift > 0:
        d = int(shift // 1)
        r = shift % 1
        for t in range(T):
            j = t - d
            if j <= 0:
                out[0] += x[t]
            else:
                out[j] += x[t] * (1 - r)
                out[j - 1] += x[t] * r

    elif shift < 0:
        d = int((-shift) // 1)
        r = (-shift) % 1
        for t in range(T):
            j = t - d
            if j <= 0:
                out[-1] += x[-t-1]
            else:
                out[-j-1] += x[-t-1] * (1 - r)
                out[-j] += x[-t-1] * r
    else:
        out[:] = x

    return out
