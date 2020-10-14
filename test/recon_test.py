# -*- coding: utf-8 -*-
"""
This module is used for showing that the Daubechies 2 boundary wavelets can
reconstruct constant and linear functions.

The boundary_wavelets.py package is licensed under the MIT "Expat" license.

Copyright (c) 2018: Josefine Holm and Steffen L. Nielsen.
"""

import matplotlib.pyplot as plt
import numpy as np
import pywt
import recon_functions as rf
import boundwave.orthonormal as ot


def test_of_con_func(J=3):
    """
    Test and plot of constant and linear function.
    """

    wavelet = 'db2'
    wavelet_coef = np.flipud(pywt.Wavelet(wavelet).dec_lo)
    phi = pywt.Wavelet(wavelet).wavefun(level=15)[0][1:]
    AL, AR = ot.ortho_matrix(J, wavelet_coef, phi)
    x = np.ones(2**J)
    x[1], x[-1] = 0, 0
    x[0] = 1 / AL[0, 0]
    x[-2] = 1 / AR[0, 0]
    res = rf.recon_boundary(x, J, wavelet, phi)

    plt.figure()
    plt.plot(np.linspace(0, 2**J, 2**J, endpoint=True), np.ones(2**J),
             label=r'$f(x)$')
    plt.plot(np.linspace(0, 2**J, len(res), endpoint=True),
             np.real(res), ls='dashed',
             label=r'$\tilde{f}(x)$')
    plt.xlabel(r'$x$')
    print('Distance between signals:', np.linalg.norm(res - np.ones(len(res))))
    print('x=', x)

    import scipy.stats as stats
    x = np.linspace(0, 2**J, 2**J, endpoint=False)
    phi = pywt.Wavelet(wavelet).wavefun(level=15)[0][1:]
    res = rf.recon_boundary(x, J, wavelet, phi)
    x1 = np.linspace(0, 2**J, len(res))[19980]
    y1 = res[19980]
    x2 = np.linspace(0, 2**J, len(res))[20000]
    y2 = res[20000]
    a = (y2 - y1) / (x2 - x1)
    b = y2 - (a * x2)
    print('a = ', a, 'b = ', b)
    y_end = a * 2**J + b
    a1 = (res[20] - res[0]) / (x2 - x1)
    a2 = (res[-1] - res[-21]) / (x2 - x1)
    r = a / a1
    r2 = a / a2
    x[1] = np.real(r) * x[1]
    x[-1] = x[-1] * np.real(r2)
    res2 = rf.recon_boundary(x, J, wavelet, phi)
    T_start = np.zeros(15, dtype=complex)
    RES_start = np.zeros(15, dtype=complex)
    T_end = np.zeros(15, dtype=complex)
    RES_end = np.zeros(15, dtype=complex)
    for i in range(15):
        t = i * 0.1
        x[0] = t
        x[-2] = t
        res2 = rf.recon_boundary(x, J, wavelet, phi)
        T_start[i] = t
        T_end[i] = t
        RES_start[i] = res2[0]
        RES_end[i] = res2[-1]
    Stat_start = stats.linregress(T_start, RES_start)
    Stat_end = stats.linregress(T_end, RES_end)
    t2 = (b - Stat_start[1]) / Stat_start[0]
    t3 = (y_end - Stat_end[1]) / Stat_end[0]
    x[0] = np.real(t2)
    x[-2] = np.real(t3)
    res2 = rf.recon_boundary(x, J, wavelet, phi)
    plt.plot(np.linspace(0, 2**J, 2**J),
             np.real(a * np.linspace(0, 2**J, 2**J, dtype=complex) + b),
             label=r'True Linear signal')
    plt.plot(np.linspace(0, 2**J, len(res2)), np.real(res2), ls='dashed',
             label=r'wavelet Reconstruction of Line')
    plt.legend()
    print('Distance between signals:',
          np.linalg.norm(
              np.real(res2) -
              np.real(a * np.linspace(0, 2**J, len(res2), dtype=complex) + b)))
    print('x=', x)


if __name__ == '__main__':
    test_of_con_func()
