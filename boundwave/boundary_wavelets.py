# -*- coding: utf-8 -*-
"""
This module is used for calculations of the boundary wavelets in the time
domain.

The boundary_wavelets.py package is licensed under the MIT "Expat" license.

Copyright (c) 2019: Josefine Holm and Steffen L. Nielsen.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
from scipy.special import binom


# =============================================================================
# Functions
# =============================================================================
def downsample(x, Shift, N, J, zero=True):
    """
    This is a function which dilates a signal. Make sure there are
    enough samples for it to make sense.

    INPUT:
        x : numpy.float64
            1d numpy array, signal to be dilated.
        Shift : int
            Time shift before dilation (for now it only supports
            integers).
        N : int
            Number of samples per 'time' unit.
        J : int
            The scale to make. Non-negative integer.
        zero=True : bool
            If true, it concatenates zeros on the signal to retain the
            original length.
    OUTPUT:
        y : numpy.float64
            The scaled version of the scaling function.

    """
    if J == 0:
        if Shift < 0:
            x1 = np.concatenate(
                (x[-Shift * N:], np.zeros(N - len(x[-Shift * N:]))))
        else:
            x1 = np.concatenate((np.zeros(Shift * N), x))
        return x1[:N]
    if Shift <= 0:
        x1 = x[-Shift * N:]
        xhat = np.sqrt(2**J) * x1[::2**J]
        if zero:
            y = np.concatenate((xhat, np.zeros(N - len(xhat))))
        else:
            y = xhat
    else:
        x1 = np.concatenate((np.zeros(Shift * N), x))
        xhat = np.sqrt(2**J) * x1[::2**J]
        if len(xhat) > N:
            y = xhat[:N]
        elif zero:
            y = np.concatenate((xhat, np.zeros(N - len(xhat))))
        else:
            y = xhat
    return y


def moments(wavelet_coef, n):
    r'''
    This function calculates the moments of :math:`\phi` up to power
    `n`, i.e. :math:`\langle x^l, \phi \rangle`, for :math:`0 \le l
    \le n`.

    INPUT:
        wavelet_coef : numpy.float64
            The wavelet coefficients, must sum to :math:`\sqrt{2}`.
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        n : int
            The highest power moment to calculate.
    OUTPUT:
        moments : numpy.float64
            A 1d array with the moments.
    '''

    moments = np.ones(n + 1)
    for l in range(1, n + 1):
        for k in range(len(wavelet_coef)):
            for m in range(l):
                moments[l] += 1 / ((2**l - 1) * np.sqrt(2)) * (
                    wavelet_coef[k] * binom(l, m) * (k + 1)**(l - m) * moments[m])
    return moments


def inner_product_phi_x(alpha, J, k, moments):
    r'''
    This function calculates the inner product between
    :math:`x^\alpha` and :math:`\phi_{J,k}`.

    INPUT:
        alpha : int
            The power of :math:`x`
        J, k : int
            The indices for :math:`\phi`.
        moments : numpy.float64
            A 1d array of moments for :math:`\phi`, up to power
            `alpha`. Can be calculated using the function
            :py:func:`moments`.
    OUTPUT:
        i : numpy.float64
            The inner product.

    '''

    i = 0
    for l in range(alpha + 1):
        i += 2**(-J + J / 2 - J * alpha) * \
            binom(alpha, l) * k**(alpha - l) * moments[l]
    return i


def boundary_wavelets(phi, J, wavelet_coef, AL=None, AR=None):
    r'''
    This function evaluates the left boundary functions.

    INPUT:
        phi : numpy.float64
            The scaling function at scale 0. (1d array)
        J : int
            The scale the scaling function has to have.
        wavelet_coef : numpy.float64
            The wavelet coefficients must sum to :math:`\sqrt{2}`.
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        AL=None : numpy.float64
            The left orthonormalisation matrix, if this is not suplied
            the functions will not be orthonormalized. Can be computed
            using :py:func:`boundwave.Orthonormal.ortho_matrix`.
        AR=None : numpy.float64
            The right orthonormalisation matrix, if this is not
            suplied the functions will not be orthonormalized. Can be
            computed using
            :py:func:`boundwave.Orthonormal.ortho_matrix`.
    OUTPUT:
        x : numpy.float64
            2d numpy array with the boundary functions in the columns;
            orthonormalised if `AL` and `AR` given.

    '''

    a = int(len(wavelet_coef) / 2)
    kLeft = np.arange(-2 * a + 2, 1)
    kRight = np.arange(2**J - 2 * a + 1, 2**J)
    Moment = moments(wavelet_coef, a - 1)
    OneStep = len(phi) // (2 * a - 1)
    xj = np.zeros((OneStep, 2 * a))
    PhiLeft = np.zeros((len(kLeft), OneStep))
    PhiRight = np.zeros((len(kRight), OneStep))
    for i in range(len(kLeft)):
        PhiLeft[i] = downsample(phi, kLeft[i], OneStep, J)
        PhiRight[i] = downsample(phi, kRight[i], OneStep, J)
    for b in range(a):
        for k in range(len(kLeft)):
            xj[:int((2 * a - 1 + kLeft[k]) * 2**(-J) * OneStep), b] += (
                inner_product_phi_x(b, J, kLeft[k], Moment) *
                PhiLeft[k, :int((2 * a - 1 + kLeft[k]) * 2**(-J) * OneStep)])
            xj[int(2**(-J) * kRight[k] * OneStep):, a + b] += (
                inner_product_phi_x(b, J, kRight[k], Moment) *
                PhiRight[k, int(2**(-J) * kRight[k] * OneStep):])
    if AL is None or AR is None:
        return xj
    else:
        x = np.zeros(np.shape(xj))
        for i in range(a):
            for j in range(a):
                x[:, i] += xj[:, j] * AL[i, j]
        for i in range(a):
            for j in range(a):
                x[:, i + a] += xj[:, j + a] * AR[i, j]
        return x
