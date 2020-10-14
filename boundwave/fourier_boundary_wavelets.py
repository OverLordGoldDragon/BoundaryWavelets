# -*- coding: utf-8 -*-
"""
This module is used for calculations of the boundary wavelets in the frequency
domain.

The boundary_wavelets.py package is licensed under the MIT "Expat" license.

Copyright (c) 2019: Josefine Holm and Steffen L. Nielsen.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import boundwave.boundary_wavelets as bw


# =============================================================================
# Functions
# =============================================================================
def rectangle(scheme):
    '''
    The Fourier transform of a rectangular window function.

    INPUT:
        scheme : numpy.float64
            A numpy array with the frequencies in which to sample.
    OUTPUT:
        chi : numpy.complex128
            A numpy array with the window function sampled in the
            freqency domain.

    '''

    chi = np.zeros(len(scheme), dtype=np.complex128)
    for i in range(len(scheme)):
        if scheme[i] == 0:
            chi[i] = 1
        else:
            chi[i] = (1 - np.exp(-2 * np.pi * 1j * scheme[i])) / \
                (2 * np.pi * 1j * scheme[i])
    return chi


def scaling_function_fourier(wavelet_coef, J, k, scheme, win, P=20):
    r'''
    This function evaluates the Fourier transform of the scaling function,
    :math:`\phi_{j,k}`, sampled in scheme.

    INPUT:
        wavelet_coef : numpy.float64
            The wavelet coefficients, must sum to :math:`\sqrt{2}`.
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        J : int
            The scale.
        k : int
            The translation.
        scheme : numpy.float64
            The points in which to evaluate.
        window=rectangle : numpy.complex128
            The window to use on the boundary functions.
        P=20 : int
            The number of factors to include in the infinite product
            in the Fourier transform of phi.
    OUTPUT:
        phi : numpy.complex128
            :math:`\hat{\phi}_{j,k}`

    '''

    h = wavelet_coef * np.sqrt(2) / 2
    e = (scheme[-1] - scheme[0]) / len(scheme)
    phi = np.zeros((len(scheme), P, len(h)), dtype=complex)
    for i in range(P):
        for l in range(len(h)):
            phi[:, i, l] = h[l] * \
                np.exp(-2 * np.pi * 1j * l * 2**(-i - J - 1) * scheme)
    phi = np.sum(phi, axis=2, dtype=np.complex128)
    phi = (2**(-J / 2) * np.exp(-2 * np.pi * 1j * k * 2**(-J) * scheme) *
           np.prod(phi, axis=1, dtype=np.complex128))
    PhiAstChi = np.convolve(phi, win, mode='same') * e
    return PhiAstChi


def fourier_boundary_wavelets(J, scheme, wavelet_coef, AL=None, AR=None,
                              win=rectangle):
    r'''
    This function evaluates the Fourier transformed boundary functions
    for db2.

    INPUT:
        J : int
            The scale.
        scheme : numpy.float64
            The sampling scheme in the Fourier domain.
        wavelet_coef : numpy.float64
            The wavelet coefficients, must sum to :math:`\sqrt{2}`.
            For Daubeshies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        AL=None : numpy.float64
            The left orthonormalisation matrix, if this is not
            supplied the functions will not be orthonormalized. Can be
            computed using
            :py:func:`boundwave.Orthonormal.ortho_matrix`.
        AR=None : numpy.float64
            The right orthonormalisation matrix, if this is not
            supplied the functions will not be orthonormalized. Can be
            computed using
            :py:func:`boundwave.Orthonormal.ortho_matrix`.
        win= :py:func:`rectangle` : numpy.complex128
            The window to use on the boundary functions.
    OUTPUT:
        x : numpy.complex128
            2d numpy array with the boundary functions in the columns;
            orthonormalised if `AL` and `AR` given.

    '''

    a = int(len(wavelet_coef) / 2)
    kLeft = np.arange(-2 * a + 2, 1)
    kRight = np.arange(2**J - 2 * a + 1, 2**J)
    xj = np.zeros((len(scheme), 2 * a), dtype=complex)
    Moment = bw.moments(wavelet_coef, a - 1)
    FourierPhiLeft = np.zeros((len(kLeft), len(scheme)), dtype=complex)
    FourierPhiRight = np.zeros((len(kRight), len(scheme)), dtype=complex)
    window = win(scheme)
    for i in range(len(kLeft)):
        FourierPhiLeft[i] = scaling_function_fourier(wavelet_coef, J,
                                                     kLeft[i], scheme, window)
        FourierPhiRight[i] = scaling_function_fourier(wavelet_coef, J,
                                                      kRight[i], scheme, window)
    for b in range(a):
        xj[:, b] = np.sum(np.multiply(bw.inner_product_phi_x(
            b, J, kLeft, Moment), np.transpose(FourierPhiLeft)), axis=1)
        xj[:, b + a] = np.sum(np.multiply(bw.inner_product_phi_x(
            b, J, kRight, Moment), np.transpose(FourierPhiRight)), axis=1)
    if AL is None or AR is None:
        return xj
    else:
        x = np.zeros(np.shape(xj), dtype=complex)
        for i in range(a):
            for j in range(a):
                x[:, i] += xj[:, j] * AL[i, j]
        for i in range(a):
            for j in range(a):
                x[:, i + a] += xj[:, j + a] * AR[i, j]
        return x
