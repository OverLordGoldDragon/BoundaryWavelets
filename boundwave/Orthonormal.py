# -*- coding: utf-8 -*-
"""
This module is used for calculations of the orthonormalization matrix for
the boundary wavelets.

The boundary_wavelets.py package is licensed under the MIT "Expat" license.

Copyright (c) 2019: Josefine Holm and Steffen L. Nielsen.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
from scipy.integrate import simps
import boundwave.boundary_wavelets as BW


# =============================================================================
# Functions
# =============================================================================
def integral(J, k, l, wavelet_coef, phi):
    r'''
    This function calculates the integral (16) numerically.

    INPUT:
        J : int
            The scale.
        k : int
            The translation for the first function.
        l : int
            The translation for the second function.
        wavelet_coef : numpy.float64
            The wavelet coefficients, must sum to :math:`\sqrt{2}`.
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        phi : numpy.float64
            The phi function, can be made with
            `pywt.Wavelet(wavelet).wavefun(level=15)`.
    OUTPUT:
        out : int
            The value of the integral.

    '''

    a = int(len(wavelet_coef) / 2)
    OneStep = len(phi) // (2 * a - 1)
    phiNorm = np.linalg.norm(BW.downsample(phi, 0, OneStep, J))
    phi1 = BW.downsample(phi, k, OneStep, J) / phiNorm
    phi2 = BW.downsample(phi, l, OneStep, J) / phiNorm
    phiProd = phi1 * phi2
    Integ = simps(phiProd)
    return Integ


def m_alpha_beta(alpha, beta, J, wavelet_coef, inte_matrix, Side):
    r'''
    This function calculates an entry in the martix :math:`M` (15).

    INPUT:
        alpha : int
            alpha
        beta : int
            beta
        J : int
            The scale.
        wavelet_coef : numpy.float64
            The wavelet coefficients, must sum to :math:`\sqrt{2}`. For
            Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo`).
        inte_matrix : numpy.float64
            A matrix with the values for the integrals calculated with
            the function :py:func:`integral` for k and l in the
            interval [-2*a+2,0] or [2**J-2*a+1,2**J-1].
        Side : str
            `'L'` for left interval boundary and `'R'` for right
            interval boundary.
    OUTPUT:
        M : numpy.float64
            Entry (alpha,beta) of the martix M

    '''

    a = int(len(wavelet_coef) / 2)
    Moment = BW.moments(wavelet_coef, a - 1)
    M = 0
    if Side == 'L':
        interval = range(-2 * a + 2, 1)
        i = 0
        for k in interval:
            j = 0
            for m in interval:
                M += (BW.inner_product_phi_x(alpha, 0, k, Moment) *
                      BW.inner_product_phi_x(beta, 0, m, Moment) *
                      inte_matrix[i, j])
                j += 1
            i += 1
    elif Side == 'R':
        interval = range(2**J - 2 * a + 1, 2**J)
        i = 0
        for k in interval:
            j = 0
            for m in interval:
                M += (BW.inner_product_phi_x(alpha, 0, k, Moment) *
                      BW.inner_product_phi_x(beta, 0, m, Moment) *
                      inte_matrix[i, j] * 2**(-J * (alpha + beta)))
                j += 1
            i += 1
    else:
        print('You must choose a side')

    return M


def ortho_matrix(J, wavelet_coef, phi):
    r'''
    This function findes the orthogonality matrix :math:`A`. First
    uses the functions :py:func:`m_alpha_beta` and :py:func:`integral`
    to make the matrix M. Then computes a Cholesky decomposition,
    which is then inverted.

    INPUT:
        J : int
            The scale.
        wavelet_coef : numpy.float64
            The wavelet coefficients, must sum to
            :math:`\sqrt{2}`. For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        phi : numpy.float64
            The phi function, can be made with
            `pywt.Wavelet(wavelet).wavefun(level=15)`.
    OUTPUT:
        AL : numpy.float64
            Left orthonormalisation matrix; to be used in
            :py:func:`boundwave.boundary_wavelets.boundary_wavelets` or
            :py:func:`boundwave.fourier_boundary_wavelets.fourier_boundary_wavelets`.
        AR : numpy.float64
            Right orthonormalisation matrix; to be used in
            :py:func:`boundwave.boundary_wavelets.boundary_wavelets` or
            :py:func:`boundwave.fourier_boundary_wavelets.fourier_boundary_wavelets`.

    '''

    a = int(len(wavelet_coef) / 2)
    ML = np.zeros((a, a))
    MR = np.zeros((a, a))
    InteL = np.zeros((2 * a - 1, 2 * a - 1))
    k = 0
    for i in range(-2 * a + 2, 1):
        m = 0
        for j in range(-2 * a + 2, i + 1):
            InteL[k, m] = integral(J, i, j, wavelet_coef, phi)
            InteL[m, k] = InteL[k, m]
            m += 1
        k += 1
    InteR = np.zeros((2 * a - 1, 2 * a - 1))
    k = 0
    for i in range(2**J - 2 * a + 1, 2**J):
        m = 0
        for j in range(2**J - 2 * a + 1, i + 1):
            InteR[k, m] = integral(J, i, j, wavelet_coef, phi)
            InteR[m, k] = InteR[k, m]
            m += 1
        k += 1
    for i in range(a):
        for j in range(i + 1):
            ML[i, j] = m_alpha_beta(i, j, J, wavelet_coef, InteL, 'L')
            ML[j, i] = ML[i, j]
    for i in range(a):
        for j in range(i + 1):
            MR[i, j] = m_alpha_beta(i, j, J, wavelet_coef, InteR, 'R')
            MR[j, i] = MR[i, j]
    h = 2**(J * np.arange(a))
    CL = np.linalg.cholesky(ML)
    AL = 2**(J / 2) * np.dot(np.linalg.inv(CL), np.diag(h))
    CR = np.linalg.cholesky(MR)
    U, S, V = np.linalg.svd(CR)
    AR = 2**(J / 2) * np.dot(np.dot(np.transpose(V), np.diag(1 / S)),
                             np.transpose(U))
    return AL, AR
