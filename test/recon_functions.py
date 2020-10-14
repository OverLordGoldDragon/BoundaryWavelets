# -*- coding: utf-8 -*-
"""
This is a module which contains reconstruction algorithms for the Daubechies
wavelets.

The boundary_wavelets.py package is licensed under the MIT "Expat" license.

Copyright (c) 2019: Josefine Holm and Steffen L. Nielsen.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pywt
import boundwave.boundary_wavelets as BW
import boundwave.orthonormal as Ot


# =============================================================================
# Functions
# =============================================================================

def recon_boundary(wavelet_coef, J, wavelet, phi):
    '''
    This function reconstructs a 1D signal in time from its wavelet
    coefficients, using boundary wavelets at the edge.

    INPUT:
        wavelet_coef : numpy.float64
            The wavelet decomposition. Can be made using decom_boundary().
        J : int
            The scale of the wavelet.
        avelet : str
            The name of the wavelet to use. For instance `'db2'`.
        phi : numpy.float64
            The scaling function at scale 0. (1d array)
    OUTPUT:
        x : numpy.float64
            The reconstructed signal, the length of the signal
            is `2**(N-J)*len(wavelet_coef)`.
    '''
    h = np.flipud(pywt.Wavelet(wavelet).dec_lo)
    AL, AR = Ot.ortho_matrix(J, h, phi)
    Q = BW.boundary_wavelets(phi, J, h, AL=AL, AR=AR)
    N = int(np.log2(len(phi) / (len(h) - 1)))
    phi = BW.downsample(phi, 0, 2**N, J, zero=False)
    OneStep = 2**(N - J)
    m = np.shape(Q)[0]
    a = np.shape(Q)[1] // 2
    Boundary = np.transpose(Q)
    x = np.zeros(OneStep * len(wavelet_coef), dtype=complex)
    for i in range(a):
        x[:m] += wavelet_coef[i] * Boundary[i]
    k = 1
    for i in range(a, len(wavelet_coef) - a):
        x[k * OneStep:k * OneStep +
            len(phi)] += wavelet_coef[i] * phi * 2**(-(J) / 2)
        k += 1
    for i in range(a):
        x[-m:] += wavelet_coef[-i - 1] * Boundary[-i - 1]
    return x


def recon_mirror(wavelet_coef, J, wavelet, phi):
    '''
    This function reconstructs a 1D signal in time from its wavelet
    coefficients, using mirroring of the signal at the edge.

    INPUT:
        wavelet_coef : numpy.float64
            The wavelet decomposition. Can be made using
            decom_mirror().
        J : int
            The scale of the wavelet.
        wavelet : str
            The name of the wavelet to use. For instance `'db2'`.
        phi : numpy.float64
            The scaling function at scale 0. (1d array)
    OUTPUT:
        x : numpy.float64
            The reconstructed signal, the length of the signal
            is `2**(N-J)*len(wavelet_coef)`.
    '''

    h = np.flipud(pywt.Wavelet(wavelet).dec_lo)
    N = int(np.log2(len(phi) / (len(h) - 1)))
    phi = BW.downsample(phi, 0, 2**N, J, zero=False)
    OneStep = 2**(N - J)
    a = int(len(phi) / OneStep) - 1
    x = np.zeros(OneStep * len(wavelet_coef) + (a) * OneStep, dtype=complex)
    for i in range(len(wavelet_coef)):
        x[i * OneStep:i * OneStep +
            len(phi)] += wavelet_coef[i] * phi * 2**(-(J) / 2)
    x = x[OneStep * (a):-OneStep * (a)]
    return x


def decom_boundary(signal, J, wavelet, phi):
    '''
    This function makes a wavelet decomposition of a 1D signal in
    time, using boundary wavelets at the edge.

    INPUT:
        signal : numpy.float64
            The signal to be decomposed.
        J : int
            The scale of the wavelet.
        wavelet : str
            The name of the wavelet to use. For instance `'db2'`.
        phi : numpy.float64
            The scaling function at scale 0. (1d array)
    OUTPUT:
        x : numpy.float64
            The decomposition.
    '''
    h = np.flipud(pywt.Wavelet(wavelet).dec_lo)
    a = int(len(h) / 2)
    N = int(np.log2(len(phi) / (len(h) - 1)))
    AL, AR = Ot.ortho_matrix(J, h, phi)
    Boundary = BW.boundary_wavelets(phi, J, h, AL=AL, AR=AR)
    x = np.zeros(2**J)
    for i in range(a):
        x[i] = np.inner(Boundary[:, i], signal)
    for i in range(1, 2**J - 2 * a + 1):
        x[i - 1 + a] = np.inner(BW.downsample(phi, i, 2**N, J, zero=True) *
                                np.sqrt(2**J), signal)
    for i in range(a):
        x[-1 - i] = np.inner(Boundary[:, -1 - i], signal)
    x /= len(signal)
    return x


def decom_mirror(signal, J, wavelet, phi):
    '''
    This function makes a wavelet decomposition of a 1D signal in
    time, using mirroring of the signal at the edge.

    INPUT:
        signal : numpy.float64
            The signal to be decomposed.
        J : int
            The scale of the wavelet.
        wavelet : str
            The name of the wavelet to use. For instance `'db2'`.
        phi : numpy.float64
            The scaling function at scale 0. (1d array)
    OUTPUT:
        x : numpy.float64
            The decomposition.
    '''
    h = np.flipud(pywt.Wavelet(wavelet).dec_lo)
    N = int(np.log2(len(phi) / (len(h) - 1)))
    OneStep = 2**(N - J)
    a = int(len(h) / 2)
    x = np.zeros(2**J + (2 * a - 2))
    for i in range(2 * a - 2):
        phi1 = BW.downsample(phi, 0, 2**N, J, zero=False) * np.sqrt(2**J)
        signal1 = np.concatenate(
            (np.flipud(signal[:OneStep * (i + 1)]), signal))
        signal1 = signal1[:len(phi1)]
        x[i] = np.inner(phi1, signal1)
    for i in range(2**J - 2 * a + 2):
        x[i + 2 * a - 2] = np.inner(BW.downsample(phi, i, 2**N, J, zero=True) *
                                    np.sqrt(2**J), signal)
    for i in range(2 * a - 2):
        phi1 = BW.downsample(phi, 0, 2**N, J, zero=False) * np.sqrt(2**J)
        signal1 = np.concatenate(
            (signal, np.flipud(signal[-OneStep * (i + 1):])))
        signal1 = signal1[-len(phi1):]
        x[2**J + i] = np.inner(phi1, signal1)
    x /= len(signal)
    return x
