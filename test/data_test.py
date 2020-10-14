# -*- coding: utf-8 -*-
"""
This module is used for testing the boundary wavelets on ECG data.

The boundary_wavelets.py package is licensed under the MIT "Expat" license.

Copyright (c) 2019: Josefine Holm and Steffen L. Nielsen.
"""

import scipy.io as sp
import numpy as np
import pywt
import matplotlib.pyplot as plt
import recon_functions as rf


def test_plot(Name='Data', row=1, section=214, J=7, N=12, wavelet='db3'):
    '''
    This function makes decompositions and reconstructions of a chosen
    section of the data, both with boundary wavelets and with mirrored
    extension. The difference between the orignal signal and the two
    reconstructions are calculated and printed and all three signals
    are plotted in the same figure.

    INPUT:
        Name : str
            The MATLAB data file from which to load.
        row : int
            The row in the dataset to use.
        section : int
            Which section of the data to use. The samples that will be
            used are: `[section*2**N:section*2**N+2**N]`.
        J : int
            The scale.
        N : int
            The number of iterations to use in the cascade algorithm.
        wavelet : str
            The name of the wavelet to be used. eg: `'db2'`.
    '''

    data = sp.loadmat(Name)
    phi = pywt.Wavelet(wavelet).wavefun(level=14)[0][1:]
    phi = phi[::2**(14 - N)]
    signal = data['val'][row, section * 2**N:section * 2**N + 2**N]
    x1 = rf.decom_boundary(signal, J, wavelet, phi)
    new_signal1 = np.real(rf.recon_boundary(x1, J, wavelet, phi))
    x2 = rf.decom_mirror(signal, J, wavelet, phi)
    new_signal2 = np.real(rf.recon_mirror(x2, J, wavelet, phi))
    dif1 = np.sum(np.abs(signal - new_signal1)**2)**(1 / 2) / 2**N
    dif2 = np.sum(np.abs(signal - new_signal2)**2)**(1 / 2) / 2**N
    print(dif1, dif2)

    plt.figure()
    plt.plot(signal, label='Original')
    plt.plot(new_signal1, label='Boundary wavelets')
    plt.plot(new_signal2, label='Mirror')
    plt.xlabel('Sample index')
    plt.legend()
    return


def test(Name='Data', row=1, J=7, N=12, wavelet='db3'):
    '''
    This function makes decompositions and reconstructions of several
    sections of the data, both with boundary wavelets and with
    mirrored extension. The differences between the orignal signal and
    the two reconstructions are calculated. The test is run for as
    many disjoint sections of the signal as possible.

    INPUT:
        Name : str
            The MATLAB data file from whichto load.
        row : int
            The row in the dataset to use.
        J : int
            The scale.
        N : int
            The number of iterations to use in the cascade algorithm.
        wavelet : str
            The name of the wavelet to be used. eg: `'db2'`.
    OUTPUT:
        result : float64
            2D array. The first row is the difference between the
            original signal and the reconstruction using boundary
            wavelet. The second row is the difference between the
            original signal and the reconstruction using mirrored
            extension. The third row is the first row minus the second
            row. There is one collumn for each section of the signal.
    '''

    data = sp.loadmat(Name)
    phi = pywt.Wavelet(wavelet).wavefun(level=14)[0][1:]
    phi = phi[::2**(14 - N)]
    n = 0
    tests = int(len(data['val'][row]) / 2**N)
    result = np.zeros((3, tests))
    for i in range(tests):
        signal = data['val'][row, n:n + 2**N]
        x1 = rf.decom_boundary(signal, J, wavelet, phi)
        x2 = rf.decom_mirror(signal, J, wavelet, phi)
        new_signal1 = np.real(rf.recon_boundary(x1, J, wavelet, phi))
        new_signal2 = np.real(rf.recon_mirror(x2, J, wavelet, phi))
        result[0, i] = np.sum(np.abs(signal - new_signal1)**2)**(1 / 2) / 2**N
        result[1, i] = np.sum(np.abs(signal - new_signal2)**2)**(1 / 2) / 2**N
        n += 2**N
    result[2] = result[0] - result[1]
    plt.figure()
    plt.plot(result[1], label='Mirror', color='C1')
    plt.plot(result[0], label='Boundary', color='C0')
    plt.xlabel('Test signal')
    plt.ylabel('Difference')
    plt.legend()
    return result


if __name__ == '__main__':
    test_plot()
#    Test = test()
