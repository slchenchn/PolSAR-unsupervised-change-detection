'''
Author: Shuailin Chen
Created Date: 2021-05-18
Last Modified: 2021-05-19
	content: adapt from Song Hui's matlab code
'''

import os.path as osp

import numpy as np

from mylib import polSAR_utils as psr


def determinant(A):
    ''' Calculate determinant of a C3 matrix

    Args:
        A (ndarray): PolSAR data

    Returns:
        det (ndarray): determint
    '''

    A = psr.as_format(A, 'complex_vector_9')
    det = A[0, ...] * A[4, ...] * A[8, ...] + A[1, ...] * A[5, ...] \
        * A[6, ...] + A[2, ...] * A[3, ...] * A[7, ...] - A[2, ...] \
        * A[4, ...] * A[6, ...] - A[1, ...] * A[3, ...] * A[8, ...] \
        - A[0, ...] * A[5, ...] * A[7, ...]
    
    return det


def inverse(A):
    ''' Calculate inverse matrix of a C3 matrix

    Args:
        A (ndarray): PolSAR data

    Returns:
        inv (ndarray): inverse matrix
    '''

    A = psr.as_format(A, 'complex_vector_9')
    confA = np.zeros_like(A)
    confA[0, ...] = A[4, ...] * A[8, ...] - A[5, ...] * A[7, ...]
    confA[1, ...] = -(A[3, ...] * A[8, ...] - A[5, ...] * A[6, ...])
    confA[2, ...] = A[3, ...] * A[7, ...] - A[4, ...] * A[6, ...]
    confA[3, ...] = -(A[1, ...] * A[8, ...] - A[2, ...] * A[7, ...])
    confA[4, ...] = A[0, ...] * A[8, ...] - A[2, ...] * A[6, ...]
    confA[5, ...] = -(A[0, ...] * A[7, ...] - A[1, ...] * A[6, ...])
    confA[6, ...] = A[1, ...] * A[5, ...] - A[2, ...] * A[4, ...]
    confA[7, ...] = -(A[0, ...] * A[5, ...] - A[2, ...] * A[3, ...])
    confA[8, ...] = A[0, ...] * A[4, ...] - A[1, ...] * A[3, ...]

    adjA = np.zeros_like(A)
    for m in range(1, 4):
        for n in range(1, 4):
            adjA[(m-1)*3+n-1, ...] = confA[(n-1)*3+m-1, ...]

    det = determinant(A)
    P = 9
    inv = adjA / np.tile(det[np.newaxis], (P, 1, 1))

    return inv


def distance_by_c3(A, B, type):
    ''' Pixel-Level Difference Map, by calculating the pixelwise similarities of C3 data between two PolSAR images

    Args:
        A/B (ndarray): PolSAR data
        type (str): distance metric type, 'Bartlett' or 'rw' (revised Wishart)
            or 'srw' (symmetric revised Wishart)

    Returns:
        difference map, in shape like Arg A's 
    '''

    q = 3
    A = psr.as_format(A, 'complex_vector_9')
    B = psr.as_format(B, 'complex_vector_9')

    if type == 'Bartlett':
        logdetA = 0.5*np.real(np.log(np.abs(determinant(A))))
        logdetB = 0.5*np.real(np.log(np.abs(determinant(B))))
        D = np.log(np.abs(determinant((A+B)))) - (logdetA+logdetB)

    elif type in ('srw', 'symmetric revised Wishart'):
        iA = inverse(A)
        iB = inverse(B)
        D = np.real(
            np.sum(iA * B[[0, 3, 6, 1, 4, 7, 2, 5, 8], ...], axis=0) \
            + np.sum(iB * A[[0, 3, 6, 1, 4, 7, 2, 5, 8], ...], axis=0)
            )
        D = 0.5*D - q
    
    elif type in ('rw', 'revised Wishart'):
        logdetB = np.real(np.log(np.abs(determinant(B))))
        logdetA = np.real(np.log(np.abs(determinant(A))))
        iB = inverse(B)
        iB = iB[[0, 3, 6, 1, 4, 7, 2, 5, 8], ...]
        D = logdetB - logdetA + np.sum(iB*A, 0) - q
        D = np.real(D)

    return D
    
    
if __name__ == '__main__':
    fa = r'data/2009_SUB_SUB/C3'
    fb = r'data/2010_SUB_SUB/C3'

    c31 = psr.read_c3(fa)
    c32 = psr.read_c3(fb)

    # print(determinant(np.expand_dims(c31[:, :15, 0], 2)))
    # print(np.squeeze(inverse(np.expand_dims(c31[:, :15, 0], 2)).T))

    c31 = np.expand_dims(c31[:, :15, 0], 2)
    c32 = np.expand_dims(c32[:, :15, 0], 2)
    print(distance_by_c3(c31, c32, 'srw'))    