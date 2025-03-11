
import numpy as np
cimport numpy as np
from numpy.fft import fftn, ifftn

def fftn_cython(double[:, :] v):
    return fftn(v)

def ifftn_cython(double complex[:, :] v_hat):
    return ifftn(v_hat)