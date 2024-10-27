import numpy as np
from scipy.optimize import curve_fit

import tensorflow as tf

from calc import HARTRI_TO_KCAL

def pes(
    x : np.ndarray, 
    A11 : float, A21 : float, A31 : float,
    b11 : float, b21 : float, b31 : float,
    c1 : float
) -> np.ndarray:
    """
        Function, that could describe PES
    """
    return A11 * np.cos(b11 * x[:, 0]) + A21 * np.cos(2 * b21 * x[:, 0]) + A31 * np.cos(3 * b31 * x[:, 0]) + c1

def pes_tf(
    x : tf.Tensor,
    A11 : float, A21 : float, A31 : float,
    b11 : float, b21 : float, b31 : float,
    c1 : float
) -> tf.Tensor:
    return A11 * tf.cos(b11 * x) + A21 * tf.cos(2 * b21 * x) + A31 * tf.cos(3 * b31 * x) + c1

def pes_tf_grad(
    x : tf.Tensor,
    A11 : float, A21 : float, A31 : float,
    b11 : float, b21 : float, b31 : float,
    c1 : float
) -> tf.Tensor:
    return -(A11 * b11 * tf.sin(b11 * x) + A21 * 2 * b21 * tf.sin(2 * b21 * x) + A31 * 3 * b31 * tf.sin(3 * b31 * x))

def calc_coefs(
    x : np.ndarray,
    y : np.ndarray,
) -> np.ndarray:
    """
        x - observed points [N, inp_dims]
        y - observed signal [N]
        returns [7, inp_dims] array of coefs
    """ 
    y = (y - y.mean()) * HARTRI_TO_KCAL
    coefs, cov_matrix = curve_fit(pes, x, y, p0=np.ones(7), maxfev=10000)
    return coefs
