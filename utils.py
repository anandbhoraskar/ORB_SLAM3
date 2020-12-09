from typing import List, Sequence, Tuple
from urllib.request import urlopen
from zipfile import ZipFile

import magnum as mn
import numpy as np
import quaternion


def quat_from_coeffs(coeffs: Sequence[float]) -> np.quaternion:
    r"""Creates a quaternion from the coeffs returned by the simulator backend
    :param coeffs: Coefficients of a quaternion in :py:`[b, c, d, a]` format,
        where :math:`q = a + bi + cj + dk`
    :return: A quaternion from the coeffs
    """
    quat = np.quaternion(1, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def quat_to_coeffs(quat: np.quaternion) -> np.ndarray:
    r"""Converts a quaternion into the coeffs format the backend expects
    :param quat: The quaternion
    :return: Coefficients of a quaternion in :py:`[b, c, d, a]` format,
        where :math:`q = a + bi + cj + dk`
    """
    coeffs = np.empty(4)
    coeffs[0:3] = quat.imag
    coeffs[3] = quat.real
    return coeffs
