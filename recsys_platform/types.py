from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator


def _uint32_vector(value: NDArray[np.uint32]) -> NDArray[np.uint32]:
    if value.dtype != np.dtype(np.uint32):
        raise ValueError("array dtype must be uint32")

    if value.ndim != 1:
        raise ValueError("array must be one-dimensional")

    return value


def _uint32_matrix(value: NDArray[np.uint32]) -> NDArray[np.uint32]:
    if value.dtype != np.dtype(np.uint32):
        raise ValueError("array dtype must be uint32")

    if value.ndim != 2:
        raise ValueError("array must be two-dimensional")

    return value


def _float32_matrix(value: NDArray[np.float32]) -> NDArray[np.float32]:
    if value.dtype != np.dtype(np.float32):
        raise ValueError("array dtype must be float32")

    if value.ndim != 2:
        raise ValueError("array must be two-dimensional")

    return value


UInt32Vector = Annotated[NDArray[np.uint32], AfterValidator(_uint32_vector)]
UInt32Matrix = Annotated[NDArray[np.uint32], AfterValidator(_uint32_matrix)]
Float32Matrix = Annotated[NDArray[np.float32], AfterValidator(_float32_matrix)]
