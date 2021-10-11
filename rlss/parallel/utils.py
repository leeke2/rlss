# pylint: disable=missing-module-docstring
import functools
import typing as T
import operator
from multiprocessing.shared_memory import SharedMemory
import numpy as np

def create_shared_memory_array(
    shape: T.Union[T.List, T.Tuple],
    dtype: T.Union[T.List, np.dtype],
    shm: T.Optional[T.Union[T.List, SharedMemory]] = None,
    initialize: bool = False
) -> T.Tuple[T.Union[T.List, SharedMemory], T.Union[T.List, np.ndarray]]:
# pylint: disable=missing-function-docstring

    if isinstance(shape, T.List):
        if shm is not None:
            return tuple(zip(*[create_shared_memory_array(*args, initialize=initialize)
                              for args in zip(shape, dtype, shm)]))

        return tuple(zip(*[create_shared_memory_array(*args, initialize=initialize)
                          for args in zip(shape, dtype)]))

    if shm is not None:  # create from existing shared memory
        arr = np.ndarray(shape, buffer=shm.buf, dtype=dtype)
    else:
        num_elem = functools.reduce(operator.mul, shape)
        mem_size = num_elem * np.dtype(dtype).itemsize

        shm = SharedMemory(create=True, size=mem_size)
        arr = np.ndarray(shape, buffer=shm.buf, dtype=dtype)

        if initialize:
            arr[:] = np.zeros(shape)[:]

    return shm, arr
