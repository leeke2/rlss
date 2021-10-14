# pylint: disable=missing-module-docstring
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from .utils import create_shared_memory_array

class ReplayMemory: # pylint: disable=missing-class-docstring
    def __init__(self, dtypes, shapes, buffer_size: int = 1_000): # pylint: disable=missing-function-docstring
        self.dtypes = dtypes
        self.shapes = shapes

        shapes = [(shape[0] * buffer_size, *shape[1:]) for shape in shapes]

        self.status_shm, self.ready = create_shared_memory_array(shape=(buffer_size, 1), dtype=np.bool)
        self.shms, self.buffer = create_shared_memory_array(shape=shapes, dtype=dtypes, initialize=True)
        
        self.buffer_len_shm = SharedMemory(create=True, size=np.dtype(np.uint32).itemsize)
        self.buffer_len = np.ndarray((1, ), buffer=self.buffer_len_shm.buf, dtype=np.uint32)

        self.sps_shm = SharedMemory(create=True, size=np.dtype(np.float32).itemsize)
        self.sps_arr = np.ndarray((1, ), buffer=self.sps_shm.buf, dtype=np.float32)

        self.buffer_size = buffer_size

    def __len__(self):
        return self.buffer_len[0]

    @property
    def sps(self):
        return self.sps_arr[0]

    @property
    def recreate_sps_fn(self):
        return lambda: (self.sps_shm, np.ndarray((1, ), buffer=self.sps_shm.buf, dtype=np.float32))

    @property
    def recreate_buffer_len_fn(self):
        return lambda: (self.buffer_len_shm, np.ndarray((1, ), buffer=self.buffer_len_shm.buf, dtype=np.uint32))

    @property
    def recreate_buffer_fn(self):
        shapes = [(shape[0] * self.buffer_size, *shape[1:]) for shape in self.shapes]
        return lambda: create_shared_memory_array(shape=shapes, dtype=self.dtypes, shm=self.shms)

    @property
    def recerate_ready_fn(self):
        return lambda: create_shared_memory_array(shape=(self.buffer_size, 1), dtype=np.bool, shm=self.status_shm)

    def sample(self, batch_size: int): # pylint: disable=missing-function-docstring
        idx = np.random.choice(range(self.buffer_len[0]), size=batch_size)
        return [buf[idx] for buf in self.buffer]

    def close(self): # pylint: disable=missing-function-docstring
        for shm in [self.buffer_len_shm, self.status_shm, self.sps_shm] + list(self.shms):
            shm.close()
            shm.unlink()