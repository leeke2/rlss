import numpy as np
from typing import Optional


class BaseInstance:
    def __init__(self):
        pass


class StopSkipInstance(BaseInstance):
    def __init__(self, demands: np.ndarray, travel_time: Optional[np.ndarray]) -> None:
        self._demands = demands
        if travel_time is not None:
            self._travel_time = travel_time

    @property
    def demands(self):
        return self._demands

    @property
    def travel_time(self):
        return self._travel_time

    @property
    def has_travel_time(self):
        return '_travel_time' in self.__dict__
