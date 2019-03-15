from typing import List, Dict, Tuple
import numpy as np

Point3 = Tuple[int, int, int]
BoundBox = List[slice]
NodeData = Dict[Point3, Tuple[np.ndarray, BoundBox]]

"""
This class contains information about the Calyx volume used by the
futusa group

TODO: Add a sensitives file that keeps track of the file location
since that is also CALYX specific
"""


class BoundingBox:
    def __init__(
        self, start: np.ndarray, end: np.ndarray = None, shape: np.ndarray = None
    ):
        if end is None and shape is None:
            raise ValueError("Shape or End must be given!")
        self.start = start
        self.end = end or start + shape

    def contains_point(self, point: np.ndarray) -> bool:
        return all(self.start <= point) and all(self.end > point)

    def contains_point_with_radius(self, point: np.ndarray, radius: np.ndarray) -> bool:
        return self.contains_point(point - radius) and self.contains_point(
            point + radius
        )

    def contains_roi(self, start: np.ndarray, shape: np.ndarray):
        return self.contains_point(start) and self.contains_point(start + shape)

    def __str__(self):
        return "{} - {}".format(self.start, self.end)


class Calyx:
    def __init__(self, constants: Dict = {}):
        self.constants = constants

    @property
    def translation_phys(self):
        """
        Data translation between input coordinates (from CATMAID) to
        segmentation coordinates

        There is a one slice difference between catmaid and the fafb_v14 n5 volume
        used by the futusa group
        """
        return self.constants.get("translation", np.array([0, 0, -40]))

    @property
    def start(self) -> np.ndarray:
        """
        Coordinates in X,Y,Z order with units in nano-meters
        default: 403560, 121800, 158000
        """
        return self.constants.get("start", np.array([403560, 121800, 158000]))

    @property
    def shape(self) -> np.ndarray:
        """
        Shape in X,Y,Z order with units in nano-meters
        default: 64000, 52000, 76000
        """
        return self.constants.get("shape", np.array([64000, 52000, 76000]))

    @property
    def end(self) -> np.ndarray:
        """
        Coordinates in X,Y,Z order with units in nano-meters
        default: start + shape
        """
        return self.start + self.shape

    @property
    def resolution(self) -> np.ndarray:
        """
        Resolution in X,Y,Z order with units in nano-meters per voxel on each axis
        default: 4x4x40nm
        """
        return self.constants.get("resolution", np.array([4, 4, 40]))

    @property
    def bounding_box(self) -> BoundingBox:
        """
        Bounding box of the segmented region
        default: BoundingBox(start, shape)
        """
        return BoundingBox(self.start, shape=self.shape)
