from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import queue
import pickle
import logging

from ffskel.Skeletons import Skeleton

import json
import time

from multiprocessing import Process, Manager, Value

Point3 = Tuple[int, int, int]
BoundBox = List[slice]
NodeData = Dict[Point3, Tuple[np.ndarray, BoundBox]]


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


class Volume:
    def __init__(self, constants: Dict = {}, sensitives_file: str = "sensitives.json"):
        self.constants = constants

        self.sensitives = json.load(open(sensitives_file, "r"))

        self.mongo_db = self.sensitives["mongo_db"]
        self.frag_db_host = self.sensitives["frag_db_host"]
        self.frag_db_name = self.sensitives["frag_db_name"]
        self.edges_collection = self.sensitives["edges_collection"]
        self.mount_location = self.sensitives["mount_location"]
        self.rel_fragments_file = self.sensitives["rel_fragments_file"]
        self.fragments_dataset = self.sensitives["fragments_dataset"]

        self._node_segmentations = {}

    def __getitem__(self, key: Point3) -> Optional[Tuple[np.ndarray, List[slice]]]:
        return self._node_segmentations.get(key, None)

    def __setitem__(self, key: Point3, value: Tuple[np.ndarray, List[slice]]):
        if self._node_segmentations.get(key, None) is None:
            self._node_segmentations[key] = value
        else:
            raise ValueError("Updating segmentations is not yet supported")

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
    def view_radius(self) -> int:
        """
        Radius of viewing with units in nano-meters per voxel
        default: 600nm
        """
        return self.constants.get("view_radius", 600)

    @property
    def sampling_dist(self) -> int:
        return self.constants.get(
            "sampling_dist", self.view_radius // 2 * 3
        )  # Default: Approx 50% of voxels will be in two fovs

    @property
    def scale(self) -> np.ndarray:
        return self.constants.get("scale", np.array([10, 10, 1]))

    @property
    def voxel_shape(self) -> np.ndarray:
        """
        The resolution of each voxel in the segmentation space
        This is calculated by multiplying the original resolution by the scale factor
        """
        return self.scale * self.resolution

    @property
    def fov_shape(self) -> np.ndarray:
        """
        Shape of the field of view.
        """
        num_blocks = (2 * self.view_radius) // self.voxel_shape
        num_blocks += (num_blocks + 1) % 2
        return num_blocks * self.voxel_shape

    def get_roi(self, center: np.ndarray) -> Tuple:
        voxel_shape = self.voxel_shape
        fov_shape = self.fov_shape
        center_block = center - center % voxel_shape
        block_offset = (fov_shape // voxel_shape) // 2
        start = center_block - block_offset * voxel_shape
        return start, fov_shape

    @property
    def bounding_box(self) -> BoundingBox:
        """
        Bounding box of the segmented region
        default: BoundingBox(start, shape)
        """
        return BoundingBox(self.start, shape=self.shape)

    @property
    def fragments_file(self):
        return "{}/{}".format(self.mount_location, self.rel_fragments_file)

    @property
    def missing_branches(self):
        dataset = pickle.load(open("missing_branch_dataset.obj", "rb"))
        for skid, data in dataset.items():
            skeleton = Skeleton()
            skeleton.input_nid_pid_x_y_z(data["skeleton_nodes"])
            for branch_chop in data["removed_branch_nodes"]:
                yield (
                    skeleton.delete_branch(branch_chop),
                    "branch_chop",
                    branch_chop,
                    skid,
                )
            for segment_chop in data["removed_segment_nodes"]:
                yield (
                    skeleton.delete_segment(segment_chop),
                    "segment_chop",
                    segment_chop,
                    skid,
                )

    @staticmethod
    def _build_edge_dict(
        edges: Set[Tuple[int, int]], nids: List[int]
    ) -> Dict[int, Optional[int]]:

        # reroot and rebuild skeleton to contain maximum number of nodes under
        # a single root
        edge_dicts = [{}]
        edge_dict = edge_dicts[0]
        unseen_edges = edges
        a, b = edges.pop()
        edge_dict[b] = None
        edge_dict[a] = b
        while len(unseen_edges) > 0:
            temp = set()
            for a, b in unseen_edges:
                if b in edge_dict and a not in edge_dict:
                    edge_dict[a] = b
                elif a in edge_dict and b not in edge_dict:
                    edge_dict[b] = a
                elif a in edge_dict and b in edge_dict:
                    pass
                else:
                    temp.add((a, b))
            if len(temp) == len(unseen_edges):
                edge_dicts.append({})
                edge_dict = edge_dicts[-1]
                a, b = temp.pop()
                edge_dict[b] = None
                edge_dict[a] = b
            unseen_edges = temp

        if len(edge_dicts) > 1:
            sizes = [len(x) for x in edge_dicts]
            logging.info(
                "Multiple disconnected components of length: {} found! Using largest one!".format(
                    sizes
                )
            )
            return edge_dicts[sizes.index(max(sizes))]

        return edge_dict
