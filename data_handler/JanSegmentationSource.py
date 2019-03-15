from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import queue
import pickle
import logging
from pathlib import Path


import daisy
import lsd
import funlib.segment


from ffskel.Skeletons import Skeleton

import json
import time

from multiprocessing import Process, Manager, Value

from .calyx import Calyx, BoundingBox

Point3 = Tuple[int, int, int]
BoundBox = List[slice]
NodeData = Dict[Point3, Tuple[np.ndarray, BoundBox]]


class JanSegmentationSource:
    def __init__(
        self,
        constants: Dict = {},
        sensitives_file: str = "sensitives.json",
        volume=None,
    ):
        if volume is None:
            self.volume = Calyx
        else:
            self.volume = volume

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
        return self.volume.start

    @property
    def shape(self) -> np.ndarray:
        return self.volume.shape

    @property
    def end(self) -> np.ndarray:
        return self.volume.end

    @property
    def resolution(self) -> np.ndarray:
        return self.volume.resolution

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

    def _get_roi(self, center: np.ndarray) -> Tuple:
        """
        Get a region of interest from a center coordinate. Note that coordinates
        are provided by the input data space, thus we need to take into account
        the translation.

        This is a private function since using the translation will not provide
        the expected regions when used in other contexts
        """
        voxel_shape = self.voxel_shape
        fov_shape = self.fov_shape
        center_block = center - center % voxel_shape
        block_offset = (fov_shape // voxel_shape) // 2
        start = center_block - block_offset * voxel_shape
        start += self.translation_phys
        return start, fov_shape

    @property
    def bounding_box(self) -> BoundingBox:
        self.volume.bounding_box

    @property
    def fragments_file(self):
        return "{}/{}".format(self.mount_location, self.rel_fragments_file)

    @property
    def missing_branch_file(self):
        return Path(
            self.constants.get(
                "missing_branch_file",
                "/".join(("datasets", "missing_branch_dataset.obj")),
            )
        )

    @property
    def missing_branches(self):
        dataset = pickle.load(self.missing_branch_file.open("rb"))
        for skid, data in dataset.items():
            skeleton = Skeleton()
            new = True
            skeleton.input_nid_pid_x_y_z_strahler(data["skeleton_nodes"])
            for branch_chop in data["removed_branch_nodes"]:
                yield (
                    skid,
                    skeleton,
                    skeleton.delete_branch(branch_chop),
                    "branch_chop",
                    branch_chop,
                    new,
                )
                new = False
            for segment_chop in data["removed_segment_nodes"]:
                yield (
                    skid,
                    skeleton,
                    skeleton.delete_branch(segment_chop),
                    "segment_chop",
                    segment_chop,
                    new,
                )
                new = False

    @property
    def false_merge_file(self):
        return Path(
            self.constants.get(
                "false_merge_file", "/".join(("datasets", "false_merge_dataset.obj"))
            )
        )

    @property
    def false_merges(self):
        dataset = pickle.load(self.false_merge_file.open("rb"))
        for skid, data in dataset.items():
            log = data["split_log"]
            skeleton = Skeleton()
            skeleton.input_nid_pid_x_y_z(data["skeleton_nodes"])
            yield (skid, skeleton, log)

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

    def data_fetcher(
        self,
        worker_id: int,
        node_queue: queue.Queue,
        results_queue: queue.Queue,
        done_workers: Value,
    ):
        while True:
            try:
                node_coords = node_queue.get(False)
                logging.debug("Got node {}!".format(node_coords))
            except queue.Empty:
                logging.debug("Worker {} Done".format(worker_id))
                with done_workers.get_lock():
                    done_workers.value += 1
                break
            try:
                stored = self[(node_coords[0], node_coords[1], node_coords[2])]
                if stored is None:
                    segmentation_data, bounds = self.get_segmentation(node_coords)
                    results_queue.put(
                        (
                            (node_coords[0], node_coords[1], node_coords[2]),
                            segmentation_data,
                            bounds,
                        )
                    )
                    logging.debug("Successfully segmented node {}".format(node_coords))
                else:
                    logging.debug(
                        "Node {} has previously been segmented!".format(node_coords)
                    )
            except ValueError as e:
                # This error should only be caused by the roi being outside
                # segmented volume bounds
                logging.debug("Node failed! {}".format(e))
                pass
            except Exception as e:
                logging.warn("Unknown Error: {}".format(e))
                pass

    def query_local_segmentation(self, roi, threshold):
        # open fragments
        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)

        # open RAG DB
        rag_provider = lsd.persistence.MongoDbRagProvider(
            self.frag_db_name,
            host=self.frag_db_host,
            mode="r",
            edges_collection=self.edges_collection,
        )

        segmentation = fragments[roi]
        segmentation.materialize()
        rag = rag_provider[roi]

        if len(rag.nodes()) == 0:
            return segmentation

        components = rag.get_connected_components(threshold)

        values_map = np.array(
            [
                [fragment, i]
                for i in range(len(components))
                for fragment in components[i]
            ],
            dtype=np.uint64,
        )
        old_values = values_map[:, 0]
        new_values = values_map[:, 1]
        funlib.segment.arrays.replace_values(
            segmentation.data, old_values, new_values, inplace=True
        )

        return segmentation

    def query_center_object(self, roi, threshold):
        segmentation = self.query_local_segmentation(roi, threshold)
        center = np.array(segmentation.shape) // 2
        center_obj = segmentation.data[center[0], center[1], center[2]]
        seg_fragments = set(segmentation.data.reshape(-1))
        values_map = np.array(
            [
                [fragment, 1 if fragment == center_obj else 0]
                for fragment in seg_fragments
            ],
            dtype=np.uint64,
        )
        old_values = values_map[:, 0]
        new_values = values_map[:, 1]
        funlib.segment.arrays.replace_values(
            segmentation.data, old_values, new_values, inplace=True
        )
        return segmentation

    def get_segmentation(self, center: np.ndarray) -> Tuple[np.ndarray, List[slice]]:
        """
        Jan's segmentation and the Saalfeld N5 have a 1 slice translation
        compared to the data in CATMAID. Thus we must scale the start
        down by one slice in the z direction.
        """
        roi_start, roi_shape = self._get_roi(center)
        if self.bounding_box.contains_roi(roi_start, roi_shape):
            # ROI input in physical dimensions i.e. nano meters [Z, Y, X].
            # Output in pixel dimensions (Z/40, Y/4, X/4)
            segmentation = (
                self.query_center_object(
                    daisy.Roi(roi_start[::-1], roi_shape[::-1]), threshold=0.3
                )
                .data.transpose([2, 1, 0])
                .astype(float)
            )
            try:
                data = (
                    (255 * segmentation)
                    .reshape(
                        [
                            segmentation.shape[0] // self.scale[0],
                            self.scale[0],
                            segmentation.shape[1] // self.scale[1],
                            self.scale[1],
                            segmentation.shape[2] // self.scale[2],
                            self.scale[2],
                        ]
                    )
                    .mean(5)
                    .mean(3)
                    .mean(1)
                )
            except Exception as e:
                logging.debug("Scale: {}".format(self.scale))
                logging.debug("Shape: {}".format(segmentation.shape))
                raise e

            downsampled_bounds = list(
                map(
                    slice,
                    center // self.resolution // self.scale
                    - self.fov_shape // self.resolution // self.scale // 2,
                    center // self.resolution // self.scale
                    + self.fov_shape // self.resolution // self.scale // 2
                    + 1,
                )
            )

            return (data, downsampled_bounds)
        else:
            logging.debug(
                "Center: {}; Queried bounds: {}; volume bounds: {}".format(
                    center, (roi_start, roi_shape), self.bounding_box
                )
            )
            raise ValueError("Roi is not contained in segmented volume")

    def segment_skeleton(self, skeleton: Skeleton, num_processes: int = 8) -> int:
        manager = Manager()
        # Queue of seeds to be picked up by workers.
        node_queue = manager.Queue()
        # Queue of results from workers.
        results_queue = manager.Queue()

        done_fetchers = Value("i", 0)

        branch_nodes = list(skeleton.get_interesting_nodes(branches=True))
        all_nodes = list(skeleton.get_nodes())
        num_nodes = len(all_nodes)
        num_branches = len(branch_nodes)
        logging.info("{} nodes with {} branches".format(num_nodes, num_branches))
        for node in all_nodes[:]:
            try:
                node_queue.put(node.value.center)
            except RecursionError:
                logging.debug("Maximum recursion depth hit. Too many nodes!")
                node_queue = manager.Queue()
                continue

        fetchers = []

        logging.debug("Starting Fetchers!")
        for fetcher_id in range(num_processes):
            fetcher = Process(
                target=self.data_fetcher,
                args=(fetcher_id, node_queue, results_queue, done_fetchers),
            )
            fetcher.start()
            fetchers.append(fetcher)

        num_done = 0
        start = time.time()
        while done_fetchers.value < num_processes or not results_queue.empty():
            try:
                node, data, bounds = results_queue.get(True, 5)
                self[node] = (data, bounds)
                num_done += 1
                if num_done % 50 == 0:
                    logging.info(
                        "{} out of {} done! avg: {:.3f} seconds per node".format(
                            num_done, num_nodes, (time.time() - start) / num_done
                        )
                    )
            except TypeError as e:
                logging.debug("Waiting...")
                logging.debug(e)
                num_done += 1
                pass
            except queue.Empty:
                logging.debug(
                    "Empty Queue! {}/{} fetchers done".format(
                        done_fetchers.value, num_processes
                    )
                )
                pass
        logging.info(
            "{} fetchers done! {} nodes skipped!".format(
                done_fetchers.value, num_nodes - num_done
            )
        )

        for wid, fetcher in enumerate(fetchers):
            fetcher.join()
            manager.shutdown()

        return num_done
