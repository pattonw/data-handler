from typing import List, Dict, Tuple, Optional
import numpy as np
import queue
from pathlib import Path

import logging
import time

from multiprocessing import Process, Manager, Value
import pickle

from JanMongo import JanSegmentationSource

from ffskel.Skeletons.octrees import OctreeVolume
from sparse_contour import contour_sparse_octree_volume


def rank_from_map(sub_nid_com_map, key):
    ranking = sorted(
        [tuple([k, v[0], v[1]]) for k, v in sub_nid_com_map.items()], key=lambda x: x[2]
    )
    return [x[0] for x in ranking].index(key), len(ranking)


logging.basicConfig(level=logging.INFO, filename="jan_mongo_segmentation_results.log")

done_skele_file = Path("data", "skeleton_data", "Jan_Seg", "done_skeles_dict.obj")
jans_segmentations = JanSegmentationSource()
print(
    "block_shape: {}".format(
        jans_segmentations.fov_shape / jans_segmentations.voxel_shape
    )
)


for skeleton in jans_segmentations.skeletons:
    if skeleton == 16:
        continue
    if not done_skele_file.is_file():
        with open(done_skele_file, "wb") as f:
            pickle.dump({}, f)

    with open(done_skele_file, "rb") as f:
        done_skeles = pickle.load(f)
        if done_skeles.get(skeleton, False):
            logging.info("skeleton {} has already been segmented!".format(skeleton))
            continue
        logging.info("segmenting skeleton {}!".format(skeleton))

    def _data_populator(bounds):
        return np.zeros(np.array(bounds[1]) - np.array(bounds[0]))

    sampled_tree_mask = OctreeVolume(
        [50, 50, 50],
        (
            (jans_segmentations.start - jans_segmentations.fov_shape)
            // jans_segmentations.voxel_shape,
            (
                jans_segmentations.end
                + jans_segmentations.fov_shape
                + jans_segmentations.voxel_shape
                - 1
            )
            // jans_segmentations.voxel_shape,
        ),
        np.uint8,
        _data_populator,
    )

    skeleton_data = []

    unsampled_tree = jans_segmentations.extract_skeleton(skeleton)
    unsampled_tree.calculate_strahlers()
    num_filtered = unsampled_tree.filter_nodes_by_bounds(
        (
            jans_segmentations.start + jans_segmentations.fov_shape // 2,
            jans_segmentations.end - jans_segmentations.fov_shape // 2 - 1,
        )
    )
    print("{} nodes filtered out of skeleton {}!".format(num_filtered, skeleton))

    sampled_tree = unsampled_tree.resample_segments(
        jans_segmentations.sampling_dist, 1000, 0.01
    )
    jans_segmentations.segment_skeleton(sampled_tree)

    for node in sampled_tree.get_nodes():
        try:
            data, bounds = jans_segmentations[tuple(node.value.center)]
            sampled_tree_mask[bounds] += (data > 127).astype(np.uint8)
        except KeyError:
            logging.info("No data for node {}!".format(node.key))
        except TypeError:
            logging.info("Node {} data was None".format(node.value.center))

    sampled_tree.transform(0, 1 / jans_segmentations.voxel_shape)
    sampled_tree.fov_shape = (
        jans_segmentations.fov_shape // jans_segmentations.voxel_shape
    )
    sampled_tree.resolution = jans_segmentations.fov_shape
    sampled_tree_ranking = sampled_tree.get_nid_branch_score_map(key="location")

    print("Start calculating missing branch scores!")

    for segment in unsampled_tree.get_segments():
        print(
            "Splitting tree around segment from node {} to node {}".format(
                segment[0].key, segment[-1].key
            )
        )
        trees, branch_points = unsampled_tree.split(segment)
        for sub_tree, branch_nodes in zip(trees, branch_points):
            print("starting subtree with node {} cut out!".format(branch_nodes[1].key))

            if len(sub_tree.arbor.root.children) == 0:
                print("skipped this subtree")
                continue

            smooth_skel = sub_tree.resample_segments(
                jans_segmentations.sampling_dist, 1000, 0.01
            )
            new_branch_point = smooth_skel.arbor.root
            for potential_branch_node in smooth_skel.get_nodes():
                if (
                    (
                        abs(
                            potential_branch_node.value.center
                            - branch_nodes[0].value.center
                        )
                        ** 2
                    ).sum()
                    < (
                        abs(
                            new_branch_point.value.center - branch_nodes[0].value.center
                        )
                        ** 2
                    ).sum()
                ):
                    new_branch_point = potential_branch_node
            branch_point_offset = (
                abs(
                    (new_branch_point.value.center - branch_nodes[0].value.center) ** 2
                ).sum()
                ** 0.5
            )
            print(
                "dist from approx branch point to real: {}".format(branch_point_offset)
            )

            num_segmented = jans_segmentations.segment_skeleton(smooth_skel)
            logging.info("{} nodes segmented".format(num_segmented))
            error = False
            for node in smooth_skel.get_nodes():
                try:
                    data, bounds = jans_segmentations[tuple(node.value.center)]
                    node.value.mask = (data > 127).astype(np.uint8)
                except KeyError:
                    logging.info("No data for node {}!".format(node.key))
                    error = True
                except TypeError:
                    logging.info("Node {} data was None".format(node.value.center))
                    error = True

            smooth_skel.transform(0, 1 / jans_segmentations.voxel_shape)
            smooth_skel.fov_shape = (
                jans_segmentations.fov_shape // jans_segmentations.voxel_shape
            )
            smooth_skel.resolution = jans_segmentations.fov_shape

            branch_scores = smooth_skel.get_sub_nid_branch_scores(
                sampled_tree_mask, sampled_tree_ranking
            )
            rank, n = rank_from_map(branch_scores, new_branch_point.key)

            print("Sub_tree done. Branch found with score: {}".format((rank + 1) / n))

            skeleton_data.append(
                (
                    branch_nodes[0].key,
                    branch_nodes[0].strahler
                    if branch_nodes[0].strahler is not None
                    else 0,
                    rank,
                    0 if branch_nodes[0].value.mask is None else 1,
                    sum(
                        [
                            0 if node.value.mask is None else 1
                            for node in branch_nodes[0].get_neighbors()
                            + [branch_nodes[0]]
                        ]
                    )
                    / (len(branch_nodes[0].get_neighbors()) + 1),
                    n,
                    branch_nodes[1].strahler
                    if branch_nodes[1].strahler is not None
                    else 0,
                    skeleton,
                    error,
                    segment[0].key,
                    segment[1].key,
                    branch_point_offset,
                )
            )

    pickle.dump(
        skeleton_data, open("data/skeleton_data/Jan_Seg/{}.obj".format(skeleton), "wb")
    )

    done_skeles = pickle.load(open(done_skele_file, "rb"))
    done_skeles[skeleton] = True
    pickle.dump(done_skeles, open(done_skele_file, "wb"))

