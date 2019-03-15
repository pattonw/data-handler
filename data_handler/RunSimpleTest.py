import numpy as np
from pathlib import Path

import logging

import pickle

from JanSegmentationSource import JanSegmentationSource

from ffskel.Skeletons.octrees import OctreeVolume
from ffskel.Skeletons.skeletons import Skeleton

logging.getLogger().setLevel(logging.INFO)

"""
This script runs a single skeleton through JanSegmentation
"""


def rank_from_map(sub_nid_com_map, key):
    ranking = sorted(
        [tuple([k, v[0], v[1]]) for k, v in sub_nid_com_map.items()], key=lambda x: x[2]
    )
    return [x[0] for x in ranking].index(key), len(ranking)


done_skele_file = Path("done_skeles_dict.obj")
jans_segmentations = JanSegmentationSource()

save_file = Path("testing_data.obj")
save_data = []
if not save_file.exists():
    pickle.dump(save_data, save_file.open("wb"))

save_data = pickle.load(save_file.open("rb"))

if len(save_data) == 0:
    for (
        skid,
        whole_skeleton,
        chopped_skeleton,
        chop_type,
        chop,
        new_skeleton,
    ) in jans_segmentations.missing_branches:

        save_data = [
            [
                skid,
                [
                    (
                        node.key,
                        node.parent_key,
                        node.value.center[0],
                        node.value.center[1],
                        node.value.center[2],
                        node.strahler,
                    )
                    for node in chopped_skeleton.get_nodes()
                ],
                chop_type,
                chop,
                new_skeleton,
                None,
            ]
        ]
        pickle.dump(save_data, save_file.open("wb"))
        break

data = save_data[0]
skid, chopped_nodes, chop_type, chop, new_skeleton, segmentations = data
chopped_skeleton = Skeleton()
chopped_skeleton.input_nid_pid_x_y_z_strahler(chopped_nodes)

print("Testing skeleton {} with {} - {}".format(skid, chop_type, chop))


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

unsampled_tree = chopped_skeleton
num_filtered = unsampled_tree.filter_nodes_by_bounds(
    (
        jans_segmentations.start + jans_segmentations.fov_shape // 2,
        jans_segmentations.end - jans_segmentations.fov_shape // 2 - 1,
    )
)
print("{} nodes filtered out of skeleton {}!".format(num_filtered, chopped_skeleton))

sampled_tree = unsampled_tree.resample_segments(
    jans_segmentations.sampling_dist, 1000, 0.01
)
if save_data[0][-1] is not None:
    jans_segmentations._node_segmentations = save_data[0][-1]
jans_segmentations.segment_skeleton(sampled_tree)
if save_data[0][-1] is None:
    save_data[0][-1] = jans_segmentations._node_segmentations
    pickle.dump(save_data, save_file.open("wb"))

for node in sampled_tree.get_nodes():
    try:
        data, bounds = jans_segmentations[tuple(node.value.center)]
        sampled_tree_mask[bounds] += (data > 127).astype(np.uint8)
    except KeyError:
        logging.info("No data for node {}!".format(node.key))
    except TypeError:
        logging.info("Node {} data was None".format(node.value.center))

sampled_tree.fov_shape = jans_segmentations.fov_shape
sampled_tree.resolution = jans_segmentations.fov_shape
sampled_tree.seg.create_octrees_from_nodes(sampled_tree.get_nodes())
sampled_tree_ranking = sampled_tree.get_nid_branch_score_map(key="location")

print(sampled_tree_ranking)

