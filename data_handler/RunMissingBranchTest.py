import numpy as np
from pathlib import Path

import logging

import pickle

from .JanSegmentationSource import JanSegmentationSource

"""
Runs the missing branch dataset through jans segmentation
"""


def rank_from_map(sub_nid_com_map, key):
    ranking = sorted(
        [tuple([k, v[0], v[1]]) for k, v in sub_nid_com_map.items()], key=lambda x: x[2]
    )
    return [x[0] for x in ranking].index(key), len(ranking)


def rank_from_location_map(sub_nid_com_map, branch_location):
    closest = np.array([0, 0, 0])
    for coord in sub_nid_com_map:
        if np.linalg.norm(branch_location - coord) < np.linalg.norm(
            branch_location - closest
        ):
            closest = coord
    return rank_from_map(sub_nid_com_map, closest)


def validate_chop(whole, chopped, chop):
    remaining_nodes = [node.key for node in chopped.get_nodes()]
    assert chop[0] in remaining_nodes and not chop[1] in remaining_nodes

def run():

    logging.basicConfig(level=logging.INFO, filename="results/missing_branches.log")

    done_skele_file = Path("results/done_skeles_dict.obj")
    jans_segmentations = JanSegmentationSource()

    missing_branch_data = []
    for (
        skid,
        whole_skeleton,
        chopped_skeleton,
        chop_type,
        chop,
        new_skeleton,
    ) in jans_segmentations.missing_branches:
        if new_skeleton:
            missing_branch_data = []
            jans_segmentations._node_segmentations = {}
        if not done_skele_file.is_file():
            with done_skele_file.open("wb") as f:
                pickle.dump({}, f)

        with done_skele_file.open("rb") as f:
            done_skeles = pickle.load(f)
            if done_skeles.get((skid, chop_type, chop[0], chop[1]), False):
                logging.info(
                    "skeleton {} and chop {} has already been segmented!".format(skid, chop)
                )
                continue
            logging.info("segmenting skeleton {} with chop {}!".format(skid, chop))

        unsampled_tree = chopped_skeleton
        try:
            num_filtered = unsampled_tree.filter_nodes_by_bounds(
                (
                    jans_segmentations.start + jans_segmentations.fov_shape // 2,
                    jans_segmentations.end - jans_segmentations.fov_shape // 2 - 1,
                )
            )
        except ValueError as e:
            logging.warn("All nodes filtered out!")
            logging.debug(e)
            continue
        logging.info("{} nodes filtered out of skeleton {}!".format(num_filtered, skid))
        try:
            validate_chop(whole_skeleton, chopped_skeleton, chop)
            filtered_out = False
        except AssertionError:
            logging.warn("branch location filtered out of Tree")
            filtered_out = True

        sampled_tree = unsampled_tree.resample_segments(
            900, 1000, 0.01
        )
        jans_segmentations.segment_skeleton(sampled_tree, 64)

        logging.debug("Segmentation done!")

        for node in sampled_tree.get_nodes():
            try:
                data, bounds = jans_segmentations[tuple(node.value.center)]
                node.value.mask = (data >= 177).astype(np.uint8)
            except KeyError:
                logging.debug("No data for node {}!".format(node.key))
            except TypeError:
                logging.debug("Node {} data was None".format(node.value.center))

        logging.debug("Segmentation stored in nodes")

        sampled_tree.fov_shape = jans_segmentations.fov_shape
        sampled_tree.resolution = jans_segmentations.fov_shape
        sampled_tree.seg.create_octrees_from_nodes(sampled_tree.get_nodes())
        logging.debug("Octrees created")
        location_score_map = sampled_tree.get_nid_branch_score_map(key="location")
        nid_score_map = {
            node.key: location_score_map[tuple(node.value.center)]
            for node in sampled_tree.get_nodes()
        }
        smoothed_scores = sampled_tree._smooth_scores(nid_score_map)
        smoothed_location_score_map = {
            tuple(node.value.center): smoothed_scores[node.key]
            for node in sampled_tree.get_nodes()
        }
        logging.debug("Scoring done")
        rank_unsmoothed = rank_from_location_map(
            location_score_map, whole_skeleton.nodes[chop[0]].value.center
        )
        rank_smoothed = rank_from_location_map(
            smoothed_location_score_map, whole_skeleton.nodes[chop[0]].value.center
        )
        logging.warn(
            "Missing branch node {} on skeleton {} found with smoothed rank {} vs unsmoothed {}".format(
                chop[1], skid, rank_smoothed, rank_unsmoothed
            )
        )

        missing_branch_data.append(
            (
                skid,
                chop_type,
                chop,
                [whole_skeleton.nodes[nid].value.center for nid in chop],
                (rank_smoothed, rank_unsmoothed),
                location_score_map,
                smoothed_location_score_map,
                whole_skeleton.extract_data(),
                filtered_out,
            )
        )

        pickle.dump(
            missing_branch_data,
            open("results/missing_branch_data/{}.obj".format(skid), "wb"),
        )

        done_skeles = pickle.load(done_skele_file.open("rb"))
        done_skeles[(skid, chop_type, chop[0], chop[1])] = True
        pickle.dump(done_skeles, done_skele_file.open("wb"))

