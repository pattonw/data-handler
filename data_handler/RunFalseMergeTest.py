import numpy as np
from pathlib import Path

import logging

import pickle

from .JanSegmentationSource import JanSegmentationSource

"""
Runs the false merge dataset through Jans segmentation.
"""


def rank_from_map(connectivity_map, location):
    closest_pair = None
    dist = None
    for (nid, pid), ((n_center, p_center), score) in connectivity_map.items():
        # calculate distance between line from n to p and the location we are looking for
        d = np.linalg.norm(
            np.cross(p_center - n_center, n_center - location)
        ) / np.linalg.norm(p_center - n_center)
        if closest_pair is None or d < dist:
            closest_pair = (nid, pid)
            dist = d

    ranking = sorted(
        [tuple([k, v[1]]) for k, v in connectivity_map.items()], key=lambda x: x[2]
    )
    return [x[0] for x in ranking].index(closest_pair), len(ranking)

def run():

    logging.basicConfig(level=logging.INFO, filename="results/false_merges.log")

    done_skele_file = Path("results/done_false_merge_dict.obj")
    jans_segmentations = JanSegmentationSource()

    false_merge_data = []
    for (skid, skeleton, log) in jans_segmentations.false_merges:
        # --------SETUP-----------
        # import skeletons and splits, check if this skeleton/split
        # has already been run, filter nodes that are outside calyx bounds
        if not done_skele_file.is_file():
            with done_skele_file.open("wb") as f:
                pickle.dump({}, f)

        with done_skele_file.open("rb") as f:
            done_skeles = pickle.load(f)
            if done_skeles.get((skid, log), False):
                logging.info("skeleton {} has already been segmented!".format(skid))
                continue
            logging.info("Running split: {}!".format(log))

        unsampled_tree = skeleton
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

        split_location = np.array(log[3:6])

        # --------Segmentation--------
        # Sample node points from skeleton
        # gather segmentations

        sampled_tree = unsampled_tree.resample_segments(
            jans_segmentations.sampling_dist, 1000, 0.01
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

        # ----------Calculate Rankings----------
        # calculate rankings, store necessary data, record work done
        # so that its not repeated

        score_map = sampled_tree.get_node_connectivity()
        logging.debug("Scoring done")
        rank = rank_from_map(score_map, split_location)
        logging.warn(
            "False merge at location {} found at rank {}".format(split_location, rank)
        )

        false_merge_data.append((skid, log, skeleton.extract_data(), score_map))

        pickle.dump(
            false_merge_data, open("results/false_merge_data/{}.obj".format(skid), "wb")
        )

        done_skeles = pickle.load(done_skele_file.open("rb"))
        done_skeles[(skid, log)] = True
        pickle.dump(done_skeles, done_skele_file.open("wb"))

