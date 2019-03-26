import sys
from data_handler.JanSegmentationSource import JanSegmentationSource
from sarbor import Skeleton
import numpy as np
import logging
import json
import pickle
from pathlib import Path
from sarbor import OctreeVolume


def query_jans_segmentation(config, output_file_base):
    logging.basicConfig(level=logging.INFO)

    skel = Skeleton()
    constants = {
        "original_resolution": np.array([4, 4, 40]),
        "start_phys": np.array([0, 0, 0]),
        "shape_phys": np.array([253952 * 4, 155648 * 4, 7063 * 40]),
        "downsample_scale": np.array([10, 10, 1]),
        "leaf_voxel_shape": np.array([128, 128, 128]),
        "fov_voxel_shape": np.array([45, 45, 45]),
    }
    skel.seg._constants = constants

    nodes = config.skeleton.nodes
    skel.input_nid_pid_x_y_z(nodes)

    if config.skeleton.strahler_filter and False:
        skel.filter_nodes_by_strahler(
            config.skeleton.min_strahler, config.skeleton.max_strahler
        )
    if config.skeleton.resample:
        processed_skel = skel.resample_segments(
            config.skeleton.resample_delta, 1000, 0.000001
        )
    else:
        processed_skel = skel
    processed_skel.seg._constants = skel.seg._constants

    jans_segmentations = JanSegmentationSource()

    jans_segmentations.constants["fov_shape_voxels"] = np.array([45, 45, 45])

    jans_segmentations.segment_skeleton(processed_skel, num_processes=32)
    for node in processed_skel.get_nodes():
        try:
            data, bounds = jans_segmentations[tuple(node.value.center)]
            processed_skel.fill(node.key, (data > 127).astype(np.uint8))
            logging.info(
                "Node {} had data with max value {}!".format(node.key, data.max())
            )
        except KeyError:
            logging.info("No data for node {}!".format(node.key))
        except TypeError:
            logging.info("Node {} data was None".format(node.value.center))

    processed_skel.save_data_for_CATMAID(output_file_base)


"""
RunSimpleTest
"""


def run_simple_test():
    def rank_from_map(sub_nid_com_map, key):
        ranking = sorted(
            [tuple([k, v[0], v[1]]) for k, v in sub_nid_com_map.items()],
            key=lambda x: x[2],
        )
        return [x[0] for x in ranking].index(key), len(ranking)

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
    print(
        "{} nodes filtered out of skeleton {}!".format(num_filtered, chopped_skeleton)
    )

    sampled_tree = unsampled_tree.resample_segments(900, 1000, 0.01)
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


"""
RunFalseMergeTest
"""


def run_false_merge_test():
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

        sampled_tree = unsampled_tree.resample_segments(900, 1000, 0.01)

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


"""
RunMissingBranchTest
"""


def run_missing_branch_test():
    def rank_from_map(sub_nid_com_map, key):
        ranking = sorted(
            [tuple([k, v[0], v[1]]) for k, v in sub_nid_com_map.items()],
            key=lambda x: x[2],
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
                    "skeleton {} and chop {} has already been segmented!".format(
                        skid, chop
                    )
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

        sampled_tree = unsampled_tree.resample_segments(900, 1000, 0.01)
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
