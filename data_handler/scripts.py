from data_handler.JanSegmentationSource import JanSegmentationSource
from sarbor import Skeleton
from sarbor.arbors import Node
import numpy as np
import logging
import pickle
from pathlib import Path
from sarbor import OctreeVolume
import random


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
        "{} nodes filtered out of skeleton {}!\n{} nodes remaining!".format(
            num_filtered, skid, len(list(unsampled_tree.get_nodes()))
        )
    )

    sampled_tree = unsampled_tree.resample_segments(900, 1000, 0.0001)
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
            [tuple([k, v[1]]) for k, v in connectivity_map.items()], key=lambda x: x[1]
        )
        return [x[0] for x in ranking].index(closest_pair), len(ranking)

    logging.basicConfig(level=logging.INFO, filename="test_results/false_merges.log")

    done_skele_file = Path("test_results/done_false_merge_dict.obj")
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

        skeleton.calculate_strahlers()

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

        # Sample node points from skeleton
        try:
            sampled_tree = unsampled_tree.resample_segments(900, 1000, 0.01)
        except ValueError as e:
            continue

        # set jans_segmentation fov_shape_voxels
        constants = {
            "original_resolution": np.array([4, 4, 40]),
            "start_phys": np.array([0, 0, 0]),
            "shape_phys": np.array([253952 * 4, 155648 * 4, 7063 * 40]),
            "downsample_scale": np.array([10, 10, 1]),
            "leaf_voxel_shape": np.array([128, 128, 128]),
            "fov_voxel_shape": np.array([45, 45, 45]),
        }
        jans_segmentations.constants["fov_shape_voxels"] = np.array([45, 45, 45])
        sampled_tree.seg._constants = constants

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

        false_merge_data.append((skid, log, skeleton.extract_data(), score_map, rank))

        pickle.dump(
            false_merge_data,
            open("test_results/false_merge_data/{}.obj".format(skid), "wb"),
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

    logging.basicConfig(
        level=logging.INFO, filename="test_results/missing_branches.log"
    )

    # create a file to keep track of which skeletons have been segmented
    done_skele_file = Path("test_results/done_skeles_dict.obj")
    if not done_skele_file.is_file():
        with done_skele_file.open("wb") as f:
            pickle.dump({}, f)

    jans_segmentations = JanSegmentationSource()

    # Data is stored per skeleton we analyze, this way the file
    # doesn't get so large it becomes a bottle neck
    missing_branch_data = []
    for (
        skid,
        whole_skeleton,
        chopped_skeleton,
        chop_type,
        chop,
        new_skeleton,
    ) in jans_segmentations.missing_branches:
        # If this is a new skeleton, remove any old cache
        if new_skeleton:
            missing_branch_data = []
            jans_segmentations._node_segmentations = {}

        # Check if this skeleton has been done before
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

        # Downsample tree to only contain nodes in the Calyx
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

        # Check if the branch node was filtered out of the skeleton,
        # i.e. it was not in the calyx
        try:
            validate_chop(whole_skeleton, chopped_skeleton, chop)
        except AssertionError:
            # go to next skeleton
            continue

        # resample tree
        sampled_tree = unsampled_tree.resample_segments(900, 1000, 0.01)

        # set jans_segmentation fov_shape_voxels
        constants = {
            "original_resolution": np.array([4, 4, 40]),
            "start_phys": np.array([0, 0, 0]),
            "shape_phys": np.array([253952 * 4, 155648 * 4, 7063 * 40]),
            "downsample_scale": np.array([10, 10, 1]),
            "leaf_voxel_shape": np.array([128, 128, 128]),
            "fov_voxel_shape": np.array([45, 45, 45]),
        }
        jans_segmentations.constants["fov_shape_voxels"] = np.array([45, 45, 45])
        sampled_tree.seg._constants = constants

        # segment the tree
        jans_segmentations.segment_skeleton(sampled_tree, 64)
        logging.debug("Segmentation done!")

        # retrieve segmentations associated with each seed point and insert them into the sarbor
        for node in sampled_tree.get_nodes():
            try:
                data, bounds = jans_segmentations[tuple(node.value.center)]
                node.value.mask = (data >= 177).astype(np.uint8)
            except KeyError:
                logging.debug("No data for node {}!".format(node.key))
            except TypeError:
                logging.debug("Node {} data was None".format(node.value.center))
        logging.debug("Segmentation stored in nodes")

        # create octrees from stored segmentation
        sampled_tree.seg.create_octrees_from_nodes(
            sampled_tree.get_nodes(), interpolate_dist_nodes=3
        )
        logging.debug("Octrees created")

        # get branch node scores by location
        location_score_map = sampled_tree.get_nid_branch_score_map(key="location")

        # get branch node scores by nid
        nid_score_map = {
            node.key: location_score_map[tuple(node.value.center)]
            for node in sampled_tree.get_nodes()
        }

        # smooth branch node scores
        smoothed_scores = sampled_tree._smooth_scores(nid_score_map)
        smoothed_location_score_map = {
            tuple(node.value.center): smoothed_scores[node.key]
            for node in sampled_tree.get_nodes()
        }
        logging.debug("Scoring done")

        # get unsmoothed rank
        rank_unsmoothed = rank_from_location_map(
            location_score_map, whole_skeleton.nodes[chop[0]].value.center
        )
        # get smoothed rank
        rank_smoothed = rank_from_location_map(
            smoothed_location_score_map, whole_skeleton.nodes[chop[0]].value.center
        )
        logging.info(
            "Missing branch node {} on skeleton {} found with smoothed rank {} vs unsmoothed {}".format(
                chop[1], skid, rank_smoothed, rank_unsmoothed
            )
        )

        # save data
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
            )
        )

        pickle.dump(
            missing_branch_data,
            open("test_results/missing_branch_data/{}.obj".format(skid), "wb"),
        )

        done_skeles = pickle.load(done_skele_file.open("rb"))
        done_skeles[(skid, chop_type, chop[0], chop[1])] = True
        pickle.dump(done_skeles, done_skele_file.open("wb"))


"""
Connectivity_Stats
"""


def run_connectivity_stats():
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

    logging.basicConfig(
        level=logging.INFO, filename="test_results/connectivity_stats.log"
    )

    # create a file to keep track of which skeletons have been segmented
    done_skele_file = Path("test_results/done_stats_dict.obj")
    if not done_skele_file.is_file():
        with done_skele_file.open("wb") as f:
            pickle.dump({}, f)

    jans_segmentations = JanSegmentationSource()

    # Data is stored per skeleton we analyze, this way the file
    # doesn't get so large it becomes a bottle neck
    missing_branch_data = []
    for (
        skid,
        whole_skeleton,
        chopped_skeleton,
        chop_type,
        chop,
        new_skeleton,
    ) in jans_segmentations.missing_branches:
        # If this is a new skeleton, remove any old cache
        if new_skeleton:
            missing_branch_data = []
            jans_segmentations._node_segmentations = {}
        else:
            continue

        # Check if this skeleton has been done before
        with done_skele_file.open("rb") as f:
            done_skeles = pickle.load(f)
            if done_skeles.get(skid, False):
                logging.info("skeleton {} has already been segmented!".format(skid))
                continue
            logging.info("segmenting skeleton {}!".format(skid))

        # Downsample tree to only contain nodes in the Calyx
        unsampled_tree = whole_skeleton
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

        # resample tree
        sampled_tree = unsampled_tree.resample_segments(500, 1000, 0.01)

        n = len(sampled_tree.nodes)
        for i in range(n, n + 10):
            s = random.choice(range(n))
            start = sampled_tree.nodes[s]
            parent = sampled_tree.nodes[s].parent
            if parent is not None:
                vector = parent.value.center - start.value.center
                center = (parent.value.center + start.value.center) // 2
                perp_vec = np.cross(vector, [1, 0, 0])
                perp_vec = perp_vec / np.linalg.norm(perp_vec) * 500
                new_node = Node(key=s, center=center + perp_vec)
                start.add_child(new_node)

        # set jans_segmentation fov_shape_voxels
        constants = {
            "original_resolution": np.array([4, 4, 40]),
            "start_phys": np.array([0, 0, 0]),
            "shape_phys": np.array([253952 * 4, 155648 * 4, 7063 * 40]),
            "downsample_scale": np.array([10, 10, 1]),
            "leaf_voxel_shape": np.array([128, 128, 128]),
            "fov_voxel_shape": np.array([45, 45, 45]),
        }
        jans_segmentations.constants["fov_shape_voxels"] = np.array([45, 45, 45])
        sampled_tree.seg._constants = constants

        # segment the tree
        jans_segmentations.segment_skeleton(sampled_tree, 64)
        logging.debug("Segmentation done!")

        # retrieve segmentations associated with each seed point and insert them into the sarbor
        for node in sampled_tree.get_nodes():
            try:
                data, bounds = jans_segmentations[tuple(node.value.center)]
                node.value.mask = (data >= 177).astype(np.uint8)
            except KeyError:
                logging.debug("No data for node {}!".format(node.key))
            except TypeError:
                logging.debug("Node {} data was None".format(node.value.center))
        logging.debug("Segmentation stored in nodes")

        # create octrees from stored segmentation
        sampled_tree.seg.create_octrees_from_nodes(
            sampled_tree.get_nodes(), interpolate_dist_nodes=3
        )
        logging.debug("Octrees created")

        # get connectivity scores
        connectivity_scores = sampled_tree.get_node_connectivity()
        connected_node_scores = [
            v[1] for k, v in connectivity_scores.items() if k[0] in range(n)
        ]
        disconnected_node_scores = [
            v[1] for k, v in connectivity_scores.items() if k[0] >= n
        ]
        logging.info(
            "connected_node_scores: #{} -> {}\nrandom_node_scores: #{} -> {}".format(
                len(connected_node_scores),
                sum(connected_node_scores) / (len(connected_node_scores) + 1),
                len(disconnected_node_scores),
                sum(disconnected_node_scores) / (len(disconnected_node_scores) + 1),
            )
        )

        # save data
        missing_branch_data.append(
            (skid, whole_skeleton.extract_data(), connectivity_scores), n
        )

        pickle.dump(
            missing_branch_data,
            open("test_results/missing_branch_data/{}.obj".format(skid), "wb"),
        )

        done_skeles = pickle.load(done_skele_file.open("rb"))
        done_skeles[skid] = True
        pickle.dump(done_skeles, done_skele_file.open("wb"))
