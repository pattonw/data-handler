import sys
from data_handler.JanSegmentationSource import JanSegmentationSource
from sarbor import Skeleton
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def seeds_from_skeleton(filename):
    import csv

    coords = []
    ids = []
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            coords.append([int(float(x)) for x in row[2:][::-1]])
            if row[1].strip() == "null" or row[1].strip() == "none":
                ids.append([int(float(row[0])), None])
            elif row[0] == row[1]:
                ids.append([int(float(row[0])), None])
            else:
                ids.append([int(float(x)) for x in row[:2]])
    return [ids[i] + coords[i] for i in range(len(ids))]


_, skeleton_file, output_file_base, job_config = sys.argv

skel = Skeleton()
constants = {
    "original_resolution": np.array([4, 4, 40]),
    "start_phys": np.array([0, 0, 0]),
    "shape_phys": np.array([253952 * 4, 155648 * 4, 7063 * 40]),
    "downsample_scale": np.array([10, 10, 1]),
    "leaf_voxel_shape": np.array([128, 128, 128]),
}
skel.seg._constants = constants

nodes = seeds_from_skeleton(skeleton_file)
skel.input_nid_pid_x_y_z(nodes)

jans_segmentations = JanSegmentationSource()

jans_segmentations.constants["fov_shape_voxels"] = np.array([45, 45, 45])
skel.seg._constants["fov_shape_voxels"] = np.array([45, 45, 45])


jans_segmentations.segment_skeleton(skel, num_processes=32)
for node in skel.get_nodes():
    try:
        data, bounds = jans_segmentations[tuple(node.value.center)]
        skel.fill(node.key, (data > 127).astype(np.uint8))
        logging.info("Node {} had data with max value {}!".format(node.key, data.max()))
    except KeyError:
        logging.info("No data for node {}!".format(node.key))
    except TypeError:
        logging.info("Node {} data was None".format(node.value.center))

skel.save_data_for_CATMAID(output_file_base)
