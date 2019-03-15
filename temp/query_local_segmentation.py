import daisy
import logging
import lsd
import time
import numpy as np
import funlib.segment

import matplotlib.pyplot as plt
import json

logging.basicConfig(level=logging.INFO)

sensitives = json.load(open("sensitives.json", "r"))

db_host = sensitives["frag_db_host"]
db_name = sensitives["frag_db_name"]
edges_collection = sensitives["edges_collection"]
mount_location = sensitives["mount_location"]
rel_fragments_file = sensitives["rel_fragments_file"]
fragments_dataset = sensitives["fragments_dataset"]


fragments_file = "{}/{}".format(mount_location, rel_fragments_file)


def query_local_segmentation(roi, threshold):

    # open fragments
    fragments = daisy.open_ds(fragments_file, fragments_dataset)

    # open RAG DB
    rag_provider = lsd.persistence.MongoDbRagProvider(
        db_name, host=db_host, mode="r", edges_collection=edges_collection
    )

    print("Reading fragments and RAG in %s" % roi)
    start = time.time()
    segmentation = fragments[roi]
    segmentation.materialize()
    rag = rag_provider[roi]
    print("%.3fs" % (time.time() - start))

    print("Number of nodes in RAG: %d" % (len(rag.nodes())))
    print("Number of edges in RAG: %d" % (len(rag.edges())))

    if len(rag.nodes()) == 0:
        print("No nodes found")
        return segmentation

    print("Merging...")
    start = time.time()
    components = rag.get_connected_components(threshold)
    print("%.3fs" % (time.time() - start))

    print("Relabelling fragments...")
    start = time.time()
    values_map = np.array(
        [
            [fragment, component[0]]
            for component in components
            for fragment in component
        ],
        dtype=np.uint64,
    )
    old_values = values_map[:, 0]
    new_values = values_map[:, 1]
    print(dir(lsd))
    funlib.segment.arrays.replace_values(
        segmentation.data, old_values, new_values, inplace=True
    )
    print("%.3fs" % (time.time() - start))

    return segmentation


if __name__ == "__main__":

    segmentation = query_local_segmentation(
        daisy.Roi((158000, 121800, 403560), (800, 800, 800)), threshold=0.3
    )

    print(segmentation.roi)
    print(segmentation.data)

    ids = {}
    current = 0
    possible_ids = np.linspace(0.05, 0.95, 100)

    for im_slice in segmentation.data:
        im_ids = set([y for x in im_slice for y in x])
        for im_id in im_ids:
            if im_id not in ids:
                ids[im_id] = possible_ids[current]
                current += 1
        im_copy = []
        for row in im_slice:
            new_row = []
            for x in row:
                new_row.append(ids[x])
            im_copy.append(new_row)

        plt.imshow(im_copy, cmap="Paired", vmin=0, vmax=1)
        plt.show()


["agglomerate_in_block", "watershed_in_block"]
["connected_components", "labels", "replace_values"]

