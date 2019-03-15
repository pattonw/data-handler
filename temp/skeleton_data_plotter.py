from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List
import pickle

from skeleton_fetcher import SkeletonFetcher


def strahler_info(nid_pid_x_y_z_s, branch_point, chop_point):
    nid_pid_x_y_z_s = sorted(nid_pid_x_y_z_s, key=lambda x: x[5], reverse=True)
    branch_points = filter(
        lambda x: x[0] in (branch_point, chop_point), nid_pid_x_y_z_s
    )
    return (nid_pid_x_y_z_s[0][5], {x[0]: x[5] for x in branch_points})


def load_data():
    data = []
    for skel in Path("results/missing_branch_data").iterdir():
        skeleton = pickle.load(skel.open("rb"))
        for chop in skeleton:
            strahler_data = strahler_info(chop[7][0], chop[2][0], chop[2][1])
            data.append(
                (
                    chop[0],
                    chop[2][0],
                    chop[2][1],
                    chop[4][0],
                    chop[4][1],
                    strahler_data[0],
                    strahler_data[1].get(chop[2][0], None),
                    strahler_data[1].get(chop[2][1], None),
                )
            )
    return data


data = pd.DataFrame(
    load_data(),
    columns=[
        "skid",
        "keep_nid",
        "drop_nid",
        "smoothed_rank",
        "unsmoothed_rank",
        "max_strahler",
        "keep_strahler",
        "drop_strahler",
    ],
)
data["num_nodes"] = data["smoothed_rank"].map(lambda x: x[1])
data["smoothed_rank"] = data["smoothed_rank"].map(lambda x: x[0])
data["unsmoothed_rank"] = data["unsmoothed_rank"].map(lambda x: x[0])
data["smooth_score"] = (data["smoothed_rank"] + 1) / data["num_nodes"]
data["unsmooth_score"] = (data["unsmoothed_rank"] + 1) / data["num_nodes"]

data.hist(column="smooth_score")
plt.show()
data.hist(column="unsmooth_score")
plt.show()

print(data.groupby(by=["skid"]).agg({"unsmooth_score": ["mean", "std"]}))

print(data.head())


class SkeletonDataPlotter:
    def __init__(self):
        self.skeleton_fetcher = SkeletonFetcher()

    def histogram(self, data: List[int]):
        data = np.array(data)
        print(len(data))
        print(np.percentile(data, [0, 5, 10, 25, 50, 75, 90, 95, 100]))
        plt.hist(data)
        plt.show()

