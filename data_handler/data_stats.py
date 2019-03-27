from pathlib import Path
import pickle


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


def get_ranks_with_skids():
    data = load_data()
    skids_ranks = map(
        lambda x: (x[0], (x[3][0] + 0.5) / x[3][1], (x[4][0] + 0.5) / x[4][1]), data
    )
    return skids_ranks
