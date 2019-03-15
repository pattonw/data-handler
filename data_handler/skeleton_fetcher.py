import catpy
import json
from typing import NamedTuple, List, Dict
import random
import pickle
from pathlib import Path
import datetime

from sarbor.skeletons import Skeleton

"""
A script to gather data.
handles fetching from catmaid and compiling skeletons into
reusable datasets

GOAL: make datasets easily obtainable and replicable
"""


class Coord:
    def __init__(self, x: int, y: int, z: int):
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, new_x: int):
        self._x = new_x

    @property
    def y(self) -> int:
        return self._y

    @y.setter
    def y(self, new_y: int):
        self._x = new_y

    @property
    def z(self) -> int:
        return self._z

    @z.setter
    def z(self, new_z: int):
        self._z = new_z


def _parse_date(date: str):
    if date[-3] == ":":
        date = "{}{}".format(date[:-3], date[-2:])
    date = date.replace("T", " ")
    try:
        return datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f%z")
    except Exception as e:
        print(e)
        return datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S%z")


class Log:
    def __init__(
        self,
        user: str,
        operation: str,
        datetimestring: str,
        x: float,
        y: float,
        z: float,
        text: str,
    ):
        self._user = user
        self._operation = operation
        self._datetimestring = datetimestring
        self._x = x
        self._y = y
        self._z = z
        self._text = text

    @property
    def user(self) -> str:
        return self._user

    @user.setter
    def user(self, new_user: str):
        self._user = new_user

    @property
    def operation(self) -> str:
        return self._operation

    @operation.setter
    def operation(self, new_operation: str):
        self._operation = new_operation

    @property
    def datetimestring(self) -> str:
        return self._datetimestring

    @datetimestring.setter
    def datetimestring(self, new_datetimestring: str):
        self._datetimestring = new_datetimestring

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, new_x: float):
        self._x = new_x

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, new_y: float):
        self._y = new_y

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, new_z: float):
        self._z = new_z

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, new_text: str):
        self._text = new_text

    @property
    def skeleton_id(self) -> int:
        return int(self.text.split(" ")[4])

    @property
    def datetime(self) -> datetime.datetime:
        return _parse_date(self.datetimestring)

    def to_tuple(self):
        return (
            self.user,
            self.operation,
            self.datetimestring,
            self.x,
            self.y,
            self.z,
            self.text,
        )


class BoundingBox():
    def __init__(self, low: Coord, high: Coord):
        self._low = low
        self._high = high

    @property
    def low(self) -> Coord:
        return self._low

    @low.setter
    def low(self, new_low: Coord):
        self._low = new_low

    @property
    def high(self) -> Coord:
        return self._high

    @high.setter
    def high(self, new_high: Coord):
        self._high = new_high


class CatmaidQueryHandler:
    def __init__(self):
        self._api_key = None
        self._sensitives_file_name = None
        self._catpy_client = None

    @property
    def sensitives_file(self) -> str:
        if self._sensitives_file_name is None:
            return "sensitives.json"
        else:
            return self._sensitives_file_name

    @property
    def sensitives(self) -> Dict[str, str]:
        if Path(self.sensitives_file).exists():
            return json.load(open(self.sensitives_file, "r"))
        else:
            raise ValueError(
                "You must have a sensitives.json file at {}".format(
                    str(Path(self.sensitives_file).absolute())
                )
            )

    @property
    def api_key(self) -> str:
        if "API_token_FAFB" not in self.sensitives:
            raise KeyError("You must have an API token for your Catmaid instance!")
        return self.sensitives["API_token_FAFB"]

    @property
    def catpy_client(self):
        if self._catpy_client is None:
            return catpy.CatmaidClient("http://localhost:8000/", self.api_key)
        return self._catpy_client

    def skeletons_in_roi(self, roi: BoundingBox):
        params = {
            "minx": roi.low.x,
            "maxx": roi.high.x,
            "miny": roi.low.y,
            "maxy": roi.high.y,
            "minz": roi.low.z,
            "maxz": roi.high.z,
        }
        return self.catpy_client.get(["1", "skeletons", "in-bounding-box"], params)

    def reviewed_statuses(self, skeleton_ids: List[int]) -> Dict[int, float]:
        """
        Strange parsing of params on the server is the cause of wierd params variable.
        A list a is expected to be passed as dictionaries of the form:
            {"a[0]":a[0], "a[1]":a[1],...}
        rather than one key value pair: 
            {"a": a}.
        """
        params = {
            "skeleton_ids[{}]".format(i): skeleton_ids[i]
            for i in range(len(skeleton_ids))
        }
        return {
            int(k): v[1] / v[0]
            for k, v in self.catpy_client.post(
                ["1", "skeletons", "review-status"], data=params
            ).items()
        }

    def cable_length(self, skeleton_ids: List[int]):
        params = {
            "skeleton_ids[{}]".format(i): skeleton_ids[i]
            for i in range(len(skeleton_ids))
        }
        return {
            int(k): v
            for k, v in self.catpy_client.post(
                ["1", "skeletons", "cable-length"], data=params
            ).items()
        }

    def splits(self) -> List[Log]:
        params = {
            "iDisplayStart": 0,
            "iDisplayLength": 23431,
            "iSortCol_0": 2,
            "iSortingCols": 1,
            "sSortDir_0": "DESC",
            "operation_type": "split_skeleton",
        }
        log_data = [
            Log(log[0], log[1], log[2], log[3], log[4], log[5], log[6])
            for log in self.catpy_client.post(["1", "logs", "list"], params)["aaData"]
        ]
        return log_data

    def skeleton_compact_detail(
        self, skeleton_ids: List[int], with_history: bool = False
    ):
        compact_skeletons = {}
        for k in range(0, len(skeleton_ids), 500):
            params = {
                "skeleton_ids[{}]".format(i): skeleton_ids[i + k]
                for i in range(min(500, len(skeleton_ids) - k))
            }
            params["with_history"] = with_history
            compact_skeletons.update(
                self.catpy_client.post(
                    ["1", "skeletons", "compact-detail"], data=params
                )["skeletons"]
            )
        return compact_skeletons


class SkeletonFetcher:
    def __init__(self):
        self._roi = None

        self._all_skeleton_ids = None
        self._reviewed_skeleton_ids = None
        self._large_skeleton_ids = None
        self._split_skeleton_ids = None
        self._catmaid_query_handler = None

    @property
    def query(self):
        if self._catmaid_query_handler is None:
            self._catmaid_query_handler = CatmaidQueryHandler()
        return self._catmaid_query_handler

    @property
    def min_cable_length(self) -> float:
        return 10000

    @property
    def min_reviewed_percent(self) -> float:
        return 0.1

    @property
    def roi(self) -> BoundingBox:
        if self._roi is None:
            return BoundingBox(
                Coord(403560, 121800, 158000), Coord(467560, 173800, 234000)
            )
        else:
            return self._roi

    @property
    def random_seed(self) -> int:
        return 1

    @property
    def all_skeleton_ids_file(self):
        return Path("skeletons", "all_skeleton_ids.obj")

    @property
    def all_skeletons(self) -> List[int]:
        if self.all_skeleton_ids_file.exists():
            return pickle.load(open(self.all_skeleton_ids_file, "rb"))
        if self._all_skeleton_ids is None:
            self._all_skeleton_ids = self.query.skeletons_in_roi(self.roi)
            # pickle.dump(self._all_skeleton_ids, open(self.all_skeleton_ids_file, "wb"))
        return self._all_skeleton_ids

    @property
    def large_skeleton_ids_file(self):
        """
        skeletons with cablelength above the minimum
        """
        return Path("skeletons", "large_skeleton_ids.obj")

    @property
    def large_skeletons(self) -> List[int]:
        """
        skeletons with cablelength above the minimum
        """
        if self._large_skeleton_ids is None:
            if self.large_skeleton_ids_file.exists():
                return pickle.load(open(self.large_skeleton_ids_file, "rb"))
            self._large_skeleton_ids = self.get_large_skeletons()
            pickle.dump(
                self._large_skeleton_ids, open(self.large_skeleton_ids_file, "wb")
            )
        return self._large_skeleton_ids

    @property
    def reviewed_skeleton_ids_file(self):
        """
        skeletons with at least the minimum percent of reviewed nodes
        """
        return Path("skeletons", "reviewed_skeleton_ids.obj")

    @property
    def reviewed_skeletons(self) -> List[int]:
        """
        skeletons with at least the minimum percent of reviewed nodes
        """
        if self._reviewed_skeleton_ids is None:
            if self.reviewed_skeleton_ids_file.exists():
                return pickle.load(open(self.reviewed_skeleton_ids_file, "rb"))
            self._reviewed_skeleton_ids = self.get_reviewed_skeletons()
            pickle.dump(
                self._reviewed_skeleton_ids, open(self.reviewed_skeleton_ids_file, "wb")
            )
        return self._reviewed_skeleton_ids

    @property
    def split_skeleton_ids_file(self):
        """
        skeletons that have been split
        """
        return Path("skeletons", "split_skeleton_ids.obj")

    @property
    def split_skeletons(self):
        if self._split_skeleton_ids is None:
            if self.split_skeleton_ids_file.exists():
                return pickle.load(open(self.split_skeleton_ids_file, "rb"))
            self._split_skeleton_ids = self.get_split_skeletons()
            pickle.dump(
                self._split_skeleton_ids, open(self.split_skeleton_ids_file, "wb")
            )
        return self._split_skeleton_ids

    def get_large_skeletons(self, length_cutoff: float = 0):
        length_cutoff = min(length_cutoff, self.min_cable_length)
        skeleton_lengths = self.query.cable_length(self.all_skeletons)
        filtered_skeletons = [
            skid for skid, length in skeleton_lengths.items() if length > length_cutoff
        ]
        print("{} skeletons left".format(len(filtered_skeletons)))
        return filtered_skeletons

    def get_reviewed_skeletons(self, review_cutoff: float = 0):
        review_cutoff = min(review_cutoff, self.min_reviewed_percent)
        skeleton_lengths = self.query.reviewed_statuses(self.all_skeletons)
        filtered_skeletons = [
            skid
            for skid, reviewed_percent in skeleton_lengths.items()
            if reviewed_percent > review_cutoff
        ]
        print("{} skeletons left".format(len(filtered_skeletons)))
        return filtered_skeletons

    def get_split_skeletons(self):
        split_logs = self.query.splits()
        split_skeleton_ids = [split.skeleton_id for split in split_logs]
        filtered_skeletons = {
            skid: split_logs[split_skeleton_ids.index(skid)]
            for skid in self.all_skeletons
            if skid in split_skeleton_ids
        }
        return filtered_skeletons

    def get_skeletons(self, n: int) -> List[int]:
        random.seed(self.random_seed)
        skeletons = self.all_skeletons
        if n < 0 or n > len(skeletons):
            return skeletons
        else:
            return random.sample(self.all_skeletons, n)

    def chop_skeletons(
        self, skeleton_nodes, num_branch_chops: int, num_segment_chops: int
    ):
        """
        return format:
        {
            skeleton_id: {
                "skeleton_nodes": List[Tuple[nid: int,
                                            pid: Optional[int],
                                            x:int,
                                            y:int,
                                            z:int,
                                            strahler:int]],
                "removed_branch_nodes": List[Tuple[keep: int, drop: int]],
                "removed_segment_nodes": List[Tuple[keep: int, drop: int]]
            }
        }
        """
        results = {}
        for skid, nodes in skeleton_nodes.items():
            nids, pids, uids, x_coord, y_coord, z_coord, radii, confidences = zip(
                *nodes
            )
            skeleton = Skeleton()
            skeleton.input_nid_pid_x_y_z(zip(nids, pids, x_coord, y_coord, z_coord))
            branches = skeleton.get_interesting_nodes(branches=True)
            branch_neighbors = set(
                [
                    (keep.key, drop.key)
                    for keep in branches
                    for drop in keep.get_neighbors()
                ]
            )
            if len(branch_neighbors) < num_branch_chops:
                continue
            branch_chops = random.sample(branch_neighbors, num_branch_chops)

            segment_nodes = list(filter(lambda x: x.is_regular(), skeleton.get_nodes()))
            segment_neighbors = set(
                [
                    (keep.key, drop.key)
                    for keep in segment_nodes
                    for drop in keep.get_neighbors()
                ]
            )
            if len(segment_neighbors) < num_segment_chops:
                continue
            segment_chops = random.sample(segment_neighbors, num_segment_chops)
            skeleton.calculate_strahlers()
            results[skid] = {
                "skeleton_nodes": [
                    (
                        node.key,
                        node.parent_key,
                        node.value.center[0],
                        node.value.center[1],
                        node.value.center[2],
                        node.strahler,
                    )
                    for node in skeleton.nodes.values()
                ],
                "removed_branch_nodes": branch_chops,
                "removed_segment_nodes": segment_chops,
            }
            if len(results) == 100:
                break

        return results

    def build_missing_branch_dataset(
        self,
        num_skeletons: int = 100,
        num_branch_chops: int = 10,
        num_segment_chops: int = 5,
    ):
        """
        Missing branch dataset should be composed of large neurons that have been well reviewed.
        
        This dataset is comprised of 100 skeletons. Each skeleton will have:
        10 random branches removed (Missing branch)
        5 random segments ended early (Early stop)
        for a total of 1500 missing branch datapoints

        Note, the whole skeleton does not have to be contained in the calyx, just the
        nodes of interest and preferably a large number of the attached nodes

        Each sample point will be stored in a dict of the form:
        {
            skeleton_id: {
                "skeleton_nodes": List[Tuple[nid: int,
                                            pid: Optional[int],
                                            x:int,
                                            y:int,
                                            z:int]],
                "removed_branch_nodes": List[Tuple[keep: int, drop: int]],
                "removed_segment_nodes": List[Tuple[keep: int, drop: int]]
            }
        }
        """
        random.seed(self.random_seed)
        large_reviewed_skeletons = list(
            set(self.large_skeletons) & set(self.reviewed_skeletons)
        )
        missing_branch_skeleton_ids = random.sample(
            large_reviewed_skeletons, num_skeletons * 5
        )
        skeleton_compact_details = self.query.skeleton_compact_detail(
            missing_branch_skeleton_ids
        )
        missing_branch_dataset = self.chop_skeletons(
            {k: v[0] for k, v in skeleton_compact_details.items()},
            num_branch_chops,
            num_segment_chops,
        )
        pickle.dump(missing_branch_dataset, open("missing_branch_dataset.obj", "wb"))
        return missing_branch_dataset

    def recreate_skeleton(self, nodes: List, date: datetime.datetime) -> Skeleton:
        """
        If a node does not end, it is initialized with an end time slightly before the start
        time. Thus we must check that x.end is either after the time we are searching for
        or before the start time.

        nodes come in form [nid, pid, uid, x, y, z, raidius, confidence, end, start, something]
        """
        date = date - datetime.timedelta(milliseconds=1000)
        current_nodes = list(
            filter(
                lambda x: _parse_date(x[9]) < date
                and (
                    _parse_date(x[8]) > date or _parse_date(x[8]) <= _parse_date(x[9])
                ),
                nodes,
            )
        )
        current_nodes = [(x[0], x[1], x[3], x[4], x[5]) for x in current_nodes]

        return current_nodes

    def get_pre_split_skeletons(self, compact_nodes: Dict[int, List], splits):
        results = {}
        k = 0
        for skid, nodes in compact_nodes.items():
            k += 1
            if k % 100 == 0:
                print(
                    "{}/{}: {}\%".format(k, len(compact_nodes), k / len(compact_nodes))
                )
            split = splits[int(skid)]
            pre_split = self.recreate_skeleton(nodes, split.datetime)
            results[skid] = {"split_log": split.to_tuple(), "skeleton_nodes": pre_split}
        return results

    def build_false_merge_dataset(self):
        """
        Missing branch dataset is comprised of skeletons containing historical splits and some
        artificial versions.

        This dataset is comprised of 1000 skeletons with historical splits.
        1000 since each skeleton will only provide one interesting datapoint.
        Note node_a represents the node on the side of the split that keeps its skeleton id,
        node_b is the node that recieves a new skeleton id,
        Skeletons will be stored in dicts as follows:
        {
            "skeleton_id": int,
            "split_log": Tuple[user, operation, datetime, x, y, z, text]
            "skeleton_nodes": List[Tuple[nid:int,
                                         pid:Optional[int],
                                         x:int,
                                         y:int,
                                         z:int]
        }
        """

        skeleton_compact_details = self.query.skeleton_compact_detail(
            list(self.split_skeletons.keys()), with_history=True
        )
        pre_split_skeletons = self.get_pre_split_skeletons(
            {k: v[0] for k, v in skeleton_compact_details.items()}, self.split_skeletons
        )

        pickle.dump(pre_split_skeletons, open("false_merge_dataset.obj", "wb"))

        return pre_split_skeletons

