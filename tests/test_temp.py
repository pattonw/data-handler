#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `data_handler` package."""


import unittest

import data_handler

"""
x = SkeletonFetcher()
whole, large, reviewed, split = (
    x.all_skeletons,
    x.large_skeletons,
    x.reviewed_skeletons,
    x.split_skeletons,
)
print(
    "Total {} skeletons, {} are long enough, {} are reviewed enough, {} contain splits".format(
        len(whole), len(large), len(reviewed), len(split)
    )
)

print(
    "{} are long and reviewed, {} are long and split, {} are reviewed and split".format(
        len(set(large) & set(reviewed)),
        len(set(large) & set(split)),
        len(set(reviewed) & set(split)),
    )
)
print(
    "{} are long, reviewed and split".format(
        len(set(large) & set(reviewed) & set(split))
    )
)


def inspect_dataset(dataset):
    print(len(dataset))


from calyx_source import JanSegmentationSource as ss

x = ss()
trees = list(x.missing_branches)
"""


class TestTemp(unittest.TestCase):
    """Tests for `data_handler` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""
        print(data_handler)
