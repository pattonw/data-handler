# -*- coding: utf-8 -*-
"""Global configuration objects.

This module contains boilerplate configuration objects for storing and loading
configuration state.
"""

import numpy as np
import pytoml as toml


class BaseConfig(object):
    """Base class for configuration objects.

    String representation yields TOML that should parse back to a dictionary
    that will initialize the same configuration object.
    """

    def __str__(self):
        sanitized = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                sanitized[k] = v.tolist()
            else:
                sanitized[k] = v
        return toml.dumps(sanitized)

    __repr__ = __str__

    def from_toml(self, *filenames):
        """Reinitializes this Config from a list of TOML configuration files.

        Existing settings are discarded. When multiple files are provided,
        configuration is overridden by later files in the list.

        Parameters
        ----------
        filenames : interable of str
            Filenames of TOML configuration files to load.
        """
        settings_collection = []
        for filename in filenames:
            with open(filename, "rb") as fin:
                settings_collection.append(toml.load(fin))

        if len(settings_collection) > 0:
            settings = settings_collection[0].copy()
            for s in settings_collection:
                for c in s:
                    if c in settings and isinstance(settings[c], dict):
                        settings[c].update(s[c])
                    else:
                        settings[c] = s[c]
        else:
            settings = {}

        return self.__init__(settings)


class SkeletonConfig(BaseConfig):
    """Configuration for the use of volumes.

    Attributes
    ----------
    downsample : boolean
        Whether or not to downsample the skeleton.
    downsample_delta : int
        Approximate distance between each downsampled node in nm.
    strahler_filter : boolean
        Whether or not to filter nodes by strahler index.
    min_strahler : int
        The minimum strahler to leave in the skeleton.
    max_strahler : int
        The maximum strahler to leave in the skeleton.
    """

    def __init__(self, settings):
        self.resample = bool(settings.get("resample", True))
        self.resample_delta = np.array(settings.get("resample_delta", 500))
        self.strahler_filter = bool(settings.get("strahler_filter", True))
        self.min_strahler = int(settings.get("min_strahler", 0))
        self.max_strahler = int(settings.get("max_strahler", 10000))
        self.path = settings.get("path", None)

    @property
    def nodes(self):
        import csv

        coords = []
        ids = []
        with open(self.path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")
            for row in reader:
                coords.append([int(float(x)) for x in row[2:]])
                if row[1].strip() == "null" or row[1].strip() == "none":
                    ids.append([int(float(row[0])), None])
                elif row[0] == row[1]:
                    ids.append([int(float(row[0])), None])
                else:
                    ids.append([int(float(x)) for x in row[:2]])
        return [ids[i] + coords[i] for i in range(len(ids))]


class Config(object):
    """A complete collection of configuration objects.
    """

    def __init__(self, settings_collection=None):
        if settings_collection is not None:
            settings = settings_collection[0].copy()
            for s in settings_collection:
                for c in s:
                    if c in settings and isinstance(settings[c], dict):
                        settings[c].update(s[c])
                    else:
                        settings[c] = s[c]
        else:
            settings = {}

        self.skeleton = SkeletonConfig(settings.get("skeleton", {}))

    def __str__(self):
        sanitized = {}
        for n, c in self.__dict__.items():
            if not isinstance(c, BaseConfig):
                sanitized[n] = c
                continue
            sanitized[n] = {}
            for k, v in c.__dict__.items():
                if isinstance(v, np.ndarray):
                    sanitized[n][k] = v.tolist()
                else:
                    sanitized[n][k] = v
        return toml.dumps(sanitized)

    def from_toml(self, *filenames):
        """Reinitializes this Config from a list of TOML configuration files.

        Existing settings are discarded. When multiple files are provided,
        configuration is overridden by later files in the list.

        Parameters
        ----------
        filenames : interable of str
            Filenames of TOML configuration files to load.
        """
        settings = []
        for filename in filenames:
            with open(filename, "rb") as fin:
                settings.append(toml.load(fin))

        return self.__init__(settings)

    def to_toml(self, filename):
        with open(filename, "w") as tomlfile:
            tomlfile.write(str(self))
