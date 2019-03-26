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
        self.downsample = bool(settings.get("downsample", True))
        self.downsample_delta = np.array(settings.get("downsample_delta", 500))
        self.strahler_filter = bool(settings.get("strahler_filter", True))
        self.min_strahler = int(settings.get("min_strahler", 0))
        self.max_strahler = int(settings.get("max_strahler", 10000))


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
