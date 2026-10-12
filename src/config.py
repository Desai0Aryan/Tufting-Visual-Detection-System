"""YAML config loading with --set key.path=value overrides."""
import copy
from pathlib import Path

import yaml


def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like section.key=value")
        set_key(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def set_key(cfg, dotted, value):
    """Set cfg[a][b][c] for 'a.b.c'. Unknown keys are errors (typos must not
    silently do nothing), except inside the free-form `aug` section."""
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"unknown config section {dotted!r}")
        node = node[p]
    if parts[-1] not in node and parts[0] != "aug":
        raise KeyError(f"unknown config key {dotted!r}")
    node[parts[-1]] = value


def dump_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(copy.deepcopy(cfg), sort_keys=False))
