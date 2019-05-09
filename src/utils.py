"""Small helpers."""
import os
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)
