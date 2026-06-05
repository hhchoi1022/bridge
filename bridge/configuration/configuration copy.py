#%%
import json
from pathlib import Path
from astropy.time import Time
import numpy as np
from typing import Union
#%%
class Configuration:
    """
    Base Configuration class for all modules.
    Loads multiple JSON config files and merges them.
    Later files in the list override earlier ones.
    """

    def __init__(self, config_filenames: Union[str, list[str]]):
        self._dict = {}
        self._config_paths = []
        config_filenames = np.atleast_1d(config_filenames)

        for filename in config_filenames:
            config_path = Path(__file__).parent / filename
            try:
                self._dict.update(self._load_config(config_path))
                self._config_paths.append(config_path)
            except FileNotFoundError:
                print(f"⚠️ Config file {config_path} not found, skipping.")

    def __repr__(self):
        attrs = {k: (v.iso if isinstance(v, Time) else v) for k, v in self._dict.items()}
        max_key_len = max(len(key) for key in attrs.keys()) if attrs else 0
        attrs_str = "\n".join([f"{k:{max_key_len}} : {v}" for k, v in attrs.items()])
        return (f"===== Configuration =====\n"
                f"{attrs_str}")

    def __getattr__(self, name):
        if name in self._dict:
            return self._dict[name]
        raise AttributeError(f"Attribute {name} not found")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._dict[name] = value

    def _load_config(self, config_path: Path):
        with open(config_path, "r") as f:
            return json.load(f)

    def update(self, **kwargs):
        """Update config dictionary with provided key-value pairs."""
        self._dict.update(kwargs)
