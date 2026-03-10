from pathlib import Path

import torch
import yaml

from .models.gnp import PatchGNP


def load_config(path: Path) -> dict:
    """
    Load a configuration file from a yaml file.

    Parameters
    ----------
    path : str
        Path to the yaml file.

    Returns
    -------
    dict
        Dictionary containing the configuration parameters.
    """
    if not path.exists():
        raise OSError(f"Path {path} Not Found")

    with open(path, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def load_model(config: dict, model_path: Path, device: str) -> PatchGNP:
    """
    Load a model from a directory.

    Parameters
    ----------
    model_dir : Path
        Path to the model directory.

    Returns
    -------
    PatchGNP
        The loaded model.
    """
    if not model_path.exists():
        raise OSError(f"Path {model_path} Not Found")

    model = PatchGNP(device=device, **config)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model.to(device)
