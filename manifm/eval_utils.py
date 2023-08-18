"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Union, Dict, Any
import os
from glob import glob
from omegaconf import OmegaConf
import torch

from manifm.model_pl import ManifoldFMLitModule


def get_job_directory(file_or_checkpoint: Union[str, Dict[str, Any]]) -> str:
    found = False
    if isinstance(file_or_checkpoint, dict):
        chkpnt = file_or_checkpoint
        key = [x for x in chkpnt["callbacks"].keys() if "Checkpoint" in x][0]
        file = chkpnt["callbacks"][key]["dirpath"]
    else:
        file = file_or_checkpoint

    hydra_files = []
    directory = os.path.dirname(file)
    while not found:
        hydra_files = glob(
            os.path.join(os.path.join(directory, ".hydra/config.yaml")),
            recursive=True,
        )
        if len(hydra_files) > 0:
            break
        directory = os.path.dirname(directory)
        if directory == "":
            raise ValueError("Failed to find hydra config!")
    assert len(hydra_files) == 1, "Found ambiguous hydra config files!"
    job_dir = os.path.dirname(os.path.dirname(hydra_files[0]))
    return job_dir


def load_model(checkpoint: str, eval_projx=None, atol=None, rtol=None):
    chkpnt = torch.load(checkpoint, map_location="cpu")
    job_dir = get_job_directory(checkpoint)
    cfg = OmegaConf.load(os.path.join(job_dir, ".hydra/config.yaml"))

    if eval_projx is not None:
        cfg.eval_projx = eval_projx
    
    if atol is not None:
        cfg.model.atol = atol

    if rtol is not None:
        cfg.model.rtol = rtol

    model = ManifoldFMLitModule(cfg)
    model.load_state_dict(chkpnt["state_dict"])
    return cfg, model
