"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os

# Use PyTorch backend for geomstats
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import os.path as osp
import sys
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import json
from glob import glob
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from manifm.datasets import get_loaders
from manifm.model_pl import ManifoldFMLitModule

torch.backends.cudnn.benchmark = True
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    logging.getLogger("pytorch_lightning").setLevel(logging.getLevelName("INFO"))

    if cfg.get("seed", None) is not None:
        pl.utilities.seed.seed_everything(cfg.seed)

    print(cfg)

    print("Found {} CUDA devices.".format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            "{} \t Memory: {:.2f}GB".format(
                props.name, props.total_memory / (1024**3)
            )
        )

    keys = [
        "SLURM_NODELIST",
        "SLURM_JOB_ID",
        "SLURM_NTASKS",
        "SLURM_JOB_NAME",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
    ]
    log.info(json.dumps({k: os.environ.get(k, None) for k in keys}, indent=4))

    cmd_str = " \\\n".join([f"python {sys.argv[0]}"] + ["\t" + x for x in sys.argv[1:]])
    with open("cmd.sh", "w") as fout:
        print("#!/bin/bash\n", file=fout)
        print(cmd_str, file=fout)

    log.info(f"CWD: {os.getcwd()}")

    # Load dataset
    train_loader, val_loader, test_loader = get_loaders(cfg)

    # Construct model
    model = ManifoldFMLitModule(cfg)
    print(model)

    # Checkpointing, logging, and other misc.
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            monitor="val/loss_best",
            mode="min",
            filename="epoch-{epoch:03d}_step-{global_step}_loss-{val_loss:.4f}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True,
            every_n_train_steps=cfg.get("ckpt_every", None),
        ),
        LearningRateMonitor(),
    ]

    slurm_plugin = pl.plugins.environments.SLURMEnvironment(auto_requeue=False)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cwd"] = os.getcwd()
    loggers = [pl.loggers.CSVLogger(save_dir=".")]
    if cfg.use_wandb:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        loggers.append(
            pl.loggers.WandbLogger(
                save_dir=".",
                name=f"{cfg.data}_{now}",
                project="ManiFM",
                log_model=False,
                config=cfg_dict,
                resume=True,
            )
        )
    trainer = pl.Trainer(
        max_steps=cfg.optim.num_iterations,
        accelerator="gpu",
        devices=1,
        logger=loggers,
        val_check_interval=cfg.val_every,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        precision=cfg.get("precision", 32),
        gradient_clip_val=cfg.optim.grad_clip,
        plugins=slurm_plugin if slurm_plugin.detect() else None,
        num_sanity_val_steps=0,
    )

    # If we specified a checkpoint to resume from, use it
    checkpoint = cfg.get("resume", None)

    # Check if a checkpoint exists in this working directory.  If so, then we are resuming from a pre-emption
    # This takes precedence over a command line specified checkpoint
    checkpoints = glob("checkpoints/**/*.ckpt", recursive=True)
    if len(checkpoints) > 0:
        # Use the checkpoint with the latest modification time
        checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]

    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)

    train_metrics = trainer.callback_metrics

    log.info("Starting testing!")
    ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        log.warning("Best ckpt not found! Using current weights for testing...")
        ckpt_path = None
    trainer.test(model, test_loader, ckpt_path=ckpt_path)
    log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    for k, v in metric_dict.items():
        metric_dict[k] = float(v)

    with open("metrics.json", "w") as fout:
        print(json.dumps(metric_dict), file=fout)

    return metric_dict


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        print(traceback.format_exc())
        sys.exit(1)
