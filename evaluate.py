"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
import click
import pytorch_lightning as pl

from manifm.datasets import get_loaders
from manifm.eval_utils import load_model
from manifm.model_pl import ManifoldFMLitModule


@click.group()
def cli():
    pass


@cli.command()
@click.argument("checkpoint")
def nll(checkpoint):
    cfg, model = load_model(checkpoint)
    _, _, test_loader = get_loaders(cfg)

    model = model.cuda()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
    )

    test_metrics = trainer.test(model, test_loader)
    print(test_metrics)


@cli.command()
@click.argument("checkpoint")
@click.option("-a", "--all", is_flag=True)
def visualize(checkpoint, all):
    cfg, model = load_model(checkpoint)
    _, _, test_loader = get_loaders(cfg)

    if all:
        test_batch = test_loader.dataset.dataset.data
        # idx = torch.randperm(test_batch.shape[0])[:6200]
        # test_batch = test_batch[idx]
    else:
        test_batch = next(iter(test_loader))

    model = model.cuda()

    if isinstance(test_batch, torch.Tensor):
        test_batch = test_batch.cuda()
    if isinstance(test_batch, dict):
        for k, v in test_batch.items():
            test_batch[k] = test_batch[k].cuda()

    # Bypass the cfg.visualize flag.
    model.visualize(test_batch, force=True)


if __name__ == "__main__":
    cli()
