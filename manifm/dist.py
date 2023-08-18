"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
import torch
import torch.nn as nn


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.0)
    log_std = log_std + torch.tensor(0.0)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


class GaussianMM(nn.Module):
    def __init__(self, centers, std):
        super().__init__()
        self.register_buffer("centers", torch.tensor(centers))
        self.register_buffer("logstd", torch.tensor(std).log())
        self.K = self.centers.shape[0]

    def logprob(self, x):
        """Computes the log probability."""
        logprobs = normal_logprob(
            x.unsqueeze(1), self.centers.unsqueeze(0), self.logstd
        )
        logprobs = torch.sum(logprobs, dim=2)
        return torch.logsumexp(logprobs, dim=1) - math.log(self.K)

    def sample(self, n_samples):
        idx = torch.randint(self.K, (n_samples,)).to(self.centers.device)
        mean = self.centers[idx]
        return torch.randn_like(mean) * torch.exp(self.logstd) + mean
