#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Sampling-based generation strategies.

A SamplingStrategy returns samples from the input points (i.e. Tensors in feature
space), rather than the value for a set of tensors, as acquisition functions do.
The q-batch dimension has similar semantics as for acquisition functions in that the
points across the q-batch are considered jointly for sampling (where as for
q-acquisition functions we evaluate the joint value of the q-batch).
"""

# from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    # PosteriorTransform,
    # ScalarizedPosteriorTransform,
)
# from botorch.generation.utils import _flip_sub_unique
from botorch.models.model import Model
from torch import Tensor
from torch.nn import Module


def _flip_sub_unique(x: Tensor, k: int) -> Tensor:
    """Get the first k unique elements of a single-dimensional tensor, traversing the
    tensor from the back.
    Args:
        x: A single-dimensional tensor
        k: the number of elements to return
    Returns:
        A tensor with min(k, |x|) elements.
    Example:
        >>> x = torch.tensor([1, 6, 4, 3, 6, 3])
        >>> y = _flip_sub_unique(x, 3)  # tensor([3, 6, 4])
        >>> y = _flip_sub_unique(x, 4)  # tensor([3, 6, 4, 1])
        >>> y = _flip_sub_unique(x, 10)  # tensor([3, 6, 4, 1])
    NOTE: This should really be done in C++ to speed up the loop. Also, we would like
    to make this work for arbitrary batch shapes, I'm sure this can be sped up.
    """
    n = len(x)
    i = 0
    out = set()
    idcs = torch.empty(k, dtype=torch.long)
    for j, xi in enumerate(x.flip(0).tolist()):
        if xi not in out:
            out.add(xi)
            idcs[i] = n - 1 - j
            i += 1
        if len(out) >= k:
            break
    return x[idcs[: len(out)]]


class SamplingStrategy(Module, ABC):
    r"""
    Abstract base class for sampling-based generation strategies.

    :meta private:
    """

    @abstractmethod
    def forward(self, X: Tensor, num_samples: int = 1, **kwargs: Any) -> Tensor:
        r"""Sample according to the SamplingStrategy.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension).
            num_samples: The number of samples to draw.
            kwargs: Additional implementation-specific kwargs.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """

        pass  # pragma: no cover


class MaxPosteriorSampling(SamplingStrategy):
    r"""Sample from a set of points according to their max posterior value.

    Example:
        >>> MPS = MaxPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform = None,
        replacement: bool = True,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.model = model
        if objective is None:
            objective = IdentityMCObjective()
        else:
            assert 0 
        # elif not isinstance(objective, MCAcquisitionObjective):
        #     # TODO: Clean up once ScalarizedObjective is removed.
        #     if posterior_transform is not None:
        #         raise RuntimeError(
        #             "A ScalarizedObjective (DEPRECATED) and a posterior transform "
        #             "are not supported at the same time. Use only a posterior "
        #             "transform instead."
        #         )
        #     else:
        #         posterior_transform = ScalarizedPosteriorTransform(
        #             weights=objective.weights, offset=objective.offset
        #         )
        #         objective = IdentityMCObjective()
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.replacement = replacement

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X,
            observation_noise=observation_noise,
            posterior_transform=self.posterior_transform,
        )
        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        return self.maximize_samples(X, samples, num_samples)

    def maximize_samples(self, X: Tensor, samples: Tensor, num_samples: int = 1):
        obj = self.objective(samples, X=X)  # num_samples x batch_shape x N
        if self.replacement:
            # if we allow replacement then things are simple(r)
            idcs = torch.argmax(obj, dim=-1)
        else:
            # if we need to deduplicate we have to do some tensor acrobatics
            # first we get the indices associated w/ the num_samples top samples
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            # generate some indices to smartly index into the lower triangle of
            # idcs_full (broadcasting across batch dimensions)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            # pick the unique indices in order - since we look at the lower triangle
            # of the index matrix and we don't sort, this achieves deduplication
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                # TODO: Find a better way to do this
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                    [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
                    dim=-1,
                )
            else:
                # TODO: Find a general way to do this efficiently.
                raise NotImplementedError(
                    "MaxPosteriorSampling without replacement for more than a single "
                    "batch dimension is not yet implemented."
                )
        # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # to have shape batch_shape x num_samples
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # in order to use gather, we need to repeat the index tensor d times
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        # now if the model is batched batch_shape will not necessarily be the
        # batch_shape of X, so we expand X to the proper shape
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        # finally we can gather along the N dimension
        return torch.gather(Xe, -2, idcs)
