r"""Marginal log-likelihood for fitting a `SingleTaskQED` with BoTorch.

QPyTorch's `MultivariateQExponential.log_prob` falls back to the Gaussian
density when `fast_computations.log_prob` is off, and importing BoTorch turns
it off globally. So `qpytorch.mlls.ExactMarginalLogLikelihood` silently fits
a Gaussian likelihood whenever BoTorch is imported. The MLL below computes the
q-ED density directly and is registered with BoTorch, so the usual

    mll = QEDMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

works as it does for `SingleTaskGP`.
"""

from __future__ import annotations

import math
from types import NoneType

import torch
from botorch.fit import FitGPyTorchMLL, _fit_fallback
from botorch.optim.closures.model_closures import (
    GetLossClosure,
    _get_loss_closure_exact_internal,
)
from qpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


class QEDMarginalLogLikelihood(ExactMarginalLogLikelihood):
    r"""The q-ED marginal log-likelihood of the training targets.

    With `r = (y - m)^T K^{-1} (y - m)` and `n = len(y)`, the q-ED_n(m, K)
    log-density is

        log p(y) = log(q/2) - (n/2) log(2 pi) - 1/2 log|K|
                   + (q/2 - 1)(n/2) log r - 1/2 r^(q/2)

    which is the Gaussian log-likelihood at `q = 2`. Priors and added loss
    terms on the model are included, and the result is divided by `n`.
    """

    def forward(self, function_dist, target: Tensor, *params, **kwargs) -> Tensor:
        marginal = self.likelihood(function_dist, *params, **kwargs)
        power = marginal.power
        diff = target - marginal.mean
        covar = marginal.lazy_covariance_matrix.evaluate_kernel()
        inv_quad, logdet = covar.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)
        n = diff.size(-1)
        res = -0.5 * (inv_quad.pow(power / 2.0) + logdet + n * math.log(2 * math.pi))
        if float(power) != 2.0:
            res = res + 0.5 * n * (power / 2.0 - 1.0) * inv_quad.log() + torch.log(power / 2.0)
        res = self._add_other_terms(res, params)
        return res.div(function_dist.event_shape.numel())


FitGPyTorchMLL.register(QEDMarginalLogLikelihood, object, object)(_fit_fallback)
GetLossClosure.register(QEDMarginalLogLikelihood, object, object, NoneType)(
    _get_loss_closure_exact_internal
)
