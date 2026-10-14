r"""Acquisition functions for q-ED-BO.

The q-ED predictive at a point is q-ED_1(mu_t(x), sigma_t^2(x)), where mu_t and
sigma_t^2 are the GP posterior mean and variance. Both acquisitions below
reduce to their GP counterparts at q = 2.

    QEDUpperConfidenceBound (q-UCB):
        alpha(x) = mu(x) + Lambda_t^(1/q) * sigma(x)
    QEDExpectedImprovement (q-EI):
        alpha(x) = E[(f(x) - f+)_+] under the q-ED predictive, in closed form
    qQEDUpperConfidenceBound:
        Monte Carlo q-UCB for a batch of points

For batch Expected Improvement, BoTorch's `qLogExpectedImprovement` and
`qLogNoisyExpectedImprovement` already work, since they average over samples
from the q-ED predictive.
"""

from __future__ import annotations

import math

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


def c_q(power: float | Tensor, n: float = 1.0) -> Tensor:
    r"""The q-ED variance-scaling constant (Gomez et al., 1998).

        c(q, n) = 2^(2/q) * Gamma(n/2 + 2/q) / (n * Gamma(n/2))

    A q-ED_n(m, K) vector has covariance c(q, n) K. `c(2, n) = 1` for all
    `n`, which recovers the Gaussian. Computed in log-space.
    """
    power = torch.as_tensor(power, dtype=torch.double)
    n = torch.as_tensor(float(n), dtype=power.dtype, device=power.device)
    log_c = (
        (2.0 / power) * math.log(2.0)
        + torch.lgamma(n / 2.0 + 2.0 / power)
        - torch.log(n)
        - torch.lgamma(n / 2.0)
    )
    return torch.exp(log_c)


def ucb_beta(t: int, num_candidates: int, delta: float = 0.1) -> float:
    r"""The confidence schedule Lambda_t of q-UCB.

        Lambda_t = 2 log(|X| t^2 pi^2 / (6 delta))

    This is GP-UCB's beta_t (Srinivas et al., 2010). GP-UCB uses
    `Lambda_t^(1/2)` and q-UCB uses `Lambda_t^(1/q)`.
    """
    t = max(int(t), 1)
    return 2.0 * math.log(num_candidates * t * t * math.pi**2 / (6.0 * delta))


class QEDUpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""q-Upper Confidence Bound (q-UCB).

    `q-UCB(x) = mu(x) + beta^(1/q) * sigma(x)`, where `mu` and `sigma^2` are
    the parameters of the q-ED predictive of a `SingleTaskQED`. `beta` is the
    schedule Lambda_t (see `ucb_beta`). At `q = 2` this is exactly
    `UpperConfidenceBound`.

    Example:
        >>> model = SingleTaskQED(train_X, train_Y, power=1.0)
        >>> q_ucb = QEDUpperConfidenceBound(model, beta=0.2, power=1.0)
        >>> acq_value = q_ucb(test_X)
    """

    def __init__(
        self,
        model,
        beta: float | Tensor,
        power: float | Tensor,
        posterior_transform=None,
        maximize: bool = True,
    ) -> None:
        r"""
        Args:
            model: A fitted single-outcome `SingleTaskQED`.
            beta: A scalar or a `b`-dim tensor (batch mode), the schedule
                Lambda_t.
            power: The shape parameter `q` of the q-ED predictive.
            posterior_transform: A `PosteriorTransform`.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("power", torch.as_tensor(power))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate q-UCB on a `(b1 x ... bk) x 1 x d` candidate set."""
        mean, sigma = self._mean_and_sigma(X)
        ucb = (mean if self.maximize else -mean) + self.beta.pow(1.0 / self.power) * sigma
        return ucb.squeeze(-1)


class QEDExpectedImprovement(AnalyticAcquisitionFunction):
    r"""q-Expected Improvement (q-EI).

    The exact expected improvement under the q-ED predictive:

        d = (mu(x) - best_f) / sigma(x)            [sign-flipped if minimize]
        F_q(z) = Phi(sign(z) * |z|^(q/2))           standardized CDF
        M_q(a) = 2^(1/q-1)/sqrt(pi) * Gamma(1/2+1/q, a^q/2)
        alpha(x) = sigma(x) * [d * F_q(d) + M_q(|d|)]

    where `Gamma(s, x)` is the upper incomplete gamma function. At `q = 2`
    this is exactly `ExpectedImprovement`.

    Example:
        >>> model = SingleTaskQED(train_X, train_Y, power=1.0)
        >>> q_ei = QEDExpectedImprovement(model, best_f=0.2, power=1.0)
        >>> acq_value = q_ei(test_X)
    """

    def __init__(
        self,
        model,
        best_f: float | Tensor,
        power: float | Tensor,
        posterior_transform=None,
        maximize: bool = True,
    ) -> None:
        r"""
        Args:
            model: A fitted single-outcome `SingleTaskQED`.
            best_f: A scalar or a `b`-dim tensor, the best value observed so far.
            power: The shape parameter `q` of the q-ED predictive.
            posterior_transform: A `PosteriorTransform`.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("power", torch.as_tensor(power))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate q-EI on a `(b1 x ... bk) x 1 x d` candidate set."""
        mean, sigma = self._mean_and_sigma(X)
        sigma = sigma.clamp_min(1e-9)
        power = self.power.to(sigma)
        sign = 1.0 if self.maximize else -1.0
        d = sign * (mean - self.best_f) / sigma
        a = d.abs()
        # F_q(0) = Phi(0) = 1/2 since sign(0) = 0.
        Fq = 0.5 * (1.0 + torch.erf(torch.sign(d) * a.pow(power / 2.0) / math.sqrt(2)))
        # Gamma(s, x) = Q(s, x) * Gamma(s), Q the regularized upper incomplete gamma.
        s = 0.5 + 1.0 / power
        Mq = (
            2.0 ** (1.0 / power - 1.0)
            / math.sqrt(math.pi)
            * torch.exp(torch.lgamma(s))
            * torch.special.gammaincc(s.expand_as(a), a.pow(power) / 2.0)
        )
        return (sigma * (d * Fq + Mq)).squeeze(-1)


class qQEDUpperConfidenceBound(qUpperConfidenceBound):
    r"""Monte Carlo batch q-UCB for a single-output `SingleTaskQED`.

        alpha(X) = E[max_j mu_j + beta^(1/q) |Y_j - mu_j| / m_b]

    where `Y` is a joint sample from the `b`-dimensional q-ED predictive over
    the points of `X` and `m_b = 2^(1/q) Gamma(b/2 + 1/q) / (sqrt(pi)
    Gamma((b+1)/2))` is the mean absolute value of one standardized coordinate
    of a q-ED_b, so each term has mean `mu_j + beta^(1/q) sigma_j`. At `b = 1`
    this is `QEDUpperConfidenceBound` in expectation, and at `q = 2` it is
    BoTorch's `qUpperConfidenceBound`.

    Example:
        >>> model = SingleTaskQED(train_X, train_Y, power=1.0)
        >>> q_ucb = qQEDUpperConfidenceBound(model, beta=0.2, power=1.0)
        >>> acq_value = q_ucb(test_X)  # test_X: batch x b x d
    """

    def __init__(
        self,
        model,
        beta: float,
        power: float,
        sampler=None,
        objective=None,
        posterior_transform=None,
        X_pending: Tensor | None = None,
    ) -> None:
        r"""
        Args:
            model: A fitted single-outcome `SingleTaskQED`.
            beta: The schedule Lambda_t (see `ucb_beta`).
            power: The shape parameter `q` of the q-ED predictive.
            sampler: The sampler used to draw base samples.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `m x d`-dim Tensor of pending points, concatenated
                into X upon forward call.
        """
        super().__init__(
            model=model,
            beta=beta,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.power = float(power)
        self.beta_prime = beta ** (1.0 / self.power)

    def _sample_forward(self, obj: Tensor) -> Tensor:
        b, q = obj.shape[-1], self.power
        log_m = (
            math.log(2.0) / q
            + math.lgamma(b / 2 + 1.0 / q)
            - 0.5 * math.log(math.pi)
            - math.lgamma((b + 1) / 2)
        )
        mean = obj.mean(dim=0)
        return mean + self.beta_prime * math.exp(-log_m) * (obj - mean).abs()
