# Portions adapted from BoTorch (https://github.com/pytorch/botorch),
# Copyright (c) Meta Platforms, Inc. and affiliates, used under the MIT license.

r"""The q-ED surrogate for BoTorch.

`SingleTaskQED` mirrors BoTorch's `SingleTaskGP`, but its predictive at a
point is the q-exponential distribution q-ED_1(mu_t(x), sigma_t^2(x)), with
the GP posterior mean and variance as parameters. `QEDPosterior` wraps the
q-ED predictive (a QPyTorch `MultivariateQExponential`) as a BoTorch
posterior. Adapted from BoTorch's `SingleTaskGP` and `GPyTorchPosterior`.
"""

from __future__ import annotations

import math
import warnings
from contextlib import ExitStack
from typing import TYPE_CHECKING

import botorch.acquisition.cached_cholesky as _cached_cholesky
import torch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.utils import (
    add_output_dim,
    gpt_posterior_settings,
    validate_input_scaling,
)
from botorch.models.utils.gpytorch_modules import (
    MIN_INFERRED_NOISE_LEVEL,
    get_covar_module_with_dim_scaled_prior,
)
from botorch.posteriors.base_samples import _reshape_base_samples_non_interleaved
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.torch import TorchPosterior
from botorch.sampling.get_sampler import GetSampler, _get_sampler_mvn
from botorch.sampling.pathwise.posterior_samplers import DrawMatheronPaths
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.module import Module
from gpytorch.priors import LogNormalPrior
from linear_operator import settings as linop_settings
from qpytorch.distributions import (
    MultitaskMultivariateQExponential,
    MultivariateQExponential,
)
from qpytorch.likelihoods import (
    FixedNoiseQExponentialLikelihood,
    QExponentialLikelihood,
)
from qpytorch.models import ExactQEP
from torch import Tensor
from torch.distributions import Normal

if TYPE_CHECKING:
    from botorch.acquisition.objective import PosteriorTransform  # pragma: no cover


class QEDPosterior(TorchPosterior):
    r"""The q-ED predictive of a `SingleTaskQED` as a BoTorch posterior.

    At one point this is the univariate q-ED_1(mu, sigma^2) predictive. Over
    `n` points jointly (batch acquisitions, Thompson sampling) it is the
    q-ED_n with the GP posterior covariance. A q-ED_n sample is `mu + R L s`,
    with `s` uniform on the unit sphere and `R^q ~ chi2(n)`. Given a standard
    normal base sample `z`, `s = z / |z|` and `|z|^2 ~ chi2(n)` are
    independent, so `z |z|^(2/q - 1)` has the right law. This keeps BoTorch's
    normal (and quasi-MC) base samplers valid. Outputs of a multi-output model
    are independent and get one radius each.
    """

    distribution: MultivariateQExponential

    def __init__(self, distribution: MultivariateQExponential) -> None:
        r"""The q-ED predictive as a BoTorch posterior.

        Args:
            distribution: A QPyTorch `MultivariateQExponential` distribution.
        """
        super().__init__(distribution=distribution)
        self._is_mt: bool = isinstance(distribution, MultitaskMultivariateQExponential)

    @property
    def mvn(self) -> MultivariateQExponential:
        r"""Expose the distribution as a backwards-compatible attribute."""
        return self.distribution

    @property
    def power(self) -> Tensor:
        return self.distribution.power

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The shape of a base sample used for constructing posterior samples."""
        return self.distribution.batch_shape + self.distribution.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        r"""The t-batch range.

        This is used in samplers to identify the t-batch component of the
        `base_sample_shape`. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        return (0, -2) if self._is_mt else (0, -1)

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008
    ) -> torch.Size:
        r"""Returns the shape of the samples produced by the posterior with
        the given `sample_shape`.
        """
        base_shape = self.distribution.batch_shape + self.distribution.event_shape
        if not self._is_mt:
            base_shape += torch.Size([1])
        return sample_shape + base_shape

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        r"""Sample from the posterior (with gradients) using base samples.

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: A Tensor of `N(0, I)` base samples of shape
                `sample_shape x base_sample_shape`, typically obtained from
                a `Sampler`. They are mapped to q-ED base samples.

        Returns:
            Samples from the posterior, a tensor of shape
            `self._extended_shape(sample_shape=sample_shape)`.
        """
        if base_samples.shape[: len(sample_shape)] != sample_shape:
            raise RuntimeError(
                "`sample_shape` disagrees with shape of `base_samples`. "
                f"Got {sample_shape=} and {base_samples.shape=}."
            )
        # Radius over the points of each output: dim -1 is n for a single
        # output, and base samples are `... x n x t` for multi-output.
        norm = base_samples.norm(dim=-2 if self._is_mt else -1, keepdim=True)
        power = self.power.to(base_samples)
        base_samples = base_samples * norm.clamp_min(1e-12).pow(2.0 / power - 1.0)
        if self._is_mt:
            base_samples = _reshape_base_samples_non_interleaved(
                mvn=self.distribution,
                base_samples=base_samples,
                sample_shape=sample_shape,
            )
        with ExitStack() as es:
            if linop_settings._fast_covar_root_decomposition.is_default():
                es.enter_context(linop_settings._fast_covar_root_decomposition(False))
            samples = self.distribution.rsample(
                sample_shape=sample_shape, base_samples=base_samples
            )
        if not self._is_mt:
            samples = samples.unsqueeze(-1)
        return samples

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        r"""Sample from the posterior (with gradients).

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.

        Returns:
            Samples from the posterior, a tensor of shape
            `self._extended_shape(sample_shape=sample_shape)`.
        """
        if sample_shape is None:
            sample_shape = torch.Size([1])
        base_samples = torch.randn(
            sample_shape + self.base_sample_shape, dtype=self.dtype, device=self.device
        )
        return self.rsample_from_base_samples(sample_shape, base_samples)

    @property
    def mean(self) -> Tensor:
        r"""The posterior mean."""
        mean = self.distribution.mean
        return mean if self._is_mt else mean.unsqueeze(-1)

    @property
    def variance(self) -> Tensor:
        r"""The scale parameter `sigma^2` of the q-ED predictive.

        This is the second argument of q-ED_1(mu, sigma^2), which equals the
        GP posterior variance. The predictive's actual variance is
        `c(q) sigma^2` with `c(q) = c(q, 1)` (see `qed_bo.c_q`). q-UCB and
        q-EI are written in terms of `sigma`, as in the paper.
        """
        variance = self.distribution.variance
        return variance if self._is_mt else variance.unsqueeze(-1)

    def quantile(self, value: Tensor) -> Tensor:
        r"""Quantiles of the univariate q-ED predictive.

        A standardized marginal `t` satisfies `t = sign(z) |z|^(2/q)` with
        `z ~ N(0, 1)`, so its `p`-quantile maps the normal one.
        """
        if value.numel() > 1:
            return torch.stack([self.quantile(v) for v in value], dim=0)
        z = Normal(0.0, 1.0).icdf(torch.as_tensor(value).to(self.mean))
        t = torch.sign(z) * z.abs().pow(2.0 / self.power.to(z))
        return self.mean + self.variance.sqrt() * t

    def density(self, value: Tensor) -> Tensor:
        r"""Density of the univariate q-ED predictive.

        For `t = (y - mu) / sigma`,
        `p(y) = (q/2) |t|^(q/2 - 1) phi(|t|^(q/2)) / sigma`.
        """
        sigma = self.variance.sqrt()
        a = ((value - self.mean) / sigma).abs()
        half_q = self.power.to(a) / 2.0
        log_p = (
            torch.log(half_q)
            + (half_q - 1.0) * a.log()
            - 0.5 * a.pow(2.0 * half_q)
            - 0.5 * math.log(2 * math.pi)
            - sigma.log()
        )
        return log_p.exp()


# Use BoTorch's standard normal (quasi-)MC samplers for q-ED predictives.
GetSampler.register(MultivariateQExponential)(_get_sampler_mvn)


def _select_outputs(
    qed: MultivariateQExponential, output_indices: list[int] | None, dim: int
) -> MultivariateQExponential:
    r"""Turn a batch of per-output q-ED predictives (outputs along batch dim
    `dim`) into the predictive over `output_indices`: a single-output q-ED for
    one index, else a `MultitaskMultivariateQExponential`."""
    mean, covar = qed.mean, qed.lazy_covariance_matrix
    if output_indices is not None and len(output_indices) == 1:
        index = (slice(None),) * dim + (output_indices[0],)
        return MultivariateQExponential(mean[index], covar[index], power=qed.power)
    if output_indices is not None:
        idx = torch.as_tensor(output_indices, device=mean.device)
        index = (slice(None),) * dim + (idx,)
        qed = MultivariateQExponential(mean[index], covar[index], power=qed.power)
    return MultitaskMultivariateQExponential.from_batch_qep(qed, task_dim=dim)


def _untransform_posterior(
    transform: OutcomeTransform, posterior: QEDPosterior, X: Tensor
) -> Posterior:
    r"""Untransform a `QEDPosterior`, keeping it a `QEDPosterior` for `Standardize`.

    Un-standardizing is affine, so the result is a q-ED with the same `q`. We
    reuse BoTorch's Gaussian path for the new mean and covariance. Other
    transforms return a `TransformedPosterior`.
    """
    if not isinstance(transform, Standardize):
        return transform.untransform_posterior(posterior, X=X)
    qed = posterior.distribution
    if posterior._is_mt:
        mvn = MultitaskMultivariateNormal(
            qed.mean, qed.lazy_covariance_matrix, interleaved=qed._interleaved
        )
    else:
        mvn = MultivariateNormal(qed.mean, qed.lazy_covariance_matrix)
    mvn = transform.untransform_posterior(GPyTorchPosterior(mvn), X=X).distribution
    kwargs = {"interleaved": mvn._interleaved} if posterior._is_mt else {}
    return QEDPosterior(type(qed)(mvn.mean, mvn.lazy_covariance_matrix, power=qed.power, **kwargs))


class SingleTaskQED(BatchedMultiOutputGPyTorchModel, ExactQEP, FantasizeMixin):
    r"""The q-ED surrogate of q-ED-BO.

    Works like `SingleTaskGP`, but its predictive at a point `x` is
    q-ED_1(mu_t(x), sigma_t^2(x)), where mu_t and sigma_t^2 are the GP
    posterior mean and variance. The shape parameter `q` (`power`) sets the
    tails, heavier for `q < 2`, and `q = 2` recovers the GP. Hyperparameters
    are fit by maximizing the q-ED likelihood with `QEDMarginalLogLikelihood`.
    The defaults follow `SingleTaskGP` (an RBF kernel with a dimension-scaled
    lengthscale prior, a LogNormal noise prior, and `Standardize` on the
    outcomes), except that the kernel is wrapped in a `ScaleKernel`. The
    q-ED fit rescales the kernel with `n` and `q`, and without a free
    outputscale that rescaling is forced into the lengthscale.

    When the training observations include multiple outputs, this model uses
    batching to model outputs independently.

    Example:
        >>> train_X = torch.rand(20, 2, dtype=torch.float64)
        >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
        >>> model = SingleTaskQED(train_X, train_Y, power=1.0)
        >>> mll = QEDMarginalLogLikelihood(model.likelihood, model)
        >>> fit_gpytorch_mll(mll)
    """

    train_targets: Tensor
    train_inputs: tuple[Tensor]

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        power: Tensor | float = 1.5,
        likelihood: QExponentialLikelihood | None = None,
        covar_module: Module | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: An optional `batch_shape x n x m` tensor of observed
                measurement noise. If omitted, the noise level is inferred.
            power: The shape parameter q > 0 of the q-ED. Defaults to 1.5.
            likelihood: A QPyTorch q-exponential likelihood. If omitted, one
                is built with the given power.
            covar_module: The covariance (kernel) module. If omitted, uses a
                `ScaleKernel` over an RBF kernel with a dimension-scaled
                lengthscale prior.
            mean_module: The mean function. If omitted, uses a `ConstantMean`.
            outcome_transform: An outcome transform applied to the training
                data at construction and to the posterior at inference.
                Defaults to `Standardize`. Pass `None` to disable.
            input_transform: An input transform applied in the forward pass.
        """
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        if outcome_transform == DEFAULT:
            outcome_transform = Standardize(m=train_Y.shape[-1], batch_shape=train_X.shape[:-2])
        with torch.no_grad():
            transformed_X = self.transform_inputs(X=train_X, input_transform=input_transform)
        if outcome_transform is not None:
            outcome_transform.train()
            train_Y, train_Yvar = outcome_transform(Y=train_Y, Yvar=train_Yvar, X=transformed_X)
        self._validate_tensor_args(X=transformed_X, Y=train_Y, Yvar=train_Yvar)
        validate_input_scaling(
            train_X=transformed_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            ignore_X_dims=getattr(self, "_ignore_X_dims_scaling_check", None),
            check_nans_only=covar_module is not None,
        )
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, train_Yvar = self._transform_tensor_args(
            X=train_X, Y=train_Y, Yvar=train_Yvar
        )
        self.power = torch.as_tensor(power, dtype=train_X.dtype, device=train_X.device)

        if likelihood is None:
            if train_Yvar is None:
                noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
                likelihood = QExponentialLikelihood(
                    noise_prior=noise_prior,
                    noise_constraint=GreaterThan(
                        MIN_INFERRED_NOISE_LEVEL,
                        transform=None,
                        initial_value=noise_prior.mode,
                    ),
                    batch_shape=self._aug_batch_shape,
                    power=self.power,
                )
            else:
                likelihood = FixedNoiseQExponentialLikelihood(
                    noise=train_Yvar, batch_shape=self._aug_batch_shape, power=self.power
                )
        elif not torch.equal(torch.as_tensor(likelihood.power).to(self.power), self.power):
            warnings.warn(
                f"Likelihood power ({likelihood.power}) does not match "
                f"model power ({self.power}). Using model power.",
                UserWarning,
                stacklevel=2,
            )
            likelihood.power = self.power

        ExactQEP.__init__(self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood)

        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.mean_module = mean_module

        if covar_module is None:
            covar_module = ScaleKernel(
                get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=transformed_X.shape[-1],
                    batch_shape=self._aug_batch_shape,
                ),
                batch_shape=self._aug_batch_shape,
            )
            # Used for subsetting along the output dimension. See Model.subset_output.
            self._subset_batch_dict = {
                "mean_module.raw_constant": -1,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
            if train_Yvar is None:
                self._subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2
        self.covar_module: Module = covar_module

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateQExponential:
        r"""Compute the prior over the latent function at the given inputs.

        Args:
            x: A `batch_shape x n x d` tensor of inputs.

        Returns:
            A `MultivariateQExponential` distribution.
        """
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.power)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
    ) -> Posterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                If omitted, computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `QEDPosterior` over the `q` points and the selected outputs (a
            `TransformedPosterior` for non-affine outcome transforms).
        """
        self.eval()
        X = self.transform_inputs(X)
        with gpt_posterior_settings():
            if self._num_outputs > 1:
                X, output_dim_idx = add_output_dim(
                    X=X, original_batch_shape=self._input_batch_shape
                )
            # ExactQEP.__call__ in eval mode returns the posterior, not the prior.
            qed = self._apply_noise(X=X, mvn=self(X), observation_noise=observation_noise)
            if self._num_outputs > 1:
                qed = _select_outputs(qed, output_indices, output_dim_idx)
        posterior = QEDPosterior(distribution=qed)
        if hasattr(self, "outcome_transform"):
            transform = self.outcome_transform
            if self._num_outputs > 1 and output_indices is not None:
                transform = transform.subset_output(list(output_indices))
            posterior = _untransform_posterior(transform, posterior, X)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

    def _apply_noise(
        self,
        X: Tensor,
        mvn: MultivariateQExponential,
        observation_noise: bool | Tensor = False,
    ) -> MultivariateQExponential:
        """Adds the observation noise to the posterior.

        Args:
            X: A tensor of shape `batch_shape x q x d`.
            mvn: A `MultivariateQExponential` over the latent function.
            observation_noise: If True, add the observation noise from the
                likelihood. If a Tensor, use it directly as the observation
                noise (must be of shape `(batch_shape) x q x m`).

        Returns:
            The posterior predictive.
        """
        if observation_noise is False:
            return mvn
        if torch.is_tensor(observation_noise):
            noise_shape = mvn.batch_shape + mvn.event_shape
            if self.num_outputs > 1:
                obs_noise = observation_noise.transpose(-1, -2)
            else:
                obs_noise = observation_noise.squeeze(-1)
            return self.likelihood(mvn, X, noise=obs_noise.expand(noise_shape))
        return self.likelihood(mvn, X)


# BoTorch's cached-Cholesky MC acquisitions (e.g. qLogNEI) draw joint samples
# with a Gaussian low-rank update, bypassing `QEDPosterior`. BoTorch enables
# this for every GPyTorchModel and has no opt-out hook, so disable it here.
_supports_cache_root = _cached_cholesky.supports_cache_root


def _supports_cache_root_unless_qed(model) -> bool:
    return not isinstance(model, SingleTaskQED) and _supports_cache_root(model)


_cached_cholesky.supports_cache_root = _supports_cache_root_unless_qed


@DrawMatheronPaths.register(SingleTaskQED)
def _draw_matheron_paths_qed(model: SingleTaskQED, **kwargs) -> None:
    # The q-ED is not closed under marginalization: the variance at a point of
    # a joint q-ED_n predictive is c(q, n) sigma^2, which depends on the number
    # n of points sampled together. So for q != 2 there is no sample path.
    raise NotImplementedError(
        "Pathwise (Matheron) samples are not defined for q-ED surrogates: the q-ED "
        "is not closed under marginalization, so its joint predictives are not "
        "marginally consistent. For Thompson sampling, use `MaxPosteriorSampling` "
        "on a candidate set, which draws a joint q-ED sample. For q = 2, use "
        "`SingleTaskGP`."
    )
