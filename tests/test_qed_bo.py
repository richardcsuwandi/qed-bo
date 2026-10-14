from copy import deepcopy

import pytest
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qKnowledgeGradient,
    qLogNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.fit import fit_gpytorch_mll
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.sampling.pathwise import draw_matheron_paths
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator import settings as linop_settings

from qed_bo import (
    QEDExpectedImprovement,
    QEDMarginalLogLikelihood,
    QEDUpperConfidenceBound,
    SingleTaskQED,
    c_q,
    qQEDUpperConfidenceBound,
)

torch.manual_seed(0)
X = torch.rand(12, 2, dtype=torch.double)
Y = torch.sin(6 * X).sum(-1, keepdim=True)
X_test = torch.rand(7, 1, 2, dtype=torch.double)
BOUNDS = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)


def fitted(power, **kwargs):
    model = SingleTaskQED(X, kwargs.pop("Y", Y), power=power, **kwargs)
    fit_gpytorch_mll(QEDMarginalLogLikelihood(model.likelihood, model))
    return model


def test_c_q():
    assert torch.allclose(c_q(2.0), torch.tensor(1.0, dtype=torch.double))
    # c(1, 1) = 4 * Gamma(5/2) / Gamma(1/2) = 3
    assert torch.allclose(c_q(1.0), torch.tensor(3.0, dtype=torch.double))


def test_acquisitions_reduce_to_gp_at_q2():
    model = SingleTaskQED(X, Y, power=2.0)
    best_f = Y.max()
    pairs = [
        (
            QEDUpperConfidenceBound(model, beta=4.0, power=2.0),
            UpperConfidenceBound(model, beta=4.0),
        ),
        (
            QEDExpectedImprovement(model, best_f=best_f, power=2.0),
            ExpectedImprovement(model, best_f=best_f),
        ),
    ]
    with torch.no_grad():
        for ours, gp in pairs:
            assert torch.allclose(ours(X_test), gp(X_test)), type(ours).__name__


@pytest.mark.parametrize("m", [1, 2])
def test_q2_matches_single_task_gp(m):
    train_Y = torch.cat([Y, -Y], dim=-1)[:, :m]
    qed = SingleTaskQED(X, train_Y, power=2.0)
    gp = SingleTaskGP(X, train_Y, covar_module=deepcopy(qed.covar_module))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    qed.load_state_dict(gp.state_dict())
    post_gp, post_qed = gp.posterior(X_test), qed.posterior(X_test)
    assert torch.allclose(post_gp.mean, post_qed.mean)
    assert torch.allclose(post_gp.variance, post_qed.variance)
    gp.train(), qed.train()
    mll_gp = ExactMarginalLogLikelihood(gp.likelihood, gp)(gp(*gp.train_inputs), gp.train_targets)
    mll_qed = QEDMarginalLogLikelihood(qed.likelihood, qed)(
        qed(*qed.train_inputs), qed.train_targets
    )
    assert torch.allclose(mll_gp, mll_qed)


def test_q_ed_fit_rescales_outputscale_not_lengthscale():
    # The q-ED MLL wants the kernel about c(q, n) times smaller than the GP's.
    # The default ScaleKernel absorbs that in the outputscale, so the lengthscale
    # stays near the q = 2 fit instead of stretching to shrink the variance.
    gp, qed = fitted(2.0), fitted(1.0)
    ratio = qed.covar_module.base_kernel.lengthscale / gp.covar_module.base_kernel.lengthscale
    assert ((ratio > 0.67) & (ratio < 1.5)).all(), ratio
    assert qed.covar_module.outputscale < gp.covar_module.outputscale / 3


@pytest.mark.parametrize("power", [2.0, 1.0, 0.5])
def test_mll_matches_qed_density(power):
    model = SingleTaskQED(X, Y, power=power)
    model.train()
    output = model(*model.train_inputs)
    ours = QEDMarginalLogLikelihood(model.likelihood, model)(output, model.train_targets)
    # QPyTorch's own q-ED density, which BoTorch's import disables.
    with linop_settings._fast_log_prob(True):
        expected = model.likelihood(output).log_prob(model.train_targets)
    log_prior = sum(p.log_prob(c(m)).sum() for _, m, p, c, _ in model.named_priors())
    assert torch.allclose(ours, (expected + log_prior) / X.shape[0])


@pytest.mark.parametrize("power", [1.0, 0.5])
def test_predictive_samples_quantiles_and_density(power):
    model = SingleTaskQED(X, Y, power=power)
    post = model.posterior(torch.rand(1, 1, 2, dtype=torch.double))
    mean, scale2 = post.mean.squeeze(), post.variance.squeeze()
    samples = SobolQMCNormalSampler(torch.Size([2**16]), seed=0)(post).squeeze()
    # The variance of the q-ED_1 predictive is c(q) * sigma^2.
    assert torch.allclose(samples.var() / scale2, c_q(power).to(samples), rtol=0.05)
    p = torch.tensor(0.9, dtype=torch.double)
    assert torch.allclose((samples <= post.quantile(p).squeeze()).double().mean(), p, atol=0.01)
    # The density equals QPyTorch's q-ED density at n = 1.
    y = mean + 0.7 * scale2.sqrt()
    with linop_settings._fast_log_prob(True):
        expected = post.distribution.log_prob(y.reshape(1, 1)).exp()
    assert torch.allclose(post.density(y.reshape(1, 1, 1)).squeeze(), expected.squeeze())


def test_fixed_noise_and_multi_output():
    model = fitted(1.0, train_Yvar=torch.full_like(Y, 0.01))
    assert type(model.likelihood).__name__ == "FixedNoiseQExponentialLikelihood"
    model = fitted(1.0, Y=torch.cat([Y, -Y], dim=-1))
    post = model.posterior(torch.rand(5, 3, 2, dtype=torch.double))
    assert post.mean.shape == (5, 3, 2)
    assert post.rsample(torch.Size([4])).shape == (4, 5, 3, 2)
    for idx in ([1], [1, 0]):
        post, sub = (
            model.posterior(X_test, output_indices=idx),
            model.subset_output(idx).posterior(X_test),
        )
        assert torch.allclose(post.mean, sub.mean) and torch.allclose(post.variance, sub.variance)


def test_condition_on_observations_is_exact():
    model = SingleTaskQED(X, Y, power=1.0, outcome_transform=None)
    X_new, Y_new = torch.rand(3, 2, dtype=torch.double), torch.rand(3, 1, dtype=torch.double)
    model.posterior(X_new)  # conditioning needs a cached prediction strategy
    conditioned = model.condition_on_observations(X_new, Y_new)
    refit = SingleTaskQED(
        torch.cat([X, X_new]), torch.cat([Y, Y_new]), power=1.0, outcome_transform=None
    )
    refit.load_state_dict(model.state_dict())
    a, b = conditioned.posterior(X_test), refit.posterior(X_test)
    assert torch.allclose(a.mean, b.mean) and torch.allclose(a.variance, b.variance)


@pytest.mark.parametrize("power", [1.0, 0.5])
def test_batch_acquisitions_match_analytic_at_one_point(power):
    model = SingleTaskQED(X, Y, power=power)
    sampler = SobolQMCNormalSampler(torch.Size([2**16]), seed=0)
    mc = qQEDUpperConfidenceBound(model, beta=4.0, power=power, sampler=sampler)(X_test)
    exact = QEDUpperConfidenceBound(model, beta=4.0, power=power)(X_test)
    assert torch.allclose(mc, exact, rtol=0.005)
    mc = qExpectedImprovement(model, best_f=Y.max(), sampler=sampler)(X_test)
    exact = QEDExpectedImprovement(model, best_f=Y.max(), power=power)(X_test)
    assert torch.allclose(mc, exact, rtol=0.01, atol=1e-4)


def test_batch_qucb_is_botorch_qucb_at_q2():
    model = SingleTaskQED(X, Y, power=2.0)
    X_batch = torch.rand(4, 3, 2, dtype=torch.double)
    ours = qQEDUpperConfidenceBound(
        model, beta=4.0, power=2.0, sampler=SobolQMCNormalSampler(torch.Size([256]), seed=1)
    )
    gp = qUpperConfidenceBound(
        model, beta=4.0, sampler=SobolQMCNormalSampler(torch.Size([256]), seed=1)
    )
    assert torch.allclose(ours(X_batch), gp(X_batch))


def test_bo_step():
    model = fitted(1.0)
    acq = QEDUpperConfidenceBound(model, beta=4.0, power=1.0)
    x, _ = optimize_acqf(acq, bounds=BOUNDS, q=1, num_restarts=2, raw_samples=32)
    assert x.shape == (1, 2) and ((x >= 0) & (x <= 1)).all()
    # MC acquisitions work, without BoTorch's Gaussian-only cached sampling.
    acq = qLogNoisyExpectedImprovement(model, X_baseline=X)
    assert not acq._cache_root
    x, _ = optimize_acqf(acq, bounds=BOUNDS, q=3, num_restarts=2, raw_samples=32)
    assert x.shape == (3, 2)
    acq = qQEDUpperConfidenceBound(model, beta=4.0, power=1.0)
    x, _ = optimize_acqf(acq, bounds=BOUNDS, q=3, num_restarts=2, raw_samples=32)
    assert x.shape == (3, 2)
    acq = qKnowledgeGradient(model, num_fantasies=8)
    x, _ = optimize_acqf(acq, bounds=BOUNDS, q=2, num_restarts=2, raw_samples=16)
    assert x.shape == (2, 2)
    # Thompson sampling on a candidate set works, pathwise samples are undefined.
    x = MaxPosteriorSampling(model, replacement=False)(
        torch.rand(100, 2, dtype=torch.double), num_samples=3
    )
    assert x.shape == (3, 2)
    with pytest.raises(NotImplementedError, match="not marginally consistent"):
        draw_matheron_paths(model, torch.Size([2]))
