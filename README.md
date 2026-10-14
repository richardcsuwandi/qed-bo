# q-ED-BO: Robust Bayesian Optimization with Q-Exponential Surrogates

[![arXiv](https://img.shields.io/badge/arXiv-2609.32775-b31b1b.svg)](https://arxiv.org/abs/2609.32775)
[![Tests](https://github.com/richardcsuwandi/qed-bo/actions/workflows/tests.yml/badge.svg)](https://github.com/richardcsuwandi/qed-bo/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of [*Robust Bayesian Optimization with Q-Exponential
Surrogates*](https://arxiv.org/abs/2609.32775) ([project page](https://richardcsuwandi.github.io/papers/qed-bo/)).

Gaussian process surrogates are sensitive to outliers and heavy-tailed noise.
q-ED-BO keeps the GP's closed-form posterior mean $\mu_t(x)$ and variance
$\sigma_t^2(x)$, and replaces the Gaussian predictive with a univariate
$q$-exponential distribution,

$$f(x) \mid \mathcal{D}_t \sim q\text{-ED}_1\bigl(\mu_t(x), \sigma_t^2(x)\bigr),$$

whose shape parameter $q$ sets the tails. It recovers the GP at $q = 2$ and
grows heavier-tailed as $q$ decreases. This gives two closed-form
acquisitions:

- **q-UCB**, $\mu_t(x) + \Lambda_t^{1/q}\sigma_t(x)$, with regret
  $R_T = \tilde{O}\bigl((\log T)^{1/q}\sqrt{T\gamma_T}\bigr)$
- **q-EI**, the exact expected improvement under the q-ED predictive

The package is built on PyTorch, [BoTorch](https://botorch.org),
[GPyTorch](https://gpytorch.ai) and [QPyTorch](https://github.com/lanzithinking/qepytorch).

## Install

```bash
pip install git+https://github.com/richardcsuwandi/qed-bo.git
```

or, for development,

```bash
git clone https://github.com/richardcsuwandi/qed-bo.git
cd qed-bo
pip install -e ".[dev]"
```

## Usage

`SingleTaskQED` is a drop-in replacement for BoTorch's `SingleTaskGP`, so the
usual BoTorch loop works unchanged. The shape $q$ is the `power` argument.

```python
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from qed_bo import QEDMarginalLogLikelihood, QEDUpperConfidenceBound, SingleTaskQED, ucb_beta

train_X = torch.rand(10, 2, dtype=torch.double)
train_Y = torch.sin(6 * train_X).sum(-1, keepdim=True)

model = SingleTaskQED(train_X, train_Y, power=1.0)
fit_gpytorch_mll(QEDMarginalLogLikelihood(model.likelihood, model))

acq = QEDUpperConfidenceBound(model, beta=ucb_beta(t=1, num_candidates=2000), power=1.0)
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
candidate, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
```

For q-EI, use `QEDExpectedImprovement(model, best_f=train_Y.max(), power=1.0)`.
See [examples/branin.py](examples/branin.py) for a full loop on Branin with
Cauchy-corrupted observations.

Always fit with `QEDMarginalLogLikelihood`. Importing BoTorch makes QPyTorch's
own `ExactMarginalLogLikelihood` fall back to the Gaussian likelihood for every
$q$.

### Beyond the paper

The paper covers sequential BO, which only needs the univariate predictive.
For several points at once, the package uses the $n$-dimensional
q-ED$_n$ with the GP posterior covariance. This supports
`qQEDUpperConfidenceBound` (a batch q-UCB), BoTorch's Monte Carlo acquisitions
such as `qLogNoisyExpectedImprovement` and `qKnowledgeGradient`, and Thompson
sampling with BoTorch's `MaxPosteriorSampling` on a candidate set.

Pathwise (Matheron) sampling is not available. The q-ED is not closed under
marginalization, so for $q \neq 2$ the joint predictives have no consistent
sample path.

## API

| Name | Description |
| --- | --- |
| `SingleTaskQED` | The q-ED surrogate, with `SingleTaskGP`'s defaults. Supports inferred or fixed (`train_Yvar`) noise, multiple outputs, and input and outcome transforms |
| `QEDPosterior` | The q-ED predictive as a BoTorch posterior, with sampling, quantiles and densities |
| `QEDMarginalLogLikelihood` | The q-ED marginal likelihood, fit with `fit_gpytorch_mll` |
| `QEDUpperConfidenceBound` | q-UCB, $\mu_t(x) + \Lambda_t^{1/q}\sigma_t(x)$ |
| `QEDExpectedImprovement` | q-EI, the exact expected improvement under the q-ED predictive |
| `qQEDUpperConfidenceBound` | Monte Carlo q-UCB for a batch of points |
| `c_q` | Variance-scaling constant $c(q, n)$ |
| `ucb_beta` | Confidence schedule $\Lambda_t$ |

## Tests

```bash
pytest
ruff check . && ruff format --check .
```

## Citation

```bibtex
@misc{suwandi2026qed,
  title={Robust Bayesian Optimization with Q-Exponential Surrogates},
  author={Suwandi, Richard Cornelius and Lin, Zhidi and Yin, Feng and Zoubir, Abdelhak M.},
  year={2026},
  eprint={2609.32775},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## License

MIT, see [LICENSE](LICENSE). Parts of `qed_bo/models.py` are adapted from
[BoTorch](https://github.com/pytorch/botorch) (MIT, Copyright (c) Meta
Platforms, Inc. and affiliates).
