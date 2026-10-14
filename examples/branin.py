"""q-ED-BO with q-UCB on Branin with Cauchy-corrupted observations.

python examples/branin.py
"""

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin
from torch.quasirandom import SobolEngine

from qed_bo import (
    QEDMarginalLogLikelihood,
    QEDUpperConfidenceBound,
    SingleTaskQED,
    ucb_beta,
)

torch.manual_seed(0)
POWER = 1.0  # the shape q; q < 2 gives heavier tails than a GP
N_INIT, N_ITER, OUTLIER_PROB = 5, 30, 0.1

branin = Branin(negate=True)
bounds = torch.stack([torch.zeros(2), torch.ones(2)]).double()


def f(x01):
    """Noisy observation: Branin on the unit square plus occasional Cauchy spikes."""
    y = branin(branin.bounds[0] + x01 * (branin.bounds[1] - branin.bounds[0]))
    spikes = torch.distributions.Cauchy(0.0, 10.0).sample(y.shape).double()
    return (y + spikes * (torch.rand_like(y) < OUTLIER_PROB)).unsqueeze(-1)


def f_true(x01):
    return branin(branin.bounds[0] + x01 * (branin.bounds[1] - branin.bounds[0]))


train_x = SobolEngine(2, scramble=True, seed=0).draw(N_INIT).double()
train_y = f(train_x)

for t in range(1, N_ITER + 1):
    model = SingleTaskQED(train_x, train_y, power=POWER)
    fit_gpytorch_mll(QEDMarginalLogLikelihood(model.likelihood, model))
    acq = QEDUpperConfidenceBound(model, beta=ucb_beta(t, num_candidates=2000), power=POWER)
    x_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
    train_x = torch.cat([train_x, x_next])
    train_y = torch.cat([train_y, f(x_next)])
    best = f_true(train_x).max().item()
    print(f"iter {t:2d}  best true value {best:.4f}  (optimum {branin.optimal_value:.4f})")
