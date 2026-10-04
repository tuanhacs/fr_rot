from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from tst_statistics import TreePermutationStatistic
from db_tsw.db_tsw import TWConcurrentLines
from db_tsw.fr_rot import FRROTConcurrentLines
from db_tsw.rff_fr_rot import RFFFRROTConcurrentLines


def test_cached_observed_statistics_match_training_losses():
    torch.manual_seed(7)
    x, y = torch.randn(6, 4), torch.randn(6, 4)
    z = torch.cat((x, y))
    theta = torch.randn(3, 2, 4)
    theta /= theta.norm(dim=-1, keepdim=True)
    intercept = torch.randn(3, 1, 4)
    labels = torch.zeros(1, 12, dtype=torch.bool)
    labels[:, :6] = True

    cached_db = TreePermutationStatistic(z, theta, intercept, method="dbtsw", delta=2.0)(labels)[0]
    direct_db = TWConcurrentLines(p=1, delta=2.0, device="cpu")(x, y, theta, intercept)
    torch.testing.assert_close(cached_db, direct_db)

    cached_linear = TreePermutationStatistic(
        z, theta, intercept, method="fr_rot", delta=2.0, rho=0.7, fiber_tau=1.2
    )(labels)[0]
    direct_linear = FRROTConcurrentLines(
        rho=0.7, fiber_tau=1.2, delta=2.0, device="cpu"
    )(x, y, theta, intercept)
    torch.testing.assert_close(cached_linear, direct_linear)

    omega = torch.randn(3, 3, 4) / 0.8
    cached_rff = TreePermutationStatistic(
        z,
        theta,
        intercept,
        method="rff_fr_rot",
        delta=2.0,
        rho=0.7,
        fiber_tau=1.2,
        num_frequencies=3,
        rff_sigma=0.8,
        omega=omega,
    )(labels)[0]
    direct_rff = RFFFRROTConcurrentLines(
        rho=0.7,
        fiber_tau=1.2,
        num_frequencies=3,
        rff_sigma=0.8,
        delta=2.0,
        device="cpu",
    )(x, y, theta, intercept, omega=omega)
    torch.testing.assert_close(cached_rff, direct_rff)
