"""
Phase 1 stochastic sub-models.

This module implements the three "upgrade" distributions described in the
blueprint, each replacing a deterministic / uniform placeholder in the
original engine with a mathematically-grounded stochastic model:

  1. arrival_rate(t) + sample_arrivals()   — Non-Homogeneous Poisson Process
  2. sample_apache_gcs()                    — Gaussian-copula joint severity
  3. WeibullVapHazard                       — Weibull survival hazard for VAP

Every function accepts a numpy.random.Generator so simulation runs are fully
reproducible given a seed.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .config import SimulationConfig


# ---------------------------------------------------------------------------
# 1. Non-Homogeneous Poisson Process arrivals
# ---------------------------------------------------------------------------

def arrival_rate_per_hour(t_hours: float, cfg: SimulationConfig) -> float:
    """
    lambda(t): sinusoidal circadian arrival-rate function.

        lambda(t) = (lambda_base / 24) * (1 + A * sin(2*pi*(h - peak)/24))

    where h = hour-of-day. The amplitude A is clamped to [0, 0.95] so the
    rate never goes negative. This produces the trauma-spike-at-night
    behaviour described in the blueprint while preserving the same daily
    mean arrival rate as the original homogeneous model (so lambda_base is
    still directly comparable to the v1 "admissions/day" parameter).
    """
    A = float(np.clip(cfg.circadian_amplitude, 0.0, 0.95))
    hour_of_day = t_hours % 24.0
    phase = 2.0 * np.pi * (hour_of_day - cfg.circadian_peak_hour) / 24.0
    base = cfg.lambda_base_per_day / 24.0
    return max(0.0, base * (1.0 + A * np.sin(phase)))


def sample_arrivals(t_hours: float, cfg: SimulationConfig, rng: np.random.Generator) -> int:
    """
    Draw the number of new admissions in the hour [t, t+1).

    N_arrivals ~ Poisson(lambda(t))

    Unlike the v1 model there is *no* refractory gate: setting
    cfg.allow_mass_casualty = True permits N > 1 in a single hour, modelling
    simultaneous mass-casualty admissions (e.g. a multi-vehicle trauma
    event). Setting it False reproduces single-admission-per-hour behaviour
    (N capped at 1) for comparison against the baseline engine.
    """
    lam = arrival_rate_per_hour(t_hours, cfg)
    n = rng.poisson(lam)
    if not cfg.allow_mass_casualty:
        n = min(n, 1)
    return int(n)


# ---------------------------------------------------------------------------
# 2. Gaussian-copula joint APACHE II / GCS sampling
# ---------------------------------------------------------------------------

def sample_apache_gcs(cfg: SimulationConfig, rng: np.random.Generator, n: int = 1):
    """
    Generate n correlated (APACHE II, GCS) pairs via a Gaussian copula.

    Steps (exactly as specified in the blueprint):
      1. R = [[1, rho], [rho, 1]]
      2. (Z_A, Z_G) ~ N(0, R)
      3. U_A, U_G = Phi(Z_A), Phi(Z_G)                (uniform marginals)
      4. APACHE = clip(Normal(apache_mean, apache_sd).ppf(U_A))
         GCS     = scale Beta(a, b).ppf(U_G) onto [gcs_min, gcs_max]

    This removes the deterministic APACHE->GCS clamp formula used in v1 and
    replaces it with a proper joint distribution: strongly-negatively
    correlated but each with realistic marginal shape (Normal for APACHE,
    right-skewed Beta for GCS, since most Neuro-ICU admissions cluster at
    higher GCS with a heavy low-GCS tail).
    """
    rho = float(np.clip(cfg.copula_rho, -0.999, 0.999))
    R = np.array([[1.0, rho], [rho, 1.0]])
    Z = rng.multivariate_normal(mean=[0.0, 0.0], cov=R, size=n)
    U_A = stats.norm.cdf(Z[:, 0])
    U_G = stats.norm.cdf(Z[:, 1])

    apache_raw = stats.norm.ppf(U_A, loc=cfg.apache_mean, scale=cfg.apache_sd)
    apache = np.clip(np.round(apache_raw), cfg.apache_min, cfg.apache_max)

    gcs_unit = stats.beta.ppf(U_G, cfg.gcs_beta_a, cfg.gcs_beta_b)
    gcs_raw = cfg.gcs_min + gcs_unit * (cfg.gcs_max - cfg.gcs_min)
    gcs = np.clip(np.round(gcs_raw), cfg.gcs_min, cfg.gcs_max)

    if n == 1:
        return float(apache[0]), float(gcs[0])
    return apache, gcs


def sample_initial_health(apache: float, cfg: SimulationConfig, rng: np.random.Generator) -> float:
    """H0 = clip(75 - 0.6*(APACHE-15) + eps, 40, 85), eps ~ Uniform(-6,6). (unchanged from v1)"""
    eps = rng.uniform(-6.0, 6.0)
    h0 = 75.0 - 0.6 * (apache - 15.0) + eps
    return float(np.clip(h0, 40.0, 85.0))


# ---------------------------------------------------------------------------
# 3. Weibull hazard-function VAP onset
# ---------------------------------------------------------------------------

class WeibullVapHazard:
    """
    Ventilator-Associated Pneumonia onset governed by a Weibull hazard:

        h(t) = (beta/eta) * (t/eta)^(beta-1) * exp(gamma * APACHE)

    where t is hours-since-intubation. beta > 1 gives an *increasing*
    hazard the longer a patient stays intubated (consistent with clinical
    VAP epidemiology), and the exp(gamma*APACHE) term amplifies risk for
    sicker patients. This replaces the v1 model (uniform 48-96h timer with
    a flat 20% draw at expiry).

    The per-hour onset probability implied by the hazard is
        P(onset in [t, t+1) | survived to t) = 1 - exp(-h(t) * dt)
    """

    def __init__(self, cfg: SimulationConfig):
        self.beta = cfg.vap_weibull_shape
        self.eta = cfg.vap_weibull_scale_hours
        self.gamma = cfg.vap_apache_gamma

    def hazard(self, t_intubated_hours: float, apache: float) -> float:
        t = max(t_intubated_hours, 1e-6)
        base = (self.beta / self.eta) * (t / self.eta) ** (self.beta - 1.0)
        return float(base * np.exp(self.gamma * apache))

    def onset_probability(self, t_intubated_hours: float, apache: float, dt_hours: float = 1.0) -> float:
        h = self.hazard(t_intubated_hours, apache)
        p = 1.0 - np.exp(-h * dt_hours)
        return float(np.clip(p, 0.0, 1.0))

    def sample_onset(self, t_intubated_hours: float, apache: float, rng: np.random.Generator,
                      dt_hours: float = 1.0) -> bool:
        return bool(rng.random() < self.onset_probability(t_intubated_hours, apache, dt_hours))
