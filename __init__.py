"""
neuro_icu — Stochastic Discrete-Event Simulation + Constrained RL optimisation
for a Neuro-ICU, implementing the Phase 1 / Phase 2 upgrade blueprint:

  Phase 1 (simulation engine)
    - Non-Homogeneous Poisson Process (NHPP) arrivals with circadian rate
    - Gaussian-copula joint sampling of APACHE II / GCS
    - Exponential-decay health dynamics with resource-deprivation hazard
    - Weibull hazard-function VAP onset model

  Phase 2 (optimisation)
    - CMDP formulation of the resource-allocation problem
    - PPO-Lagrangian agent (numpy implementation, no external DL framework
      required) that learns capital + operational allocation policies

See README.md for usage.
"""

from .config import SimulationConfig, DISEASES

__all__ = ["SimulationConfig", "DISEASES"]
__version__ = "2.0.0"
