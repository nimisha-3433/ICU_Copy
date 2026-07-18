"""
Central configuration for the Neuro-ICU stochastic simulation.

All tunable constants live here so that the simulation engine, the RL
environment, and the analysis scripts share a single source of truth.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Disease catalogue (unchanged clinical structure from the v1 engine; the
# baseline mortality risk r_d and reference window T_d feed the Cox-type
# hazard model in mortality.py).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Diagnosis:
    name: str
    recovery_rate_per_day: float   # baseline recovery contribution to health/day
    vent_prob: float                # P(mechanical ventilation required)
    icp_prob: float                 # P(ICP monitor required)
    base_mortality_risk: float      # r_d : cumulative mortality risk over T_d hours
    min_los_days: float
    max_los_days: float


DISEASES: List[Diagnosis] = [
    Diagnosis("Severe TBI",           2.2, 0.70, 0.75, 0.18, 14, 35),
    Diagnosis("SAH",                  2.3, 0.68, 0.80, 0.15, 12, 30),
    Diagnosis("ICH",                  2.1, 0.65, 0.75, 0.20, 10, 28),
    Diagnosis("Brain Tumor Surgery",  3.8, 0.35, 0.40, 0.05, 5, 14),
    Diagnosis("Ischemic Stroke",      3.2, 0.38, 0.30, 0.10, 7, 20),
    Diagnosis("Status Epilepticus",   3.5, 0.35, 0.25, 0.06, 4, 14),
    Diagnosis("Cervical SCI",         2.0, 0.75, 0.12, 0.10, 14, 35),
    Diagnosis("Myasthenia Gravis",    3.6, 0.55, 0.06, 0.04, 5, 14),
    Diagnosis("GBS",                  2.1, 0.65, 0.05, 0.06, 14, 35),
    Diagnosis("Meningitis",           3.0, 0.45, 0.45, 0.12, 7, 21),
]

SEVERE_BRAIN_INJURY = {"Severe TBI", "SAH", "ICH"}


@dataclass
class SimulationConfig:
    # ---- Resource pool sizes (baseline / "standard care") -----------------
    beds: int = 18
    doctors: int = 8
    nurses: int = 20
    ventilators: int = 14
    icp_monitors: int = 12

    # ---- Time -------------------------------------------------------------
    dt_hours: float = 1.0
    horizon_days: float = 60.0          # simulated horizon per run/episode

    # ---- Phase 1.1 — NHPP arrivals (circadian rhythm) ---------------------
    lambda_base_per_day: float = 2.8    # mean admissions / day
    circadian_amplitude: float = 0.45   # relative amplitude of sinusoid, in [0,1)
    circadian_peak_hour: float = 2.0    # hour-of-day (0-23) of peak trauma arrivals
    allow_mass_casualty: bool = True    # if True, N_arrivals ~ Poisson(lambda(t)) per hour
                                         # (can exceed 1 arrival/hour); if False, capped at 1

    # ---- Phase 1.2 — Gaussian-copula severity ------------------------------
    copula_rho: float = -0.72           # correlation between APACHE and GCS latent normals
    apache_mean: float = 24.0
    apache_sd: float = 7.0
    apache_min: float = 8.0
    apache_max: float = 45.0
    gcs_min: float = 3.0
    gcs_max: float = 15.0
    gcs_beta_a: float = 3.2             # Beta(a,b) shape params, scaled onto [gcs_min,gcs_max]
    gcs_beta_b: float = 1.7             # (right-skewed -> most patients cluster near higher GCS)

    # ---- Phase 1.3 — exponential-decay health dynamics ---------------------
    health_k_baseline: float = 0.00035    # per-hour baseline decay-rate constant
    health_k_deprivation: float = 0.028   # additional decay-rate when a required resource is absent
    health_k_apache_scale: float = 0.0006 # additional decay-rate per APACHE point above 18
    doctor_bonus: float = 0.020
    nurse_bonus: float = 0.015
    icp_bonus: float = 0.020
    vap_penalty: float = 0.060
    waiting_decay_per_hour: float = 0.08

    # ---- Phase 1.4 — Weibull hazard VAP model ------------------------------
    # Calibrated so that P(VAP by T hours on the vent) = 1 - exp(-exp(gamma*APACHE)*(T/eta)^beta)
    # lands in the clinically-reported 9-27% range [ATS/IDSA 2005] over a typical
    # 4-10 day ventilation window, with a mean-APACHE (~24) patient.
    vap_weibull_shape: float = 1.5        # beta : >1 => hazard increases with time intubated
    vap_weibull_scale_hours: float = 480. # eta  : characteristic time scale (~20 days)
    vap_apache_gamma: float = 0.015       # gamma: APACHE severity amplification of hazard
    vap_health_penalty: float = 10.0
    vap_mortality_multiplier: float = 1.5

    # ---- Mortality model (Cox-type proportional hazards; retained from v1) -
    apache_hazard_slope: float = 0.025
    apache_hazard_ref: float = 18.0
    no_doctor_mult: float = 0.30
    no_nurse_mult: float = 0.20
    no_vent_mult: float = 0.40
    resource_health_thresh: float = 0.40
    wait_gt_48_mult: float = 1.4
    wait_24_48_mult: float = 1.2

    # ---- Discharge (dual-gate) ---------------------------------------------
    discharge_health_threshold: float = 68.0
    stability_window_hours: float = 48.0

    # ---- Cost accrual (INR) -------------------------------------------------
    cost_bed_day: float = 6000.0
    cost_vent_day: float = 2500.0
    cost_icp_day: float = 1500.0
    cost_vap_day: float = 5000.0
    cost_med_day_avg: float = 2500.0     # simplified aggregate medication cost/day
    cost_lab_day_avg: float = 900.0      # simplified aggregate lab cost/day
    cost_wait_hour: float = 500.0 / 24   # opportunity cost while waiting

    # ---- Capital expenditure (one-time, INR) — used by the RL/CMDP layer ---
    capex_bed: float = 1_500_000.0
    capex_doctor: float = 500_000.0      # onboarding + annualised recruitment cost proxy
    capex_nurse: float = 200_000.0
    capex_vent: float = 800_000.0
    capex_icp: float = 300_000.0

    # ---- Triage --------------------------------------------------------------
    triage_policy: str = "severity"      # "severity" | "survival"

    # ---- Safety constraint (CMDP) --------------------------------------------
    mortality_cap: float = 0.132         # historical baseline cohort mortality (13.2%)

    def resource_dict(self) -> Dict[str, int]:
        return dict(
            beds=self.beds,
            doctors=self.doctors,
            nurses=self.nurses,
            ventilators=self.ventilators,
            icp_monitors=self.icp_monitors,
        )
