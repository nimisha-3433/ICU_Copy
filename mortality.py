"""
Cox-type proportional hazards mortality model (retained from v1, Sec 3.4 of
the report) — this sub-model was not part of the upgrade blueprint, so its
mathematical form is unchanged. It is kept in its own module because the new
Weibull VAP hazard and exponential health decay both feed into it.
"""

from __future__ import annotations

import numpy as np

from .config import SimulationConfig, Diagnosis


def baseline_hourly_hazard(diagnosis: Diagnosis) -> float:
    """h0 = 1 - (1 - min(r_d, 0.70))^(1 / max(48, T_d_hours))"""
    r = min(diagnosis.base_mortality_risk, 0.70)
    T_d_hours = diagnosis.min_los_days * 24.0
    return 1.0 - (1.0 - r) ** (1.0 / max(48.0, T_d_hours))


def health_hazard_multiplier(health_pct: float) -> float:
    if health_pct < 5:
        return 200.0
    if health_pct < 15:
        return 40.0
    if health_pct < 25:
        return 12.0
    if health_pct < 40:
        return 4.0
    if health_pct < 55:
        return 2.0
    if health_pct < 70:
        return 1.2
    return 1.0


def hourly_death_probability(
    *,
    diagnosis: Diagnosis,
    apache: float,
    health_pct: float,
    has_doctor: bool,
    has_nurse: bool,
    vent_required: bool,
    has_vent: bool,
    vap_active: bool,
    bed_wait_hours: float,
    cfg: SimulationConfig,
) -> float:
    h0 = baseline_hourly_hazard(diagnosis)

    m_apache = 1.0 + cfg.apache_hazard_slope * (apache - cfg.apache_hazard_ref)

    m_res = 1.0
    if not has_doctor and health_pct < 50:
        m_res += cfg.no_doctor_mult
    if not has_nurse and health_pct < 50:
        m_res += cfg.no_nurse_mult
    if vent_required and not has_vent and health_pct < 40:
        m_res += cfg.no_vent_mult
    if vap_active:
        m_res *= 1.5
    if bed_wait_hours > 48:
        m_res *= cfg.wait_gt_48_mult
    elif bed_wait_hours > 24:
        m_res *= cfg.wait_24_48_mult

    m_h = health_hazard_multiplier(health_pct)

    p = h0 * m_apache * m_res * m_h
    return float(np.clip(p, 0.0, 0.90))
