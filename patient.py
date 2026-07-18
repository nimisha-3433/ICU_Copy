"""
Patient entity.

Implements Phase 1.3 of the blueprint: health now evolves via a *non-linear
exponential decay* rather than the v1 discretised linear ODE:

    H(t+1) = H(t) * exp(-k(t)) + Delta_intervention(t)

k(t) is small (near-zero net decay) while the patient has the resources
they need, and grows sharply — driving a fast "crash" rather than a slow
linear slide — when a required resource (doctor, nurse, ventilator) is
unavailable, scaled further by illness severity (APACHE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import Diagnosis, SimulationConfig
from .distributions import sample_apache_gcs, sample_initial_health, WeibullVapHazard
from . import mortality as mort


@dataclass
class JourneyEvent:
    time_days: float
    event: str
    detail: str = ""


class Patient:
    """A single Neuro-ICU patient tracked through the DES."""

    _counter = 0

    def __init__(self, admit_time_hours: float, diagnosis: Diagnosis,
                 cfg: SimulationConfig, rng: np.random.Generator, initial: bool = False):
        Patient._counter += 1
        self.id = f"PT-{Patient._counter:05d}"
        self.admit_time = admit_time_hours
        self.diagnosis = diagnosis
        self.cfg = cfg
        self.rng = rng

        self.apache, self.gcs = sample_apache_gcs(cfg, rng, n=1)
        self.health = sample_initial_health(self.apache, cfg, rng)
        if initial:
            self.health = float(rng.uniform(55, 80))

        self.vent_required = bool(rng.random() < diagnosis.vent_prob)
        self.icp_required = bool(rng.random() < diagnosis.icp_prob)

        self.vap = False
        self.vap_hazard = WeibullVapHazard(cfg)
        self.hours_intubated = 0.0
        self._vap_mortality_active = False

        # FSM states: "waiting_bed" -> "occupying" (has a bed; health dynamics and
        # mortality apply continuously regardless of staff assignment, per the
        # exponential-decay/deprivation-hazard model) -> "discharged" | "dead".
        # `critical` is a display sub-flag (health < 30), not a separate FSM state,
        # so a staff-deprived patient still evolves instead of freezing in queue.
        self.status = "occupying" if initial else "waiting_bed"
        self.critical = False
        self.bed_id: Optional[int] = None
        self.doctor_id: Optional[str] = None
        self.nurse_id: Optional[str] = None
        self.has_vent = False
        self.has_icp = False

        self.bed_wait_hours = 0.0
        self.resource_wait_hours = 0.0
        self.stable_hours = 0.0
        self.discharge_time: Optional[float] = None
        self.outcome: Optional[str] = None  # "discharged" | "dead"

        self.cost_bed = self.cost_vent = self.cost_icp = 0.0
        self.cost_med = self.cost_lab = self.cost_vap = self.cost_wait = 0.0
        self.total_cost = 0.0
        self._last_cost_day = -1

        self.journey: List[JourneyEvent] = [JourneyEvent(
            admit_time_hours / 24.0,
            "Initial admission" if initial else "Arrival",
            f"{diagnosis.name} | APACHE {self.apache:.0f} | GCS {self.gcs:.0f} | "
            f"Health {self.health:.0f}%",
        )]

    # ------------------------------------------------------------------
    def los_days(self, now_hours: float) -> float:
        end = self.discharge_time if self.discharge_time is not None else now_hours
        return (end - self.admit_time) / 24.0

    def log(self, now_hours: float, event: str, detail: str = ""):
        self.journey.append(JourneyEvent(now_hours / 24.0, event, detail))

    # ------------------------------------------------------------------
    def accrue_costs(self, now_hours: float):
        """Daily cost accrual (fires once per simulated day, matching v1)."""
        day = int(now_hours // 24)
        if day == self._last_cost_day:
            return
        self._last_cost_day = day

        self.cost_bed += self.cfg.cost_bed_day
        if self.has_vent:
            self.cost_vent += self.cfg.cost_vent_day
        if self.has_icp:
            self.cost_icp += self.cfg.cost_icp_day
        if self.vap:
            self.cost_vap += self.cfg.cost_vap_day
        self.cost_med += self.cfg.cost_med_day_avg
        self.cost_lab += self.cfg.cost_lab_day_avg

        self.total_cost = (self.cost_bed + self.cost_vent + self.cost_icp +
                            self.cost_vap + self.cost_med + self.cost_lab + self.cost_wait)

    def accrue_wait_cost(self):
        self.cost_wait += self.cfg.cost_wait_hour
        self.total_cost += self.cfg.cost_wait_hour

    # ------------------------------------------------------------------
    def decay_rate_k(self) -> float:
        """
        k(t): exponential decay-rate constant for this hour.

        Baseline is tiny (near-conservation of health absent intervention);
        it grows substantially when a *required* resource is missing, with
        the size of the jump scaled by APACHE severity — sicker patients
        crash faster when deprived.
        """
        cfg = self.cfg
        k = cfg.health_k_baseline
        severity_scale = 1.0 + cfg.health_k_apache_scale * max(0.0, self.apache - cfg.apache_hazard_ref)

        deprived = False
        if self.doctor_id is None and self.health < 50:
            deprived = True
        if self.nurse_id is None and self.health < 50:
            deprived = True
        if self.vent_required and not self.has_vent and self.health < 40:
            deprived = True

        if deprived:
            k += cfg.health_k_deprivation * severity_scale
        return k

    def intervention_delta(self) -> float:
        cfg = self.cfg
        delta = self.diagnosis.recovery_rate_per_day / 24.0
        if self.doctor_id is not None:
            delta += cfg.doctor_bonus
        if self.nurse_id is not None:
            delta += cfg.nurse_bonus
        if self.has_icp:
            delta += cfg.icp_bonus
        if self.vap:
            delta -= cfg.vap_penalty
        return delta

    def step_health(self):
        """H(t+1) = clip(H(t) * exp(-k(t)) + Delta_intervention(t), 0, 100)"""
        k = self.decay_rate_k()
        delta = self.intervention_delta()
        h_new = self.health * np.exp(-k) + delta
        self.health = float(np.clip(h_new, 0.0, 100.0))

    def step_waiting_decay(self):
        self.health = float(max(0.0, self.health - self.cfg.waiting_decay_per_hour))

    # ------------------------------------------------------------------
    def step_vap(self, now_hours: float) -> bool:
        """Weibull-hazard VAP onset check for ventilated, non-VAP patients."""
        if not (self.has_vent and not self.vap):
            return False
        self.hours_intubated += 1.0
        onset = self.vap_hazard.sample_onset(self.hours_intubated, self.apache, self.rng)
        if onset:
            self.vap = True
            self._vap_mortality_active = True
            self.health = float(max(20.0, self.health - self.cfg.vap_health_penalty))
            self.log(now_hours, "VAP onset", f"Weibull hazard triggered; health -> {self.health:.0f}%")
            return True
        return False

    # ------------------------------------------------------------------
    def hourly_death_probability(self) -> float:
        return mort.hourly_death_probability(
            diagnosis=self.diagnosis,
            apache=self.apache,
            health_pct=self.health,
            has_doctor=self.doctor_id is not None,
            has_nurse=self.nurse_id is not None,
            vent_required=self.vent_required,
            has_vent=self.has_vent,
            vap_active=self.vap,
            bed_wait_hours=self.bed_wait_hours,
            cfg=self.cfg,
        )

    def min_los_days(self) -> float:
        from .config import SEVERE_BRAIN_INJURY
        return 10.0 if self.diagnosis.name in SEVERE_BRAIN_INJURY else 5.0
