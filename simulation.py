"""
Time-stepped Discrete-Event Simulation engine for the Neuro-ICU (Phase 1).

Each call to .step() advances the clock by cfg.dt_hours and executes, in
order (same five-stage loop as the v1 engine, Sec 3.1):

    1. Patient arrival evaluation      (NHPP, Sec Phase 1.1)
    2. Triage and bed assignment       (multi-criteria priority queueing)
    3. Staff / resource allocation
    4. Health dynamics update          (exponential decay, Sec Phase 1.3)
    5. Discharge eligibility check     (dual-gate criterion)

VAP onset (Weibull hazard, Phase 1.4) and mortality (Cox-type hazard) are
evaluated as part of stage 4.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import SimulationConfig, DISEASES, SEVERE_BRAIN_INJURY
from .distributions import sample_arrivals
from .patient import Patient


def _priority_score(p: Patient, policy: str) -> float:
    """Eq. 3.16 / 3.17 — multi-criteria severity score, or pure survival (health)."""
    if policy == "survival":
        return p.health
    score = 1.2 * p.apache + 1.5 * (15 - p.gcs)
    if p.health < 35:
        score += 30
    if p.vent_required:
        score += 20
    if p.icp_required:
        score += 15
    if p.bed_wait_hours > 24:
        score += 15
    return score


class ResourcePool:
    """Simple counted resource pools for beds / doctors / nurses / vents / ICP monitors."""

    def __init__(self, cfg: SimulationConfig):
        self.beds_total = cfg.beds
        self.doctors_total = cfg.doctors
        self.nurses_total = cfg.nurses
        self.vents_total = cfg.ventilators
        self.icp_total = cfg.icp_monitors

        self.bed_occupancy: List[Optional[str]] = [None] * cfg.beds
        self.doctors_avail = cfg.doctors
        self.nurses_avail = cfg.nurses
        self.vents_avail = cfg.ventilators
        self.icp_avail = cfg.icp_monitors

    def resize(self, cfg: SimulationConfig):
        """
        Grow/shrink pools to match cfg (used by the RL capital-action layer).

        Beds can only be *removed* if a free (unoccupied) bed exists to remove —
        you cannot decommission a bed with a patient in it. If fewer free beds
        exist than requested, cfg.beds is corrected back up to what was actually
        achievable so cfg and the live pool never drift out of sync (which would
        otherwise corrupt bed_id indices for currently-admitted patients).
        """
        if cfg.beds > self.beds_total:
            self.bed_occupancy += [None] * (cfg.beds - self.beds_total)
        elif cfg.beds < self.beds_total:
            free_idx = [i for i, b in enumerate(self.bed_occupancy) if b is None]
            n_remove = min(self.beds_total - cfg.beds, len(free_idx))
            for i in sorted(free_idx[:n_remove], reverse=True):
                del self.bed_occupancy[i]
            cfg.beds = len(self.bed_occupancy)  # snap back to what was achievable

        def _resize_pool(avail, total, target):
            """A resource can only be decommissioned if it's currently free; if the
            requested cut exceeds what's free, shrink by as much as is achievable
            and snap `target` back so cfg stays consistent with the live pool."""
            delta = target - total
            if delta >= 0:
                return avail + delta, total + delta, target
            removable = min(-delta, avail)
            new_avail = avail - removable
            new_total = total - removable
            return new_avail, new_total, new_total

        self.doctors_avail, self.doctors_total, cfg.doctors = _resize_pool(
            self.doctors_avail, self.doctors_total, cfg.doctors)
        self.nurses_avail, self.nurses_total, cfg.nurses = _resize_pool(
            self.nurses_avail, self.nurses_total, cfg.nurses)
        self.vents_avail, self.vents_total, cfg.ventilators = _resize_pool(
            self.vents_avail, self.vents_total, cfg.ventilators)
        self.icp_avail, self.icp_total, cfg.icp_monitors = _resize_pool(
            self.icp_avail, self.icp_total, cfg.icp_monitors)

        self.beds_total = len(self.bed_occupancy)

    @property
    def beds_avail(self) -> int:
        return sum(1 for b in self.bed_occupancy if b is None)


@dataclass
class SimStats:
    total_admissions: int = 0
    total_discharges: int = 0
    total_deaths: int = 0
    no_bed_deaths: int = 0
    vap_cases: int = 0
    total_los_days: float = 0.0
    total_cost: float = 0.0
    events: List[str] = field(default_factory=list)


class NeuroICUSimulation:
    """Stateful, steppable Neuro-ICU simulation. Call .step() repeatedly, or .run()."""

    def __init__(self, cfg: Optional[SimulationConfig] = None, seed: Optional[int] = None):
        self.cfg = cfg or SimulationConfig()
        self.rng = np.random.default_rng(seed)
        self.pyrandom = random.Random(seed)

        self.time_hours: float = 0.0
        self.pool = ResourcePool(self.cfg)
        self.patients: Dict[str, Patient] = {}
        self.completed: List[Patient] = []
        self.stats = SimStats()

        self._seed_initial_cohort()

    # ------------------------------------------------------------------
    def _seed_initial_cohort(self):
        init_count = min(int(self.cfg.beds * 0.7), self.cfg.beds)
        for i in range(init_count):
            dx = self.pyrandom.choice(DISEASES)
            p = Patient(0.0, dx, self.cfg, self.rng, initial=True)
            p.bed_id = i
            self.pool.bed_occupancy[i] = p.id
            if self.pool.doctors_avail > 0 and self.rng.random() < 0.8:
                p.doctor_id = f"DOC-{i+1}"
                self.pool.doctors_avail -= 1
            if self.pool.nurses_avail > 0 and self.rng.random() < 0.8:
                p.nurse_id = f"NUR-{i+1}"
                self.pool.nurses_avail -= 1
            if p.vent_required and self.pool.vents_avail > 0 and self.rng.random() < 0.7:
                p.has_vent = True
                self.pool.vents_avail -= 1
            if p.icp_required and self.pool.icp_avail > 0 and self.rng.random() < 0.7:
                p.has_icp = True
                self.pool.icp_avail -= 1
            self.patients[p.id] = p
            self.stats.total_admissions += 1

    # ------------------------------------------------------------------
    def _admit(self, dx, initial=False):
        p = Patient(self.time_hours, dx, self.cfg, self.rng, initial=initial)
        self.patients[p.id] = p
        self.stats.total_admissions += 1
        self.stats.events.append(f"[Day {self.time_hours/24:.1f}] Arrival {p.id} | {dx.name}")
        return p

    def _release(self, p: Patient):
        if p.bed_id is not None:
            self.pool.bed_occupancy[p.bed_id] = None
            p.bed_id = None
        if p.doctor_id is not None:
            self.pool.doctors_avail += 1
            p.doctor_id = None
        if p.nurse_id is not None:
            self.pool.nurses_avail += 1
            p.nurse_id = None
        if p.has_vent:
            self.pool.vents_avail += 1
            p.has_vent = False
        if p.has_icp:
            self.pool.icp_avail += 1
            p.has_icp = False

    def _assign_beds(self):
        waiting = [p for p in self.patients.values() if p.status == "waiting_bed"]
        waiting.sort(key=lambda p: _priority_score(p, self.cfg.triage_policy), reverse=True)
        for p in waiting:
            if self.pool.beds_avail <= 0:
                p.bed_wait_hours += 1
                p.step_waiting_decay()
                continue
            idx = next(i for i, b in enumerate(self.pool.bed_occupancy) if b is None)
            self.pool.bed_occupancy[idx] = p.id
            p.bed_id = idx
            p.status = "occupying"
            p.log(self.time_hours, "Bed assigned", f"Bed {idx+1}")

    def _assign_staff(self):
        """
        Fill missing doctor/nurse/ventilator/ICP-monitor for bedded patients each
        tick (a patient keeps retrying for a resource it still needs, rather than
        a single all-or-nothing attempt at bed-assignment time). A patient is NOT
        blocked from health/mortality updates while it waits -- deprivation instead
        shows up as an elevated decay rate (Patient.decay_rate_k) and mortality
        multiplier (mortality.hourly_death_probability), matching the blueprint's
        resource-deprivation hazard formulation (Eq. 3.8-3.9).
        """
        occupying = [p for p in self.patients.values() if p.status == "occupying"]
        occupying.sort(key=lambda p: _priority_score(p, self.cfg.triage_policy), reverse=True)
        for p in occupying:
            if p.doctor_id is None and self.pool.doctors_avail > 0:
                p.doctor_id = f"DOC-{self.pool.doctors_total - self.pool.doctors_avail + 1}"
                self.pool.doctors_avail -= 1
                p.log(self.time_hours, "Doctor assigned")
            if p.nurse_id is None and self.pool.nurses_avail > 0:
                p.nurse_id = f"NUR-{self.pool.nurses_total - self.pool.nurses_avail + 1}"
                self.pool.nurses_avail -= 1
            if p.vent_required and not p.has_vent and self.pool.vents_avail > 0:
                p.has_vent = True
                self.pool.vents_avail -= 1
            if p.icp_required and not p.has_icp and self.pool.icp_avail > 0:
                p.has_icp = True
                self.pool.icp_avail -= 1
            if p.doctor_id is None or p.nurse_id is None:
                p.resource_wait_hours += 1

    def _update_health_and_survival(self):
        deaths_this_step = 0
        for p in list(self.patients.values()):
            p.accrue_costs(self.time_hours)
            if p.status != "occupying":
                continue

            p.step_health()

            onset = p.step_vap(self.time_hours)
            if onset:
                self.stats.vap_cases += 1

            if self.rng.random() < p.hourly_death_probability():
                p.status = "dead"
                p.outcome = "dead"
                p.discharge_time = self.time_hours
                self.stats.total_deaths += 1
                deaths_this_step += 1
                self._release(p)
                self.completed.append(p)
                del self.patients[p.id]
                self.stats.events.append(f"[Day {self.time_hours/24:.1f}] Death {p.id} | {p.diagnosis.name}")
                continue

            p.critical = p.health < 30
        return deaths_this_step

    def _check_discharge(self):
        for p in list(self.patients.values()):
            if p.status != "occupying":
                continue
            los = p.los_days(self.time_hours)
            if los < p.min_los_days():
                continue
            if p.health >= self.cfg.discharge_health_threshold:
                p.stable_hours += 1
                if p.stable_hours >= self.cfg.stability_window_hours:
                    p.status = "discharged"
                    p.outcome = "discharged"
                    p.discharge_time = self.time_hours
                    self.stats.total_discharges += 1
                    self.stats.total_los_days += los
                    self._release(p)
                    self.completed.append(p)
                    del self.patients[p.id]
                    self.stats.events.append(f"[Day {self.time_hours/24:.1f}] Discharge {p.id} | LOS {los:.1f}d")
            else:
                p.stable_hours = 0

    # ------------------------------------------------------------------
    def step(self) -> Dict:
        """Advance the simulation by one dt_hours tick. Returns a small event summary."""
        self.time_hours += self.cfg.dt_hours

        n_new = sample_arrivals(self.time_hours, self.cfg, self.rng)
        for _ in range(n_new):
            dx = self.pyrandom.choice(DISEASES)
            self._admit(dx)

        for p in self.patients.values():
            if p.status == "waiting_bed":
                p.accrue_wait_cost()
        for p in list(self.patients.values()):
            if p.status == "waiting_bed" and p.health <= 2 and p.bed_wait_hours > 72:
                p.status = "dead"
                p.outcome = "dead"
                p.discharge_time = self.time_hours
                self.stats.total_deaths += 1
                self.stats.no_bed_deaths += 1
                self.completed.append(p)
                del self.patients[p.id]
                self.stats.events.append(f"[Day {self.time_hours/24:.1f}] Death (no bed) {p.id}")

        self._assign_beds()
        self._assign_staff()
        deaths = self._update_health_and_survival()
        self._check_discharge()

        return {"t": self.time_hours, "new_admissions": n_new, "deaths": deaths}

    def run_hours(self, n_hours: float):
        n = int(round(n_hours / self.cfg.dt_hours))
        for _ in range(n):
            self.step()

    def run_days(self, n_days: float):
        self.run_hours(n_days * 24.0)

    # ------------------------------------------------------------------
    def mortality_rate(self) -> float:
        total = self.stats.total_admissions
        return self.stats.total_deaths / total if total else 0.0

    def occupancy(self) -> float:
        used = self.pool.beds_total - self.pool.beds_avail
        return used / self.pool.beds_total if self.pool.beds_total else 0.0

    def aggregate_health(self) -> float:
        active = [p.health for p in self.patients.values() if p.status in ("active", "critical")]
        return float(np.mean(active)) if active else 100.0

    def total_cost(self) -> float:
        running = sum(p.total_cost for p in self.patients.values())
        completed = sum(p.total_cost for p in self.completed)
        return running + completed

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Export completed + in-progress patients as a cohort dataframe, matching the
        v1 CSV schema (neuro_icu_1500_patients_final.csv) with added columns for the
        new stochastic sub-models."""
        rows = []
        for p in list(self.completed) + list(self.patients.values()):
            rows.append(dict(
                patient_id=p.id,
                diagnosis=p.diagnosis.name,
                apache=p.apache,
                gcs=p.gcs,
                final_health=round(p.health, 1),
                outcome=p.outcome or p.status,
                los_days=round(p.los_days(self.time_hours), 2),
                total_cost=round(p.total_cost, 0),
                bed_wait_hours=round(p.bed_wait_hours, 1),
                vent_used=int(p.has_vent or p.hours_intubated > 0),
                icp_used=int(p.has_icp),
                vap=int(p.vap),
                doctor_assigned=int(p.doctor_id is not None),
                nurse_assigned=int(p.nurse_id is not None),
                expected_los_days=round((p.diagnosis.min_los_days + p.diagnosis.max_los_days) / 2, 1),
                hourly_death_prob_pct=round(p.hourly_death_probability(), 6),
            ))
        return pd.DataFrame(rows)

    def summary(self) -> Dict:
        df = self.to_dataframe()
        discharged = df[df.outcome == "discharged"]
        dead = df[df.outcome == "dead"]
        return dict(
            total_patients=len(df),
            deaths=len(dead),
            discharges=len(discharged),
            mortality_rate_pct=100 * len(dead) / len(df) if len(df) else 0.0,
            mean_los_days=discharged.los_days.mean() if len(discharged) else float("nan"),
            mean_cost=df.total_cost.mean() if len(df) else float("nan"),
            total_cost=df.total_cost.sum() if len(df) else 0.0,
            vap_rate_pct=100 * df.vap.mean() if len(df) else 0.0,
            vent_rate_pct=100 * df.vent_used.mean() if len(df) else 0.0,
        )
