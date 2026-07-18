import simpy
import numpy as np
import scipy.stats as stats
import pandas as pd

class UpgradedStochasticCore:
    """Handles advanced statistical modeling for arrivals, severities, and complications."""
    def __init__(self, base_arrival_rate=2.8/24, surge_amplitude=1.5/24, copula_rho=-0.75):
        # 2.8 patients/day baseline converted to hourly rate[cite: 1]
        self.base_arrival = base_arrival_rate 
        self.surge_amplitude = surge_amplitude
        
        # Gaussian Copula Setup for APACHE II and GCS correlation
        self.cov_matrix = np.array([[1.0, copula_rho], [copula_rho, 1.0]])
        self.cholesky_L = np.linalg.cholesky(self.cov_matrix)

    def get_next_arrival_time_nhpp(self, current_time):
        """Non-Homogeneous Poisson Process (NHPP) using Lewis-Shedler Thinning."""
        max_lambda = self.base_arrival + self.surge_amplitude
        t = current_time
        while True:
            t += np.random.exponential(1.0 / max_lambda)
            # Sinusoidal circadian oscillation (peaks every 24 hours)
            current_lambda = self.base_arrival + self.surge_amplitude * np.sin(2 * np.pi * t / 24)
            if np.random.random() < (current_lambda / max_lambda):
                return t

    def generate_patient_severity(self):
        """Generates realistic, correlated clinical metrics using a Gaussian Copula."""
        # 1. Sample correlated latent standard normals
        z = np.random.normal(size=2)
        correlated_z = self.cholesky_L @ z
        
        # 2. Convert to uniform margins via normal CDF
        u = stats.norm.cdf(correlated_z)
        
        # 3. Inverse transform sampling to target clinical bounds
        # APACHE II distribution baseline: mean=24.46, sd=7.09[cite: 1]
        apache = int(np.clip(stats.norm.ppf(u[0], loc=24.46, scale=7.09), 10, 45))
        
        # GCS bound between 3 and 15 (Targeting mean approx 9.1)[cite: 1]
        gcs = int(np.clip(3 + 12 * stats.beta.ppf(u[1], a=4, b=2), 3, 15))
        
        # Map continuous physiological health reserve fraction H(0) [0.0 - 1.0]
        h_0 = float(np.clip(stats.beta.rvs(a=2, b=max(1.5, apache/10.0)), 0.15, 0.95))
        
        # Randomly assign a neurological diagnosis from the defined cohort[cite: 1]
        diagnoses = ['SAH', 'ICH', 'Severe TBI', 'Status Epilepticus', 'Ischemic Stroke']
        dx = np.random.choice(diagnoses, p=[0.25, 0.15, 0.25, 0.15, 0.20])
        
        return dx, apache, gcs, h_0

    def get_weibull_vap_hazard(self, t_vent, apache):
        """Calculates dynamic hazard rate for Ventilator-Associated Pneumonia."""
        shape = 1.6   # Hazard increases significantly over time
        scale = 96.0  # Characteristic timeline for onset window
        gamma = 0.03  # Scaling factor for baseline severity accentuation
        
        hazard = (shape / scale) * ((t_vent / scale) ** (shape - 1)) * np.exp(gamma * apache)
        return min(hazard, 0.85)

class NeuroPatient:
    """Tracks state data, clinical parameters, and timeline metrics for an individual."""
    def __init__(self, pid, dx, apache, gcs, health):
        self.pid = pid
        self.dx = dx
        self.apache = apache
        self.gcs = gcs
        self.health = health
        
        # Operational Timestamps
        self.arrival_time = 0
        self.admission_time = None
        self.discharge_time = None
        
        # Resource hours & Complications
        self.vent_hours = 0
        self.icp_hours = 0
        self.has_vap = False
        self.status = "Waiting" # Waiting, Admitted, Discharged, Deceased

class NeuroICU_Engine:
    """Core Discrete Event Simulation model executing patient processing loops."""
    def __init__(self, num_beds=18, num_vents=14, num_icp=12, num_nurses=20):
        self.env = simpy.Environment()
        self.stochastic = UpgradedStochasticCore()
        
        # Define SimPy Resource Capacities based on standard configuration[cite: 1]
        self.beds = simpy.Resource(self.env, capacity=num_beds)
        self.vents = simpy.Resource(self.env, capacity=num_vents)
        self.icp = simpy.Resource(self.env, capacity=num_icp)
        self.nurses = simpy.Resource(self.env, capacity=num_nurses)
        
        # System State Logging
        self.all_patients = {}
        self.operational_log = []
        self.total_cost_inr = 0.0

    def log_system_state(self):
        """Captures structured snapshots of resource bottlenecks hourly."""
        self.operational_log.append({
            "hour": self.env.now,
            "bed_occupancy": self.beds.count,
            "vent_utilization": self.vents.count,
            "icp_utilization": self.icp.count,
            "nurse_utilization": self.nurses.count,
            "queue_length": len(self.beds.queue),
            "cumulative_cost": self.total_cost_inr
        })

    def patient_lifecycle_process(self, patient):
        """Tracks the non-linear patient trajectory throughout their stay."""
        patient.arrival_time = self.env.now
        self.all_patients[patient.pid] = patient
        
        # Request Bed Asset via FIFO Queue
        bed_request = self.beds.request()
        yield bed_request
        
        patient.admission_time = self.env.now
        patient.status = "Admitted"
        
        # Daily resource baseline rate extraction (calculated per hour)
        # Bed Base Cost: INR 6,000/day[cite: 1]
        hourly_bed_cost = 6000 / 24 
        
        # Track active resource allocations
        vent_request = None
        icp_request = None

        while patient.status == "Admitted":
            yield self.env.timeout(1) # Execute simulation in hourly ticks[cite: 1]
            self.total_cost_inr += hourly_bed_cost
            
            # Non-linear clinical evaluation triggers
            vent_needed = patient.health < 0.45
            icp_needed = patient.dx in ['Severe TBI', 'ICH'] and patient.health < 0.60
            
            # Dynamic physiological baseline decay constant
            k = 0.015 + (0.045 if patient.has_vap else 0.0)
            
            # Ventilator Allocation Domain Logic
            if vent_needed:
                if vent_request is None and self.vents.count < self.vents.capacity:
                    vent_request = self.vents.request()
                    yield vent_request
                
                if vent_request and vent_request.triggered:
                    patient.vent_hours += 1
                    self.total_cost_inr += (2500 / 24) # INR 2,500/day ventilator fee[cite: 1]
                    k -= 0.012 # Mechanical ventilation stabilizes drift
                    
                    # Compute Weibull risk factor for long-term intubation
                    if not patient.has_vap:
                        hazard = self.stochastic.get_weibull_vap_hazard(patient.vent_hours, patient.apache)
                        if np.random.random() < hazard:
                            patient.has_vap = True
                            self.total_cost_inr += 5000 # Instantaneous VAP clinical surcharge[cite: 1]
                else:
                    k += 0.075 # Exponential health crash due to ventilator deprivation
            else:
                if vent_request:
                    self.vents.release(vent_request)
                    vent_request = None

            # ICP Monitor Allocation Domain Logic
            if icp_needed:
                if icp_request is None and self.icp.count < self.icp.capacity:
                    icp_request = self.icp.request()
                    yield icp_request
                
                if icp_request and icp_request.triggered:
                    patient.icp_hours += 1
                    self.total_cost_inr += (1500 / 24) # INR 1,500/day monitor fee[cite: 1]
                    k -= 0.004
                else:
                    k += 0.025 # Elevated threat matrix if missing required neuromonitoring
            else:
                if icp_request:
                    self.icp.release(icp_request)
                    icp_request = None

            # Update continuous health variable via non-linear exponential drift
            patient.health = float(np.clip(patient.health * np.exp(-k), 0.0, 1.0))
            
            # Hourly systemic state logging hook
            if int(self.env.now) % 1 == 0:
                self.log_system_state()

            # Terminal Gate Criteria
            if patient.health <= 0.05:
                patient.status = "Deceased"
                patient.discharge_time = self.env.now
            elif patient.health >= 0.80 and not vent_needed:
                # Dual-Gate Discharge check: patient must be stable and off life-support
                patient.status = "Discharged"
                patient.discharge_time = self.env.now

        # Clean up remaining resources on exit
        if vent_request: self.vents.release(vent_request)
        if icp_request: self.icp.release(icp_request)
        self.beds.release(bed_request)

    def arrival_generator_loop(self):
        """NHPP scheduler producing patient arrivals across global runtime timeline."""
        pid = 0
        while True:
            next_arrival_time = self.stochastic.get_next_arrival_time_nhpp(self.env.now)
            yield self.env.timeout(next_arrival_time - self.env.now)
            
            dx, apache, gcs, h_0 = self.stochastic.generate_patient_severity()
            patient = NeuroPatient(pid, dx, apache, gcs, h_0)
            pid += 1
            
            self.env.process(self.patient_lifecycle_process(patient))

    def run_simulation(self, duration_hours=720):
        """
        Starts the simulator thread execution block.
        Returns Pandas DataFrames requested by the PyScript frontend.
        """
        self.env.process(self.arrival_generator_loop())
        self.env.run(until=duration_hours)
        
        # Package and export summary histories to be visualized by the frontend
        patient_data = [{
            "pid": p.pid, 
            "dx": p.dx, 
            "apache": p.apache, 
            "gcs": p.gcs,
            "health": p.health, 
            "status": p.status, 
            "vent_hours": p.vent_hours,
            "has_vap": p.has_vap, 
            "los_days": (p.discharge_time - p.arrival_time)/24 if p.discharge_time else None
        } for p in self.all_patients.values()]
        
        df_patients = pd.DataFrame(patient_data)
        df_ops = pd.DataFrame(self.operational_log)
        
        # Fallback to prevent empty DataFrame errors in plotting
        if df_ops.empty:
            df_ops = pd.DataFrame({
                "hour": [0], "bed_occupancy": [0], "vent_utilization": [0], 
                "icp_utilization": [0], "nurse_utilization": [0], 
                "queue_length": [0], "cumulative_cost": [0.0]
            })
            
        return df_patients, df_ops
