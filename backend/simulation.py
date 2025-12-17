import numpy as np
import pandas as pd

class BedSimulationEngine:
    def __init__(self, current_census, num_beds=50, arrival_rate_lambda=5):
        self.census = current_census  # List of remaining LOS for current patients
        self.num_beds = num_beds
        self.arrival_rate = arrival_rate_lambda # Avg patients per day (Poisson)
        
    def run_forecast(self, days=30, simulations=100):
        """
        Runs Monte Carlo simulation to predict occupancy risk.
        """
        results = np.zeros((simulations, days))
        
        for sim in range(simulations):
            # Clone initial state
            current_beds = self.census.copy()
            
            for day in range(days):
                # 1. Discharge Logic: Decrement LOS for all active patients
                current_beds = [los - 1 for los in current_beds if los > 1]
                
                # 2. Admission Logic: Sample new arrivals from Poisson distribution
                new_arrivals = np.random.poisson(self.arrival_rate)
                
                # 3. Assign LOS to new arrivals (Log-Normal distribution based on hospital stats)
                new_los_values = np.random.lognormal(mean=1.5, sigma=0.6, size=new_arrivals)
                current_beds.extend(new_los_values)
                
                # 4. Record Occupancy
                occupancy = len(current_beds)
                results[sim, day] = occupancy
                
        # Aggregate results
        p50 = np.median(results, axis=0)
        p95 = np.percentile(results, 95, axis=0) # "Worst case" scenario
        
        return {
            "days": list(range(1, days + 1)),
            "median_occupancy": p50.tolist(),
            "risk_occupancy": p95.tolist(),
            "shortage_days": np.where(p95 > self.num_beds)[0].tolist()
        }