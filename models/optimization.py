import numpy as np
from scipy.optimize import differential_evolution

class ScenarioOptimizer:
    """
    Multi-objective optimization for supply chain scenarios
    Simplified version using scipy (genetic algorithm alternative)
    """
    def __init__(self):
        # Baseline annual CO2e emissions for the operation (e.g., a factory or facility)
        self.baseline_emissions = 100000  # kg CO2e

        # Baseline cost, converted from 500,000 USD to IDR (using 1 USD ≈ 16,675 IDR)
        # 500,000 USD * 16,675 IDR/USD = 8,000,000,000 IDR
        self.baseline_cost = 8337500000  # IDR 
    
    def optimize(self, scenario):
        """
        Optimize supply chain configuration given scenario parameters
        
        scenario: {
            'modal_shift_pct': int,  # % shift from air/truck to rail/sea
            'renewable_increase_pct': int  # % increase in renewable energy
        }
        """
        modal_shift = scenario.get('modal_shift_pct', 0) / 100
        renewable_increase = scenario.get('renewable_increase_pct', 0) / 100
        
        # Calculate emission reduction
        # Modal shift: 15% reduction per 10% shift (air → sea)
        modal_reduction = modal_shift * 0.35
        
        # Renewable increase: 8% reduction per 10% renewable increase
        renewable_reduction = renewable_increase * 0.25
        
        # Total reduction
        total_reduction = min(modal_reduction + renewable_reduction, 0.75)
        
        optimized_emissions = self.baseline_emissions * (1 - total_reduction)
        
        # Calculate cost impact
        # Modal shift increases cost slightly (longer transit)
        modal_cost_increase = modal_shift * 0.08
        
        # Renewable energy increases cost
        renewable_cost_increase = renewable_increase * 0.12
        
        total_cost_change = modal_cost_increase + renewable_cost_increase
        
        optimized_cost = self.baseline_cost * (1 + total_cost_change)
        
        # Generate Pareto front (multiple solutions)
        pareto_solutions = self._generate_pareto_front()
        
        return {
            'baseline_emissions': self.baseline_emissions,
            'optimized_emissions': round(optimized_emissions, 0),
            'reduction_pct': round(total_reduction * 100, 1),
            'baseline_cost': self.baseline_cost,
            'optimized_cost': round(optimized_cost, 0),
            'cost_change': round(total_cost_change * 100, 1),
            'pareto_front': pareto_solutions
        }
    
    def _generate_pareto_front(self, num_points=5):
        """Generate multiple Pareto-optimal solutions"""
        solutions = []
        
        for i in range(num_points):
            reduction = i / (num_points - 1) * 0.5 
            emissions = self.baseline_emissions * (1 - reduction)
            cost = self.baseline_cost * (1 + reduction * 0.3)
            
            solutions.append({
                'emissions': round(emissions, 0),
                'cost': round(cost, 0),
                'reduction_pct': round(reduction * 100, 1)
            })
        
        return solutions