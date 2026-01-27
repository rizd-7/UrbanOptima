import os
import logging
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from config import CONFIG
from MultiCriteriaAnalyzer import MultiCriteriaAnalyzer
from SmartSwarmOptimizer import SmartSwarmOptimizer




class AblationStudyEngine:
    """
    SCIENTIFIC VALIDATION SUITE
    Conducts an ablation study to quantify the contribution of each system component.
    Methodology:
    1. Define a "Ground Truth" evaluator (Full Physics + Full Constraints).
    2. Run 5 scenarios removing one component at a time.
    3. Compare the final sites against the Ground Truth metric.
    """
    def __init__(self, graph, buildings, facilities):
        self.G = graph
        self.buildings = buildings
        self.facilities = facilities
        self.base_weights = CONFIG["MULTI_CRITERIA_WEIGHTS"].copy()
        
        # Pre-calculate the "Ground Truth" Composite Score (Full System)
        # We need this to evaluate how well the ablated models perform
        # relative to the best possible standard.
        self.analyzer = MultiCriteriaAnalyzer(graph, buildings, facilities)
        self.full_composite, self.full_criteria = self.analyzer.analyze_all_criteria()
        
    def run_ablation_study(self):
        logging.info("="*80)
        logging.info("PHASE 8: ABLATION STUDY & VALIDATION")
        logging.info("="*80)
        
        results = {}
        
        # 1. Random Baseline (The "Dartboard" approach)
        results['baseline_random'] = self._run_random()
        
        # 2. No Spatial Constraints (Clustering allowed)
        results['no_constraints'] = self._run_no_constraints()
        
        # 3. No 2SFCA (Ignore accessibility gaps)
        results['no_accessibility'] = self._run_no_access()
        
        # 4. No Physics/Terrain (Assume flat earth)
        results['no_physics'] = self._run_no_physics()
        
        # 5. Full System (The proposed solution)
        results['full_system'] = self._run_full_system()
        
        # Visualize
        self._plot_ablation_results(results)
        
        return results

    def _evaluate_set(self, nodes, check_constraints=True):
        """
        Universal Evaluator: 
        Scores ANY set of nodes based on the FULL SYSTEM criteria.
        This represents the 'Ground Truth' quality of the decision.
        """
        if not nodes: return 0
        
        # A. Sum of Ground Truth Composite Scores
        total_score = sum(self.full_composite.get(n, 0) for n in nodes)
        
        # B. Apply Clustering Penalty (if enabled in Ground Truth)
        # Even if the sub-model ignored constraints, the validation metric must penalize it.
        penalty = 0
        if check_constraints:
            coords = np.array([[self.G.nodes[n]['x'], self.G.nodes[n]['y']] for n in nodes])
            if len(nodes) > 1:
                from scipy.spatial import distance_matrix
                dists = distance_matrix(coords, coords) * 111000 # deg to meters
                # Count pairs closer than threshold
                limit = CONFIG["OPTIMIZATION_PARAMS"]["MIN_FACILITY_SPACING_M"]
                # (count < limit) - diagonal
                violations = (dists < limit).sum() - len(nodes)
                penalty = violations * 0.25  # Heavy penalty for clustering
        
        return max(0, total_score - penalty)

    def _run_random(self):
        logging.info("  Testing Scenario: Random Baseline...")
        candidates = list(self.G.nodes())
        # Average of 10 random trials to be fair
        scores = []
        for _ in range(10):
            selection = random.sample(candidates, CONFIG["OPTIMIZATION_PARAMS"]["N_NEW_FACILITIES"])
            scores.append(self._evaluate_set(selection))
        return np.mean(scores)

    def _run_no_constraints(self):
        logging.info("  Testing Scenario: No Spatial Constraints...")
        # Run Optimizer with spacing = 0
        original_spacing = CONFIG["OPTIMIZATION_PARAMS"]["MIN_FACILITY_SPACING_M"]
        CONFIG["OPTIMIZATION_PARAMS"]["MIN_FACILITY_SPACING_M"] = 0
        
        optimizer = SmartSwarmOptimizer(self.G, self.full_composite, self.full_criteria)
        selection = optimizer.run()
        
        # Restore config
        CONFIG["OPTIMIZATION_PARAMS"]["MIN_FACILITY_SPACING_M"] = original_spacing
        
        return self._evaluate_set(selection)

    def _run_no_access(self):
        logging.info("  Testing Scenario: No 2SFCA (Accessibility)...")
        # Temporarily zero out accessibility weight
        mod_weights = self.base_weights.copy()
        mod_weights['accessibility_gap'] = 0
        
        # Re-weight composite
        mod_composite = {n: 0.0 for n in self.G.nodes()}
        for crit, w in mod_weights.items():
            for n in self.G.nodes():
                mod_composite[n] += self.full_criteria[crit].get(n, 0) * w
        
        optimizer = SmartSwarmOptimizer(self.G, mod_composite, self.full_criteria)
        selection = optimizer.run()
        
        return self._evaluate_set(selection)

    def _run_no_physics(self):
        logging.info("  Testing Scenario: No Physics (Flat Earth)...")
        # Simulate by removing terrain score and slope penalties
        mod_weights = self.base_weights.copy()
        mod_weights['terrain_suitability'] = 0
        
        mod_composite = {n: 0.0 for n in self.G.nodes()}
        for crit, w in mod_weights.items():
            for n in self.G.nodes():
                mod_composite[n] += self.full_criteria[crit].get(n, 0) * w
                
        optimizer = SmartSwarmOptimizer(self.G, mod_composite, self.full_criteria)
        selection = optimizer.run()
        
        # Note: The _evaluate_set will punish this selection if it picked 
        # nodes on steep slopes (because _evaluate_set uses the REAL terrain scores)
        return self._evaluate_set(selection)

    def _run_full_system(self):
        logging.info("  Testing Scenario: Full Integrated System...")
        optimizer = SmartSwarmOptimizer(self.G, self.full_composite, self.full_criteria)
        selection = optimizer.run()
        return self._evaluate_set(selection)

    def _plot_ablation_results(self, results):
        try:
            plt.figure(figsize=(10, 6))
            
            scenarios = ['Random', 'No Constraints', 'No Physics', 'No 2SFCA', 'Full System']
            scores = [
                results['baseline_random'],
                results['no_constraints'],
                results['no_physics'],
                results['no_accessibility'],
                results['full_system']
            ]
            
            colors = ['#95a5a6', '#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']
            
            bars = plt.bar(scenarios, scores, color=colors, edgecolor='black', alpha=0.8)
            
            # Add improvement labels
            base = results['baseline_random']
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                if score > base:
                    imp = ((score - base) / base) * 100
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'+{imp:.0f}%',
                             ha='center', va='bottom', fontweight='bold')
            
            plt.ylabel("Global Suitability Score")
            plt.title("Ablation Study: Component Contribution Analysis", fontweight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            output_path = os.path.join(CONFIG["OUTPUT_DIR"], "ablation_study.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"   Ablation chart saved: {output_path}")
            
        except Exception as e:
            logging.warning(f"   couldnt not generate ablation plot: {e}")