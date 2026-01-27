import os
import sys
import time
import logging
import warnings
import random
import json
from math import exp, radians, cos, sin, asin, sqrt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Core dependencies
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import requests
from shapely.geometry import Point, Polygon, MultiPolygon
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import percentileofscore

# ML Stack
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

# Visualization
import folium
from folium import plugins
import matplotlib.pyplot as plt
import seaborn as sns


#Local
from config import CONFIG


class SensitivityAnalyzer:
    """
    Tests the robustness of the decision model by varying the Multi-Criteria Weights.
    Hypothesis: Core "best" sites should remain relatively stable across different priorities.
    """
    def __init__(self, analyzer_instance, optimizer_class):
        self.analyzer = analyzer_instance
        self.OptimizerCls = optimizer_class
        self.G = analyzer_instance.G
        self.criteria_scores = analyzer_instance.criteria_scores
        
        # Define Scenarios
        self.scenarios = {
            "Accessibility-First": {
                "accessibility_gap": 0.60,
                "population_density": 0.10,
                "terrain_suitability": 0.10,
                "network_centrality": 0.10,
                "spatial_equity": 0.10
            },
            "Population-First": {
                "accessibility_gap": 0.10,
                "population_density": 0.60,
                "terrain_suitability": 0.10,
                "network_centrality": 0.10,
                "spatial_equity": 0.10
            },
            "Balanced": {
                "accessibility_gap": 0.20,
                "population_density": 0.20,
                "terrain_suitability": 0.20,
                "network_centrality": 0.20,
                "spatial_equity": 0.20
            },
            "Equity-First": {
                "accessibility_gap": 0.15,
                "population_density": 0.15,
                "terrain_suitability": 0.10,
                "network_centrality": 0.15,
                "spatial_equity": 0.45
            }
        }

    def run_sensitivity_analysis(self):
        logging.info("="*80)
        logging.info("PHASE 9.5: SENSITIVITY ANALYSIS")
        logging.info("="*80)
        
        results = []
        
        for name, weights in self.scenarios.items():
            logging.info(f"  Testing Scenario: {name}...")
            
            # 1. Recalculate Composite Score
            composite = self._recompute_composite(weights)
            
            # 2. Run Optimization
            # We instantiate a fresh optimizer with the NEW composite scores
            optimizer = self.OptimizerCls(self.G, composite, self.criteria_scores)
            selected_nodes = optimizer.run()
            
            results.append({
                "scenario": name,
                "weights": weights,
                "selected_nodes": selected_nodes
            })
            
        self._analyze_overlaps(results)
        return results

    def _recompute_composite(self, weights):
        """Quickly re-sums the weighted scores without re-running spatial analysis"""
        composite = {n: 0.0 for n in self.G.nodes()}
        
        for criterion, weight in weights.items():
            scores = self.criteria_scores.get(criterion, {})
            for node, score in scores.items():
                composite[node] += score * weight
                
        return composite

    def _analyze_overlaps(self, results):
        """Visualizes how often nodes are selected across scenarios"""
        
        # Count occurrences of each node
        node_counts = {}
        all_selected = set()
        
        for res in results:
            for node in res['selected_nodes']:
                node_counts[node] = node_counts.get(node, 0) + 1
                all_selected.add(node)
        
        # Sort by frequency
        sorted_nodes = sorted(all_selected, key=lambda n: node_counts[n], reverse=True)
        top_nodes = sorted_nodes[:10] # Take top 10 unique nodes found
        
        # Prepare Data for Heatmap
        # Rows: Scenarios, Cols: Top Nodes
        matrix = np.zeros((len(results), len(top_nodes)))
        
        scenario_names = [r['scenario'] for r in results]
        node_labels = [str(n) for n in top_nodes]
        
        for i, res in enumerate(results):
            for j, node in enumerate(top_nodes):
                if node in res['selected_nodes']:
                    matrix[i, j] = 1
        
        # Plotting
        try:
            plt.figure(figsize=(10, 6))
            sns.heatmap(matrix, annot=True, cmap="YlGnBu", cbar=False,
                        xticklabels=node_labels, yticklabels=scenario_names,
                        linewidths=.5, linecolor='gray')
            
            plt.title("Sensitivity Analysis: Stability of Selected Sites", fontweight='bold')
            plt.xlabel("Top Candidate Nodes (ID)")
            plt.ylabel("Weight Scenarios")
            
            output_path = os.path.join(CONFIG["OUTPUT_DIR"], "sensitivity_analysis.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logging.info(f"   Sensitivity heatmap saved: {output_path}")
            
            # Console Summary
            robust_count = sum(1 for n in node_counts.values() if n == len(results))
            logging.info(f"   Robustness Check: {robust_count} sites appeared in ALL {len(results)} scenarios")
            
        except Exception as e:
            logging.warning(f"   couldnt not generate sensitivity plot: {e}")
