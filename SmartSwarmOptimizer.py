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


os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
warnings.filterwarnings('ignore')



            
class SmartSwarmOptimizer:
    """PSO with spatial conflict resolution"""
    def __init__(self, graph, composite_scores, criteria_scores):
        self.G = graph
        self.composite_scores = composite_scores
        self.criteria_scores = criteria_scores
        self.valid_nodes = self._apply_constraints()
        
    def _apply_constraints(self):
        logging.info("="*80)
        logging.info("PHASE 5: OPTIMIZATION WITH ZONING CONSTRAINTS")
        logging.info("="*80)
        
        valid = []
        constraints = CONFIG["ZONING_CONSTRAINTS"]
        
        for n in self.G.nodes():
            is_valid = True
            edges = list(self.G.edges(n, data=True))
            
            if not edges:
                continue
            
            for _, _, d in edges:
                # Highway check
                if d.get('highway') in constraints['excluded_highways']:
                    is_valid = False
                    break
                
                # Railway check
                if 'railway' in constraints['excluded_features'] and d.get('railway'):
                    is_valid = False
                    break
                
                # Tunnel check
                if 'tunnel' in constraints['excluded_features'] and d.get('tunnel') == 'yes':
                    is_valid = False
                    break
            
            # Slope check
            slopes = [abs(d.get('slope', 0)) for _, _, d in edges]
            if slopes and max(slopes) > constraints['max_slope']:
                is_valid = False
            
            # Score check
            if self.composite_scores.get(n, 0) < CONFIG["CLASSIFICATION_RULES"]["MIN_JUSTIFICATION_SCORE"]:
                is_valid = False
            
            if is_valid:
                valid.append(n)
        
        logging.info(f"   Valid candidates: {len(valid)} / {len(self.G.nodes)} ({100*len(valid)/len(self.G.nodes):.1f}%)")
        return valid
    
    def run(self):
        logging.info("  Initializing Particle Swarm Optimization...")
        
        params = CONFIG["OPTIMIZATION_PARAMS"]
        n_fac = params["N_NEW_FACILITIES"]
        
        if len(self.valid_nodes) < n_fac:
            logging.error(f"  ✗ Insufficient valid nodes ({len(self.valid_nodes)} < {n_fac})")
            return []
        
        # Initialize swarm
        n_particles = params["SWARM_PARTICLES"]
        particles = np.array([
            random.sample(range(len(self.valid_nodes)), n_fac)
            for _ in range(n_particles)
        ])
        
        velocities = np.random.randn(n_particles, n_fac) * 0.1
        pbest_pos = particles.copy()
        pbest_val = np.full(n_particles, -np.inf)
        gbest_pos = particles[0].copy()
        gbest_val = -np.inf
        
        # Optimization loop
        for iteration in range(params["SWARM_ITERATIONS"]):
            for i in range(n_particles):
                nodes = [self.valid_nodes[idx] for idx in particles[i]]
                
                # Fitness evaluation
                score = self._evaluate_fitness(nodes)
                
                if score > pbest_val[i]:
                    pbest_val[i] = score
                    pbest_pos[i] = particles[i].copy()
                
                if score > gbest_val:
                    gbest_val = score
                    gbest_pos = particles[i].copy()
            
            # Update velocities and positions
            r1, r2 = np.random.rand(2, n_particles, n_fac)
            
            velocities = (
                params["INERTIA"] * velocities +
                params["COGNITIVE"] * r1 * (pbest_pos - particles) +
                params["SOCIAL"] * r2 * (gbest_pos - particles)
            )
            
            particles = particles + velocities.astype(int)
            particles = np.clip(particles, 0, len(self.valid_nodes) - 1)
            
            if iteration % 5 == 0:
                logging.info(f"    Iteration {iteration:3d} | Best Score: {gbest_val:.4f}")
        
        best_nodes = [self.valid_nodes[idx] for idx in gbest_pos]
        logging.info(f"   Optimization complete | Final Score: {gbest_val:.4f}")
        
        return best_nodes
    
    def _evaluate_fitness(self, nodes):
        """Fitness = composite scores - spacing penalty"""
        
        # Base score
        base = sum(self.composite_scores.get(n, 0) for n in nodes)
        
        # Spacing penalty (prevent clustering)
        coords = np.array([[self.G.nodes[n]['x'], self.G.nodes[n]['y']] for n in nodes])
        dist_matrix = distance_matrix(coords, coords) * 111000  # to meters
        
        min_spacing = CONFIG["OPTIMIZATION_PARAMS"]["MIN_FACILITY_SPACING_M"]
        violations = (dist_matrix < min_spacing).sum() - len(nodes)  # exclude diagonal
        
        penalty = violations * 0.2
        
        return base - penalty


