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


class FeatureEngineer:
    """Extract graph features for ML"""
    def __init__(self, graph):
        self.G = graph
    
    def extract(self):
        logging.info("  Extracting topological features...")
        
        data = []
        node_ids = []
        
        degree_dict = dict(self.G.degree())
        
        for n, d in self.G.nodes(data=True):
            # Topology
            deg = degree_dict.get(n, 0)
            
            # Physics
            edges = list(self.G.edges(n, data=True))
            slopes = [abs(e[2].get('slope', 0)) for e in edges]
            avg_slope = np.mean(slopes) if slopes else 0
            max_slope = np.max(slopes) if slopes else 0
            
            # Elevation
            elevation = d.get('elevation', 0)
            
            # Clustering
            try:
                clustering = nx.clustering(self.G, n)
            except:
                clustering = 0
            
            # Betweenness (sampled for performance)
            data.append([deg, avg_slope, max_slope, elevation, clustering])
            node_ids.append(n)
        
        return pd.DataFrame(
            data,
            columns=['degree', 'avg_slope', 'max_slope', 'elevation', 'clustering'],
            index=node_ids
        )