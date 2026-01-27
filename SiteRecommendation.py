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


@dataclass
class SiteRecommendation:
    site_id: int
    node_id: any
    coords: Tuple[float, float]
    facility_type: str
    composite_score: float
    
    
    # Multi-criteria breakdown
    accessibility_score: float
    density_score: float
    terrain_score: float
    centrality_score: float
    equity_score: float
    
    # Context
    catchment_demand_sqm: int
    population_proxy: int
    nearest_competitor_m: float
    street_name: str
    slope_pct: float
    
    # Justification
    primary_rationale: str
    risk_factors: List[str]
    confidence: str
    phase: str = "Pending"
    phase_id: int = 0
    roi_pct: float = 0.0
    npv_20yr: float = 0.0
    estimated_cost_usd: float = 0.0
    priority_score: float = 0.0
    action_plan: str = ""
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}



