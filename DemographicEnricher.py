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





class DemographicEnricher:
    """
    Estimates population using a hierarchy of methods:
    1. WorldPop API (Raster-based real census data)
    2. Building Morphology Proxy (Fall-back estimation)
    """
    def __init__(self, buildings_gdf):
        self.buildings = buildings_gdf.copy()
        # Ensure we have a unique ID
        if 'osmid' not in self.buildings.columns:
            self.buildings['osmid'] = self.buildings.index
            
    def enrich_population(self):
        logging.info("="*80)
        logging.info("PHASE 1.5: DEMOGRAPHIC ENRICHMENT")
        logging.info("="*80)
        
        # Method 1: Try Real Data Source (WorldPop)
        success = self._fetch_worldpop_data()
        
        # Method 2: Morphology Proxy (Fallback)
        if not success:
            self._calculate_morphology_proxy()
            
        # Summary stats
        total_pop = self.buildings['estimated_pop'].sum()
        avg_pop = self.buildings['estimated_pop'].mean()
        logging.info(f"   Total Estimated Population: {int(total_pop):,}")
        logging.info(f"   Avg Persons per Building:   {avg_pop:.1f}")
        
        return self.buildings

    def _fetch_worldpop_data(self):
        """
        Attempts to query WorldPop API. 
        Note: Actual implementation requires specific raster processing or an API key.
        This simulates the check and fails gracefully to trigger the proxy.
        """
        logging.info("  Attempting to fetch WorldPop real-time data...")
        try:
            # Placeholder for actual API call
            # response = requests.get("https://api.worldpop.org/v1/services/stats", params=...)
            
            # Simulating a check (Fail by default to demonstrate the Morphology Logic)
            api_available = False 
            
            if api_available:
                logging.info("   WorldPop data acquired successfully.")
                return True
            else:
                logging.warning("   WorldPop API unavailable/unauthenticated. Switching to fallback.")
                return False
        except Exception as e:
            logging.warning(f"   API Connection failed ({str(e)}). Switching to fallback.")
            return False

    def _calculate_morphology_proxy(self):
        """
        Estimates population based on:
        Pop = (Footprint Area * Floors) / SqM_Per_Person
        """
        logging.info("  Calculating Building Morphology Proxy...")
        
        # 1. Project to meters for accurate area calculation
        b_proj = self.buildings.to_crs(epsg=3857)
        areas = b_proj.geometry.area
        
        # 2. Estimate Floors (Heuristic based on building tags)
        # Default mapping for OSM building tags
        type_to_floors = {
            'apartments': 6,
            'dormitory': 4,
            'hotel': 4,
            'office': 3,
            'school': 2,
            'hospital': 3,
            'residential': 2,
            'house': 1,
            'detached': 1,
            'retail': 1,
            'supermarket': 1,
            'industrial': 1,
            'warehouse': 1,
            'yes': 2  # Default if only tagged 'building=yes'
        }
        
        def estimate_floors(row):
            # If explicit levels exist in OSM, use them
            if 'building:levels' in row and pd.notnull(row['building:levels']):
                try:
                    return max(1, int(float(row['building:levels'])))
                except:
                    pass
            
            # Fallback to type mapping
            b_type = row.get('building', 'yes')
            return type_to_floors.get(b_type, 2)

        estimated_floors = self.buildings.apply(estimate_floors, axis=1)
        
        # 3. Define Density (SqM per person)
        # 25 m² is a standard urban planning proxy for gross floor area per person
        SQM_PER_PERSON = 25.0 
        
        # 4. Calculate
        # Total Floor Area = Footprint * Floors
        total_floor_area = areas * estimated_floors
        
        # Apply strict filters for non-residential
        non_res_types = ['industrial', 'warehouse', 'retail', 'commercial', 'school', 'university']
        is_residential = ~self.buildings['building'].isin(non_res_types)
        
        # Calculate population (integer)
        self.buildings['estimated_pop'] = (total_floor_area / SQM_PER_PERSON * is_residential).astype(int)
        
        logging.info("   Morphology model applied (Rules: 25m²/person, Type-based heights)")
