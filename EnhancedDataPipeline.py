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


class EnhancedDataPipeline:
    """Enriched data ingestion with validation"""
    def __init__(self):
        self.place = CONFIG["TARGET_LOCATION"]
        self.G = None
        self.nodes_gdf = None
        self.buildings = None
        self.facilities = None
        
    def execute(self):
        logging.info("="*80)
        logging.info("PHASE 1: DATA ACQUISITION & VALIDATION")
        logging.info("="*80)
        
        # Network
        logging.info(f"Downloading pedestrian network: {self.place}")
        self.G = ox.graph_from_place(self.place, network_type=CONFIG["NETWORK_TYPE"])
        self.nodes_gdf, _ = ox.graph_to_gdfs(self.G)
        logging.info(f"   Network: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        
        # Buildings
        logging.info("Downloading building footprints...")
        self.buildings = ox.features_from_place(self.place, tags={'building': True})
        self.buildings = self.buildings[self.buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        logging.info(f"   Buildings: {len(self.buildings)} valid polygons")
        
        # Facilities
        logging.info("Downloading healthcare facilities...")
        try:
            tags = {'amenity': ['pharmacy', 'clinic', 'doctors', 'hospital']}
            self.facilities = ox.features_from_place(self.place, tags=tags)
            self.facilities['centroid'] = self.facilities.geometry.centroid
            logging.info(f"   Existing facilities: {len(self.facilities)}")
        except:
            logging.warning("   No existing facilities found (greenfield scenario)")
            self.facilities = gpd.GeoDataFrame()
        
        # Elevation
        self._enrich_elevation_batch()
        
        # Validation
        self._validate_data()
        
        return self.G, self.buildings, self.facilities
    
    def _enrich_elevation_batch(self):
        logging.info("Querying DEM (Open-Elevation API)...")
        nodes = list(self.G.nodes(data=True))
        chunk_size = 100
        enriched = 0
        
        for i in range(0, len(nodes), chunk_size):
            chunk = nodes[i:i+chunk_size]
            payload = {"locations": [{"latitude": d['y'], "longitude": d['x']} for _, d in chunk]}
            
            try:
                resp = requests.post(
                    "https://api.open-elevation.com/api/v1/lookup",
                    json=payload,
                    timeout=10
                )
                if resp.status_code == 200:
                    results = resp.json()['results']
                    for j, res in enumerate(results):
                        self.G.nodes[chunk[j][0]]['elevation'] = res['elevation']
                        enriched += 1
                else:
                    # Fallback to zero elevation
                    for node_id, _ in chunk:
                        self.G.nodes[node_id]['elevation'] = 0
            except Exception as e:
                logging.warning(f"Elevation batch {i//chunk_size} failed: {e}")
                for node_id, _ in chunk:
                    self.G.nodes[node_id]['elevation'] = 0
            
            time.sleep(0.5)  # Rate limiting
        
        logging.info(f"   Elevation data: {enriched}/{len(nodes)} nodes")
    
    def _validate_data(self):
        """Quality checks"""
        issues = []
        
        # Check for disconnected components
        if not nx.is_strongly_connected(self.G):
            components = list(nx.strongly_connected_components(self.G))
            logging.warning(f"   Network has {len(components)} disconnected components")
            # Keep largest component
            largest = max(components, key=len)
            self.G = self.G.subgraph(largest).copy()
            logging.info(f"   Using largest component: {len(self.G.nodes)} nodes")
        
        # Check building data coverage
        if len(self.buildings) < 50:
            issues.append("Low building count - results may be unreliable")
        
        if issues:
            logging.warning("  Data quality issues detected:")
            for issue in issues:
                logging.warning(f"    - {issue}")