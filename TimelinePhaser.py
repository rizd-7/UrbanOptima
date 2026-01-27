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
from SiteRecommendation import SiteRecommendation

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





class TimelinePhaser:
    """
    Strategic Implementation Planner.
    Sorts recommendations into phases based on:
    1. Urgency (Composite Score)
    2. Financial Viability (NPV & ROI)
    """
    def __init__(self):
        # Economic Assumptions
        self.CONST_COST_PER_SQM = 1200  # USD (Construction)
        self.LAND_COST_PER_SQM = 400    # USD (Acquisition)
        self.VALUE_PER_PATIENT_YR = 150 # USD (Social/Economic benefit)
        self.DISCOUNT_RATE = 0.05       # 5% annual discount
        self.TERM_YEARS = 20

    def generate_timeline(self, recommendations: List[SiteRecommendation]):
        logging.info("="*80)
        logging.info("PHASE 7: TIMELINE & PHASING STRATEGY")
        logging.info("="*80)
        
        if not recommendations:
            return []

        # 1. Calculate Financials for everyff site
        for rec in recommendations:
            self._calculate_financial_viability(rec)

        # 2. Sort by "Implementation Priority Score"
        # Formula: 60% Urgency (Composite) + 40% Financial (Normalized ROI)
        max_roi = max(r.roi_pct for r in recommendations) if recommendations else 1
        
        for rec in recommendations:
            norm_roi = max(0, rec.roi_pct) / max_roi if max_roi > 0 else 0
            rec.priority_score = (rec.composite_score * 0.6) + (norm_roi * 0.4)

        # Sort descending
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)

        # 3. Assign Phases (Tertile split)
        n = len(recommendations)
        # Simple logic: Top 33% = Phase 1, Next 33% = Phase 2, Rest = Phase 3
        # Adjust for small numbers (if n=3, 1 in each)
        p1_cutoff = np.ceil(n / 3)
        p2_cutoff = np.ceil(2 * n / 3)

        for i, rec in enumerate(recommendations):
            if i < p1_cutoff:
                rec.phase = "Phase 1 (Year 1)"
                rec.phase_id = 1
                rec.action_plan = "Immediate Land Acquisition & Design"
            elif i < p2_cutoff:
                rec.phase = "Phase 2 (Years 2-3)"
                rec.phase_id = 2
                rec.action_plan = "Secure Funding & Environmental Permitting"
            else:
                rec.phase = "Phase 3 (Years 4-5)"
                rec.phase_id = 3
                rec.action_plan = "Reserve Land for Future Expansion"

        # 4. Generate Visual Summary
        self._plot_timeline(recommendations)
        self._log_summary(recommendations)

        return recommendations

    def _calculate_financial_viability(self, rec: SiteRecommendation):
        """
        Estimates CAPEX, OPEX, and ROI based on facility type and terrain.
        """
        # A. Determine Size based on Type
        if "Hospital" in rec.facility_type:
            size_sqm = 5000
            staff_count = 150
        elif "Clinic" in rec.facility_type:
            size_sqm = 1200
            staff_count = 25
        else: # Pharmacy
            size_sqm = 150
            staff_count = 5

        # B. CAPEX (Capital Expenditure)
        # Penalty for steep slopes (site prep cost)
        slope_penalty = 1 + (rec.slope_pct / 100) * 0.5 
        construction_cost = size_sqm * self.CONST_COST_PER_SQM * slope_penalty
        land_cost = size_sqm * self.LAND_COST_PER_SQM
        total_capex = construction_cost + land_cost

        # C. Annual Benefit (Proxy)
        # Population * Utilization Rate * Value
        annual_benefit = rec.population_proxy * self.VALUE_PER_PATIENT_YR

        # D. Net Present Value (NPV)
        # Sum [ Benefit / (1+r)^t ] - CAPEX
        npv_benefits = 0
        for t in range(1, self.TERM_YEARS + 1):
            npv_benefits += annual_benefit / ((1 + self.DISCOUNT_RATE) ** t)
        
        rec.npv_20yr = npv_benefits - total_capex
        
        # E. ROI
        rec.roi_pct = (rec.npv_20yr / total_capex) * 100
        rec.estimated_cost_usd = total_capex

    def _plot_timeline(self, recommendations):
        """Generates a Gantt-style chart for implementation"""
        try:
            plt.figure(figsize=(10, 6))
            
            phases = [r.phase_id for r in recommendations]
            scores = [r.priority_score for r in recommendations]
            labels = [f"Site {r.site_id}\n{r.facility_type}" for r in recommendations]
            colors = ['#2ecc71' if p==1 else '#f39c12' if p==2 else '#95a5a6' for p in phases]

            # Create horizontal bar chart
            plt.barh(labels, scores, color=colors)
            
            # Decoration
            plt.xlabel("Implementation Priority Score")
            plt.title("Phased Implementation Schedule", fontweight='bold')
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            
            # Custom Legend
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='#2ecc71', lw=4),
                Line2D([0], [0], color='#f39c12', lw=4),
                Line2D([0], [0], color='#95a5a6', lw=4)
            ]
            plt.legend(custom_lines, ['Phase 1 (Year 1)', 'Phase 2 (Yrs 2-3)', 'Phase 3 (Yrs 4-5)'])

            output_path = os.path.join(CONFIG["OUTPUT_DIR"], "implementation_timeline.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            logging.info(f"   Timeline visualization saved: {output_path}")
            
        except Exception as e:
            logging.warning(f"   couldnt not generate timeline plot: {e}")

    def _log_summary(self, recs):
        logging.info("\n  Phasing Strategy Summary:")
        logging.info(f"    {'PHASE':<20} | {'SITE':<10} | {'TYPE':<20} | {'ROI':<8} | {'ACTION'}")
        logging.info("-" * 90)
        for r in recs:
            logging.info(f"    {r.phase:<20} | #{r.site_id:<9} | {r.facility_type[:18]:<20} | {r.roi_pct:4.0f}%   | {r.action_plan}")