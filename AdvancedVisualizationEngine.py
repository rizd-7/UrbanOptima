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



class AdvancedVisualizationEngine:
    """Interactive multi-layer visualizations"""
    def __init__(self):
        self.output_dir = CONFIG["OUTPUT_DIR"]
    
    def generate_dashboard(self, G, criteria_scores, recommendations, composite_scores):
        logging.info("="*80)
        logging.info("PHASE 7: ADVANCED VISUALIZATION")
        logging.info("="*80)
        
        # 1. Interactive Map
        self._create_interactive_map(G, criteria_scores, recommendations, composite_scores)
        
        # 2. Decision Matrix
        self._create_decision_matrix(recommendations)
        
        # 3. Criteria Heatmaps
        self._create_criteria_comparison(criteria_scores)
        
        logging.info(f"   All visualizations saved to {self.output_dir}/")
    
    def _create_interactive_map(self, G, criteria_scores, recommendations, composite_scores):
        logging.info("  Creating interactive map...")
        
        start_node = list(G.nodes())[0]
        center = [G.nodes[start_node]['y'], G.nodes[start_node]['x']]
        
        m = folium.Map(
            location=center,
            zoom_start=15,
            tiles='CartoDB positron'
        )
        
        # Layer 1: Composite Score Heatmap
        heat_data = [
            [d['y'], d['x'], composite_scores.get(n, 0) * 1000]
            for n, d in G.nodes(data=True)
            if composite_scores.get(n, 0) > 0.3
        ]
        
        heat_layer = plugins.HeatMap(
            heat_data,
            name='Suitability Score',
            radius=20,
            blur=25,
            gradient={
                0.0: 'blue',
                0.5: 'yellow',
                0.8: 'orange',
                1.0: 'red'
            }
        )
        heat_layer.add_to(m)
        
        # Layer 2: Accessibility Gap
        access_heat = [
            [d['y'], d['x'], criteria_scores['accessibility_gap'].get(n, 0) * 800]
            for n, d in G.nodes(data=True)
            if criteria_scores['accessibility_gap'].get(n, 0) > 0.4
        ]
        
        plugins.HeatMap(
            access_heat,
            name='Accessibility Gaps',
            radius=18,
            blur=20,
            gradient={0.0: 'green', 0.5: 'yellow', 1.0: 'red'}
        ).add_to(m)
        
        # Layer 3: Recommendations
        for rec in recommendations:
            # Color by confidence
            color_map = {"HIGH": "darkgreen", "MEDIUM": "orange", "LOW": "red"}
            marker_color = color_map.get(rec.confidence, "gray")
            
            # Detailed popup
            popup_html = f"""
            <div style='font-family: Arial; width: 400px; padding: 10px;'>
                <h3 style='color: {marker_color}; margin: 0 0 10px 0; border-bottom: 3px solid {marker_color}'>
                    {rec.facility_type}
                </h3>
                
                <p style='margin: 5px 0;'><b>📍 Location:</b> {rec.street_name}</p>
                <p style='margin: 5px 0;'><b>🎯 Confidence:</b> <span style='color: {marker_color}; font-weight: bold'>{rec.confidence}</span></p>
                <p style='margin: 5px 0;'><b>📊 Composite Score:</b> {rec.composite_score:.3f}</p>
                
                <hr style='margin: 10px 0;'>
                
                <h4 style='margin: 5px 0;'>Multi-Criteria Breakdown:</h4>
                <div style='font-size: 12px;'>
                    <div style='margin: 3px 0;'>
                        <div style='display: inline-block; width: 150px;'>Accessibility Gap:</div>
                        <div style='display: inline-block; width: 100px; background: #ddd; height: 15px;'>
                            <div style='width: {rec.accessibility_score*100}%; background: #e74c3c; height: 15px;'></div>
                        </div>
                        <span style='margin-left: 5px;'>{rec.accessibility_score:.2f}</span>
                    </div>
                    <div style='margin: 3px 0;'>
                        <div style='display: inline-block; width: 150px;'>Population Density:</div>
                        <div style='display: inline-block; width: 100px; background: #ddd; height: 15px;'>
                            <div style='width: {rec.density_score*100}%; background: #3498db; height: 15px;'></div>
                        </div>
                        <span style='margin-left: 5px;'>{rec.density_score:.2f}</span>
                    </div>
                    <div style='margin: 3px 0;'>
                        <div style='display: inline-block; width: 150px;'>Terrain Suitability:</div>
                        <div style='display: inline-block; width: 100px; background: #ddd; height: 15px;'>
                            <div style='width: {rec.terrain_score*100}%; background: #2ecc71; height: 15px;'></div>
                        </div>
                        <span style='margin-left: 5px;'>{rec.terrain_score:.2f}</span>
                    </div>
                    <div style='margin: 3px 0;'>
                        <div style='display: inline-block; width: 150px;'>Network Centrality:</div>
                        <div style='display: inline-block; width: 100px; background: #ddd; height: 15px;'>
                            <div style='width: {rec.centrality_score*100}%; background: #f39c12; height: 15px;'></div>
                        </div>
                        <span style='margin-left: 5px;'>{rec.centrality_score:.2f}</span>
                    </div>
                    <div style='margin: 3px 0;'>
                        <div style='display: inline-block; width: 150px;'>Spatial Equity:</div>
                        <div style='display: inline-block; width: 100px; background: #ddd; height: 15px;'>
                            <div style='width: {rec.equity_score*100}%; background: #9b59b6; height: 15px;'></div>
                        </div>
                        <span style='margin-left: 5px;'>{rec.equity_score:.2f}</span>
                    </div>
                </div>
                
                <hr style='margin: 10px 0;'>
                
                <h4 style='margin: 5px 0;'>Impact:</h4>
                <p style='margin: 3px 0; font-size: 12px;'>• Built Area: {rec.catchment_demand_sqm:,} m²</p>
                <p style='margin: 3px 0; font-size: 12px;'>• Population: ~{rec.population_proxy:,} people</p>
                <p style='margin: 3px 0; font-size: 12px;'>• Nearest Facility: {int(rec.nearest_competitor_m) if rec.nearest_competitor_m else 'N/A'} m</p>
                
                <hr style='margin: 10px 0;'>
                
                <h4 style='margin: 5px 0;'>Justification:</h4>
                <p style='margin: 3px 0; font-size: 11px; font-style: italic;'>{rec.primary_rationale}</p>
                
                <h4 style='margin: 8px 0 3px 0;'>Risks:</h4>
                <ul style='margin: 0; padding-left: 20px; font-size: 11px;'>
                    {''.join([f"<li>{risk}</li>" for risk in rec.risk_factors])}
                </ul>
            </div>
            """
            
            folium.Marker(
                location=rec.coords,
                popup=folium.Popup(popup_html, max_width=450),
                tooltip=f"Site #{rec.site_id}: {rec.facility_type} ({rec.confidence} confidence)",
                icon=folium.Icon(
                    color=marker_color,
                    icon='hospital-o',
                    prefix='fa'
                )
            ).add_to(m)
            
            # Catchment circle
            folium.Circle(
                location=rec.coords,
                radius=1500,  # ~15 min walk
                color=marker_color,
                fill=True,
                fillColor=marker_color,
                fillOpacity=0.1,
                weight=2,
                dashArray='5, 5'
            ).add_to(m)
        
        # Layer control
        folium.LayerControl().add_to(m)
        
        # Save
        map_path = os.path.join(self.output_dir, 'interactive_map.html')
        m.save(map_path)
        logging.info(f"     Map: {map_path}")
    
    def _create_decision_matrix(self, recommendations):
        logging.info("  Creating decision matrix...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multi-Criteria Decision Matrix', fontsize=16, fontweight='bold')
        
        # Extract data
        sites = [f"Site {r.site_id}" for r in recommendations]
        criteria = ['Accessibility', 'Density', 'Terrain', 'Centrality', 'Equity']
        
        data = np.array([
            [r.accessibility_score, r.density_score, r.terrain_score, 
             r.centrality_score, r.equity_score]
            for r in recommendations
        ])
        
        # 1. Radar chart
        ax = axes[0, 0]
        angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, 1, projection='polar')
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        for i, rec in enumerate(recommendations):
            values = [rec.accessibility_score, rec.density_score, rec.terrain_score,
                     rec.centrality_score, rec.equity_score]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=sites[i], color=colors[i % 3])
            ax.fill(angles, values, alpha=0.15, color=colors[i % 3])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria)
        ax.set_ylim(0, 1)
        ax.set_title('Criteria Profiles', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        # 2. Bar chart comparison
        ax = axes[0, 1]
        x = np.arange(len(sites))
        width = 0.15
        
        for i, criterion in enumerate(criteria):
            offset = width * (i - 2)
            ax.bar(x + offset, data[:, i], width, label=criterion)
        
        ax.set_xlabel('Sites')
        ax.set_ylabel('Score')
        ax.set_title('Score Breakdown by Criterion', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sites)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Composite scores
        ax = axes[1, 0]
        composite = [r.composite_score for r in recommendations]
        confidence_colors = {'HIGH': '#2ecc71', 'MEDIUM': '#f39c12', 'LOW': '#e74c3c'}
        colors_list = [confidence_colors[r.confidence] for r in recommendations]
        
        bars = ax.barh(sites, composite, color=colors_list)
        ax.set_xlabel('Composite Score')
        ax.set_title('Final Composite Scores (Colored by Confidence)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # 4. Impact metrics
        ax = axes[1, 1]
        impact_data = np.array([
            [r.population_proxy/1000, r.catchment_demand_sqm/10000, 
             (r.nearest_competitor_m if r.nearest_competitor_m else 0)/100]
            for r in recommendations
        ])
        
        x = np.arange(len(sites))
        width = 0.25
        
        ax.bar(x - width, impact_data[:, 0], width, label='Pop. (k)', color='#3498db')
        ax.bar(x, impact_data[:, 1], width, label='Area (10k m²)', color='#e74c3c')
        ax.bar(x + width, impact_data[:, 2], width, label='Gap (100m)', color='#2ecc71')
        
        ax.set_ylabel('Scaled Values')
        ax.set_title('Impact Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sites)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        matrix_path = os.path.join(self.output_dir, 'decision_matrix.png')
        plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"     Matrix: {matrix_path}")
    
    def _create_criteria_comparison(self, criteria_scores):
        logging.info("  Creating criteria heatmaps...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Spatial Distribution of Decision Criteria', fontsize=16, fontweight='bold')
        
        criteria_names = list(criteria_scores.keys())
        
        for idx, criterion in enumerate(criteria_names):
            ax = axes[idx // 3, idx % 3]
            
            scores = list(criteria_scores[criterion].values())
            
            ax.hist(scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
            ax.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.3f}')
            
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.set_title(criterion.replace('_', ' ').title(), fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        
        hist_path = os.path.join(self.output_dir, 'criteria_distributions.png')
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"     Distributions: {hist_path}")

