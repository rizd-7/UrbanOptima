import logging
import networkx as nx
import numpy as np
import osmnx as ox
from scipy.spatial import cKDTree
from config import CONFIG

class MultiCriteriaAnalyzer:
    def __init__(self, graph, buildings, facilities):
        self.G = graph
        self.buildings = buildings
        self.facilities = facilities
        self.nodes_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=False)
        
        node_coords = np.array(list(zip(self.nodes_gdf.geometry.x, self.nodes_gdf.geometry.y)))
        self.node_tree = cKDTree(node_coords)
        
        if not facilities.empty:
            fac_coords = np.array([(p.x, p.y) for p in facilities.centroid])
            self.fac_tree = cKDTree(fac_coords)
        else:
            self.fac_tree = None
        
        self.criteria_scores = {}
        
    def analyze_all_criteria(self):
        logging.info("="*80)
        logging.info("PHASE 3: MULTI-CRITERIA SPATIAL ANALYSIS")
        logging.info("="*80)
        
        self.criteria_scores['accessibility_gap'] = self._compute_accessibility_gap()
        self.criteria_scores['population_density'] = self._compute_density_proxy()
        self.criteria_scores['terrain_suitability'] = self._compute_terrain_scores()
        self.criteria_scores['network_centrality'] = self._compute_centrality()
        self.criteria_scores['spatial_equity'] = self._compute_equity_scores()
        
        for criterion, scores in self.criteria_scores.items():
            max_val = max(scores.values()) if scores else 1
            if max_val > 0:
                self.criteria_scores[criterion] = {k: v/max_val for k, v in scores.items()}
        
        composite = self._compute_composite_score()
        self._log_criteria_stats()
        
        return composite, self.criteria_scores
    
    def _compute_accessibility_gap(self):
        logging.info("  [1/5] Computing accessibility gaps (2SFCA inversion)...")
        
        b_proj = self.buildings.to_crs(epsg=3857)
        areas = b_proj.geometry.area.values
        centroids = np.array([(b.centroid.x, b.centroid.y) for b in self.buildings.to_crs(epsg=4326).geometry])
        _, indices = self.node_tree.query(centroids)
        
        node_demand = {n: 0.0 for n in self.G.nodes()}
        node_ids = self.nodes_gdf.index
        for i, idx in enumerate(indices):
            node_demand[node_ids[idx]] += areas[i]
        
        if self.facilities.empty:
            return {n: 1.0 for n in self.G.nodes()}
        
        f_coords = np.array([(p.x, p.y) for p in self.facilities.centroid])
        _, f_indices = self.node_tree.query(f_coords)
        f_nodes = self.nodes_gdf.index[f_indices].unique()
        
        catchment = CONFIG["CATCHMENT_THRESHOLD"]
        beta = 0.1
        
        ratios = {}
        for f in f_nodes:
            subgraph = nx.ego_graph(self.G, f, radius=catchment, distance='travel_time')
            weighted_demand = 0
            for n in subgraph.nodes():
                try:
                    dist = nx.shortest_path_length(self.G, f, n, weight='travel_time')
                    weight = np.exp(-beta * (dist/catchment)**2)
                    weighted_demand += node_demand.get(n, 0) * weight
                except:
                    pass
            ratios[f] = 1.0 / weighted_demand if weighted_demand > 0 else 0
        
        access_scores = {n: 0.0 for n in self.G.nodes()}
        for f, R in ratios.items():
            if R == 0: continue
            subgraph = nx.ego_graph(self.G, f, radius=catchment, distance='travel_time')
            for n in subgraph.nodes():
                try:
                    dist = nx.shortest_path_length(self.G, f, n, weight='travel_time')
                    weight = np.exp(-beta * (dist/catchment)**2)
                    access_scores[n] += R * weight
                except:
                    pass
        
        max_access = max(access_scores.values()) if access_scores else 1
        gaps = {n: 1.0 - (score / max_access) for n, score in access_scores.items()}
        
        return gaps
    
    def _compute_density_proxy(self):
        logging.info("  [2/5] Computing population density proxies...")
        
        node_density = {n: 0.0 for n in self.G.nodes()}
        
        b_proj = self.buildings.to_crs(epsg=3857)
        areas = b_proj.geometry.area.values
        centroids = np.array([(b.centroid.x, b.centroid.y) for b in self.buildings.to_crs(epsg=4326).geometry])
        
        radius_deg = 0.002
        
        for node_id, data in self.G.nodes(data=True):
            node_pt = np.array([[data['x'], data['y']]])
            dists = np.sqrt(((centroids - node_pt)**2).sum(axis=1))
            
            weights = np.exp(-(dists / radius_deg)**2)
            node_density[node_id] = (weights * areas).sum()
        
        return node_density
    

    
    def _compute_terrain_scores(self):
        logging.info("  [3/5] Evaluating terrain suitability...")
        
        scores = {}
        max_slope = CONFIG["ZONING_CONSTRAINTS"]["max_slope"]
        
        for n in self.G.nodes():
            edges = list(self.G.edges(n, data=True))
            if not edges:
                scores[n] = 0
                continue
            
            slopes = [abs(d.get('slope', 0)) for _, _, d in edges]
            avg_slope = np.mean(slopes)
            
            if avg_slope > max_slope:
                scores[n] = 0
            else:
                scores[n] = 1.0 - (avg_slope / max_slope)
        
        return scores
    

    
    def _compute_centrality(self):
        logging.info("  [4/5] Computing network centrality...")
        
        degree_cent = nx.degree_centrality(self.G)
        
        return degree_cent
    
    def _compute_equity_scores(self):
        logging.info("  [5/5] Assessing spatial equity distribution...")
        
        if self.fac_tree is None:
            return {n: 1.0 for n in self.G.nodes()}
        
        scores = {}
        for node_id, data in self.G.nodes(data=True):
            dist, _ = self.fac_tree.query([data['x'], data['y']])
            dist_m = dist * 111000
            scores[node_id] = 1 / (1 + np.exp(-(dist_m - 1000) / 500))
        
        return scores
    
    def _compute_composite_score(self):
        logging.info("  Computing weighted composite scores...")
        
        weights = CONFIG["MULTI_CRITERIA_WEIGHTS"]
        composite = {n: 0.0 for n in self.G.nodes()}
        
        for criterion, weight in weights.items():
            for node in self.G.nodes():
                composite[node] += self.criteria_scores[criterion].get(node, 0) * weight
        
        return composite
    
    def _log_criteria_stats(self):
        logging.info("\n  Criteria Score Statistics:")
        for criterion, scores in self.criteria_scores.items():
            vals = list(scores.values())
            logging.info(f"    {criterion:25s} | Mean: {np.mean(vals):.3f} | Std: {np.std(vals):.3f}")