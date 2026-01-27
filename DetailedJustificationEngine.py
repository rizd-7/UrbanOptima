import json
import logging
import os
import time
from typing import List

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from config import CONFIG
from SiteRecommendation import SiteRecommendation


class DetailedJustificationEngine:
    """Rich contextual analysis for each recommendation"""

    def __init__(self, graph, criteria_scores, facilities):
        self.G = graph
        self.criteria_scores = criteria_scores
        self.facilities = facilities

        if not facilities.empty:
            fac_coords = np.array([(p.x, p.y) for p in facilities.centroid])
            self.fac_tree = cKDTree(fac_coords)
        else:
            self.fac_tree = None

    def generate_recommendations(self, selected_nodes, node_demand) -> List[SiteRecommendation]:
        logging.info("=" * 80)
        logging.info("PHASE 6: DECISION SUPPORT & JUSTIFICATION")
        logging.info("=" * 80)

        recommendations = []

        for i, node in enumerate(selected_nodes):
            catchment = nx.single_source_dijkstra_path_length(
                self.G, node, cutoff=CONFIG["CATCHMENT_THRESHOLD"], weight='travel_time'
            )
            total_sqm = sum(node_demand.get(n, 0) for n in catchment)
            pop_proxy = int(total_sqm / 25)

            thresholds = CONFIG["CLASSIFICATION_RULES"]
            if total_sqm >= thresholds["HOSPITAL_THRESHOLD"]:
                fac_type = "Regional Hospital"
            elif total_sqm >= thresholds["CLINIC_THRESHOLD"]:
                fac_type = "Primary Care Clinic"
            else:
                fac_type = "Community Pharmacy"

            access_score = self.criteria_scores['accessibility_gap'].get(node, 0)
            density_score = self.criteria_scores['population_density'].get(node, 0)
            terrain_score = self.criteria_scores['terrain_suitability'].get(node, 0)
            central_score = self.criteria_scores['network_centrality'].get(node, 0)
            equity_score = self.criteria_scores['spatial_equity'].get(node, 0)

            composite = (
                access_score * CONFIG["MULTI_CRITERIA_WEIGHTS"]["accessibility_gap"] +
                density_score * CONFIG["MULTI_CRITERIA_WEIGHTS"]["population_density"] +
                terrain_score * CONFIG["MULTI_CRITERIA_WEIGHTS"]["terrain_suitability"] +
                central_score * CONFIG["MULTI_CRITERIA_WEIGHTS"]["network_centrality"] +
                equity_score * CONFIG["MULTI_CRITERIA_WEIGHTS"]["spatial_equity"]
            )

            edges = list(self.G.edges(node, data=True))
            street = edges[0][2].get('name', 'Unnamed Street') if edges else "Unknown Location"

            slopes = [abs(d.get('slope', 0)) for _, _, d in edges]
            avg_slope = np.mean(slopes) * 100 if slopes else 0

            nearest_dist = None
            if self.fac_tree:
                dist, _ = self.fac_tree.query([self.G.nodes[node]['x'], self.G.nodes[node]['y']])
                nearest_dist = dist * 111000

            rationale = self._generate_rationale(
                access_score, density_score, central_score, equity_score
            )
            risks = self._assess_risks(terrain_score, nearest_dist, total_sqm)
            confidence = self._compute_confidence(composite, terrain_score)

            rec = SiteRecommendation(
                site_id=i + 1,
                node_id=node,
                coords=(self.G.nodes[node]['y'], self.G.nodes[node]['x']),
                facility_type=fac_type,
                composite_score=composite,
                accessibility_score=access_score,
                density_score=density_score,
                terrain_score=terrain_score,
                centrality_score=central_score,
                equity_score=equity_score,
                catchment_demand_sqm=int(total_sqm),
                population_proxy=pop_proxy,
                nearest_competitor_m=nearest_dist,
                street_name=street,
                slope_pct=avg_slope,
                primary_rationale=rationale,
                risk_factors=risks,
                confidence=confidence
            )

            recommendations.append(rec)
            self._print_recommendation(rec)

        self._export_json(recommendations)
        return recommendations

    def _generate_rationale(self, access, density, central, equity) -> str:
        reasons = []
        scores = {
            'accessibility_gap': access,
            'population_density': density,
            'network_centrality': central,
            'spatial_equity': equity
        }
        primary = max(scores.items(), key=lambda x: x[1])

        if primary[0] == 'accessibility_gap':
            reasons.append("Addresses lowest healthcare access in the area")
        elif primary[0] == 'population_density':
            reasons.append("Serves the highest population concentration")
        elif primary[0] == 'network_centrality':
            reasons.append("Located at a major transport hub")
        elif primary[0] == 'spatial_equity':
            reasons.append("Fills a critical service gap in an underserved neighborhood")

        if density > 0.7 and primary[0] != 'population_density':
            reasons.append("High-density residential area")
        if central > 0.7 and primary[0] != 'network_centrality':
            reasons.append("Excellent pedestrian connectivity")
        if equity > 0.7 and primary[0] != 'spatial_equity':
            reasons.append("Improves equity for underserved populations")

        return " | ".join(reasons)

    def _assess_risks(self, terrain, nearest_dist, demand) -> List[str]:
        risks = []
        if terrain < 0.5:
            risks.append("Challenging terrain may require site preparation.")
        if nearest_dist and nearest_dist < 300:
            risks.append(f"Competitor within {int(nearest_dist)}m poses market saturation risk.")
        if demand < CONFIG["CLASSIFICATION_RULES"]["CLINIC_THRESHOLD"]:
            risks.append("Lower demand zone may require phased implementation.")
        if not risks:
            risks.append("No significant implementation risks identified.")
        return risks

    def _compute_confidence(self, composite, terrain) -> str:
        if composite > 0.75 and terrain > 0.7:
            return "HIGH"
        elif composite > 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def _print_recommendation(self, rec: SiteRecommendation):
        print("\n" + "=" * 90)
        print(f"RECOMMENDATION #{rec.site_id}: {rec.facility_type.upper()}")
        print("=" * 90)
        print(f"Location:       {rec.street_name}")
        print(f"Coordinates:    {rec.coords[0]:.6f}, {rec.coords[1]:.6f}")
        print(f"Confidence:     {rec.confidence}")
        print(f"\nComposite Score: {rec.composite_score:.3f}/1.000")
        print(
            f"  ├─ Accessibility Gap:    {rec.accessibility_score:.3f} (weight: {CONFIG['MULTI_CRITERIA_WEIGHTS']['accessibility_gap']:.2f})")
        print(
            f"  ├─ Population Density:   {rec.density_score:.3f} (weight: {CONFIG['MULTI_CRITERIA_WEIGHTS']['population_density']:.2f})")
        print(
            f"  ├─ Terrain Suitability:  {rec.terrain_score:.3f} (weight: {CONFIG['MULTI_CRITERIA_WEIGHTS']['terrain_suitability']:.2f})")
        print(
            f"  ├─ Network Centrality:   {rec.centrality_score:.3f} (weight: {CONFIG['MULTI_CRITERIA_WEIGHTS']['network_centrality']:.2f})")
        print(
            f"  └─ Spatial Equity:       {rec.equity_score:.3f} (weight: {CONFIG['MULTI_CRITERIA_WEIGHTS']['spatial_equity']:.2f})")
        print(f"\nImpact Analysis:")
        print(f"  • Catchment Demand:      {rec.catchment_demand_sqm:,} m² built area")
        print(f"  • Population Estimate:   ~{rec.population_proxy:,} residents")
        print(
            f"  • Nearest Competitor:    {int(rec.nearest_competitor_m) if rec.nearest_competitor_m else 'N/A'} meters")
        print(f"  • Terrain Slope:         {rec.slope_pct:.1f}%")
        print(f"\nJustification:")
        print(f"  {rec.primary_rationale}")
        print(f"\nRisk Assessment:")
        for risk in rec.risk_factors:
            print(f"  - {risk}")

    def _export_json(self, recommendations: List[SiteRecommendation]):
        output_path = os.path.join(CONFIG["OUTPUT_DIR"], "recommendations.json")
        data = {
            "metadata": {
                "location": CONFIG["TARGET_LOCATION"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_recommendations": len(recommendations)
            },
            "recommendations": [rec.to_dict() for rec in recommendations]
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"\nRecommendations exported to {output_path}")