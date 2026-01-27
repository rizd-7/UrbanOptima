import logging
from math import exp
import numpy as np


class PhysicsEngine:
    def __init__(self, graph):
        self.G = graph

    def apply_impedance(self):
        """
        Calculates and adds physics-based travel time to each edge in the graph.

        This method iterates through each edge, calculates the slope, applies Tobler's
        hiking function to determine a realistic walking speed, and then calculates
        the travel time. It adds 'travel_time', 'slope', and 'speed_kmh' as attributes
        to each edge.

        Returns:
            networkx.MultiDiGraph: The graph with updated edge attributes.
        """
        logging.info("=" * 80)
        logging.info("PHASE 2: PHYSICS-BASED IMPEDANCE MODELING")
        logging.info("=" * 80)

        for u, v, k, data in self.G.edges(keys=True, data=True):
            length_m = data.get('length', 10.0)
            ele_u = self.G.nodes[u].get('elevation', 0)
            ele_v = self.G.nodes[v].get('elevation', 0)

            # Avoid division by zero for very short or non-existent edges
            slope = (ele_v - ele_u) / max(length_m, 1.0)

            # Tobler's hiking function: W = 6 * exp(-3.5 * |S + 0.05|)
            speed_kmh = 6 * exp(-3.5 * abs(slope + 0.05))

            # Constrain speed to realistic walking values (0.5 km/h to 7.0 km/h)
            speed_kmh = np.clip(speed_kmh, 0.5, 7.0)

            # Convert meters and km/h to travel time in minutes
            travel_time_min = (length_m / (speed_kmh / 3.6)) / 60.0

            # Update edge attributes with calculated values
            self.G[u][v][k]['travel_time'] = travel_time_min
            self.G[u][v][k]['slope'] = slope
            self.G[u][v][k]['speed_kmh'] = speed_kmh

        # Log the average speed for verification purposes
        avg_speed = np.mean([d['speed_kmh'] for _, _, _, d in self.G.edges(keys=True, data=True)])
        logging.info(f"   Physics model applied | Avg walking speed: {avg_speed:.2f} km/h")

        return self.G