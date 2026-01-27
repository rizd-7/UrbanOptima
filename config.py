CONFIG = {
    "TARGET_LOCATION": "Casbah, Algiers, Algeria",
    "NETWORK_TYPE": "walk",
    "CATCHMENT_THRESHOLD": 15,
    
    "MULTI_CRITERIA_WEIGHTS": {
        "accessibility_gap": 0.35,      # Where is access worst?
        "population_density": 0.25,     # Where are people?
        "terrain_suitability": 0.15,    # Can we build here?
        "network_centrality": 0.15,     # Transport connectivity
        "spatial_equity": 0.10          # Fair distribution
    },
    
    "OPTIMIZATION_PARAMS": {
        "N_NEW_FACILITIES": 3,
        "MIN_FACILITY_SPACING_M": 500,  # Prevent clustering
        "SWARM_PARTICLES": 30,
        "SWARM_ITERATIONS": 25,
        "INERTIA": 0.6,
        "COGNITIVE": 1.8,
        "SOCIAL": 1.8
    },
    
    "CLASSIFICATION_RULES": {
        "HOSPITAL_THRESHOLD": 50000,
        "CLINIC_THRESHOLD": 20000,
        "MIN_JUSTIFICATION_SCORE": 0.4  # Quality control
    },
    
    "ZONING_CONSTRAINTS": {
        "excluded_highways": ['motorway', 'trunk', 'motorway_link', 'trunk_link'],
        "excluded_features": ['railway', 'tunnel'],
        "max_slope": 0.15,
        "min_road_width_m": 4.0
    },
    
    "OUTPUT_DIR": "urban_output",
    "LOG_FILE": "urban_intelligence_v2.log"
}
