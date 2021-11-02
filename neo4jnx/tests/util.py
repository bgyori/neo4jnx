"""Utility functions for neo4jnx tests

The purpose is to have a small but accurate networkx graph that is
translated to a neo4jnx graph and then test the neo4jnx graph interface
against the networkx graph.
"""
import pickle
import json
from neo4jnx import Neo4jDiGraph

try:
    n4_g = Neo4jDiGraph(
        neo4j_url="neo4j://localhost:7688",
        neo4j_auth=("neo4j", "admin"),
        property_loaders={
            "statements": json.loads,
            "belief": float,
            "weight": float,
            "z_score": float,
            "corr_weight": float,
        },
    )
except Exception as e:
    print("Is there a neo4j instance running on neo4j://localhost:7688?")
    raise e

nx_g_file = (
    "/home/klas/repos/depmap_analysis/input_data/db/2021-08-09"
    "/graphs/indranet_dir_small_graph.pkl"
)
with open(nx_g_file, "rb") as f:
    nx_g = pickle.load(f)
