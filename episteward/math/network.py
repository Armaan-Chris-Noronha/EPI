"""
Contact-graph network epidemiology for inter-ward/hospital transmission.

The hospital network is modelled as a weighted directed graph:
    nodes  = wards / hospitals
    edges  = patient transfer probability per 24 h (from hospital_network.json)

Transmission probability from node i to node j per timestep:
    P(i→j) = β * w(i,j) * I_i(t) * (1 - immune_j)

    β = 0.15  for ESBL (higher community transmissibility)
    β = 0.08  for CRK  (more nosocomial, lower community spread)
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np

BETA = {
    "ESBL": 0.15,
    "CRK": 0.08,
    "default": 0.10,
}


def build_graph(hospital_network: Dict[str, Any]) -> nx.DiGraph:
    """
    Construct a networkx DiGraph from the hospital_network.json topology.

    Expected JSON format::

        {
          "nodes": ["ICU", "Ward_A", ...],
          "edges": [{"from": "ICU", "to": "Ward_A", "weight": 0.05}, ...]
        }
    """
    G = nx.DiGraph()
    G.add_nodes_from(hospital_network["nodes"])
    for edge in hospital_network["edges"]:
        G.add_edge(edge["from"], edge["to"], weight=edge["weight"])
    return G


def transmission_probability(
    graph: nx.DiGraph,
    source_ward: str,
    target_ward: str,
    infected_count: int,
    pathogen_type: str = "ESBL",
    immune: bool = False,
) -> float:
    """
    Return P(transmission from source_ward to target_ward) for one timestep.

    Returns 0.0 if there is no edge between the wards.
    """
    if not graph.has_edge(source_ward, target_ward):
        return 0.0
    if immune:
        return 0.0

    w = graph[source_ward][target_ward]["weight"]
    beta = BETA.get(pathogen_type, BETA["default"])
    return float(np.clip(beta * w * infected_count, 0.0, 1.0))


def simulate_spread(
    graph: nx.DiGraph,
    infected_wards: Dict[str, int],  # ward_id → infected patient count
    pathogen_type: str,
    isolated_wards: Set[str],
    rng: np.random.Generator,
) -> Dict[str, int]:
    """
    Simulate one 24-h transmission step across the contact graph.

    Returns a dict of new infections per ward (delta, not cumulative).
    Isolated wards have zero outbound transmission.
    """
    new_infections: Dict[str, int] = {}

    for source, count in infected_wards.items():
        if source in isolated_wards or count == 0:
            continue
        for _, target, data in graph.out_edges(source, data=True):
            immune = target in isolated_wards
            p = transmission_probability(
                graph, source, target, count, pathogen_type, immune
            )
            if p > 0:
                # Each infected patient independently transmits
                events = int(rng.binomial(count, p))
                if events > 0:
                    new_infections[target] = new_infections.get(target, 0) + events

    return new_infections


def shortest_transmission_path(
    graph: nx.DiGraph,
    source: str,
    target: str,
) -> List[str]:
    """
    Return the most likely transmission path between two wards.

    Uses highest-weight path (inverse weight = distance for Dijkstra).
    Returns empty list if no path exists.
    """
    try:
        # Convert weights to distances (higher weight = shorter path)
        inv_graph = nx.DiGraph()
        for u, v, data in graph.edges(data=True):
            w = data.get("weight", 0.0)
            inv_graph.add_edge(u, v, weight=1.0 / (w + 1e-9))
        return nx.shortest_path(inv_graph, source, target, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
