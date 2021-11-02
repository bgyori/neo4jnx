import logging

from networkx import DiGraph
from indra_network_search.search_api import IndraNetworkSearchAPI
from indra_network_search.query import *
from indra_network_search.data_models import NetworkSearchQuery


logger = logging.getLogger(__name__)


__all__ = [
    "run_shortest_simple_paths",
    "run_bfs",
    "run_shared_interactors",
    "run_open_dijkstra",
]


def run_shortest_simple_paths(graph, start, end, **filters):
    """Runs the shortest simple paths algorithm on the graph.

    Parameters
    ----------
    graph : indra.network.Graph
        The graph to run the algorithm on.
    start : str
        The starting node.
    end : str
        The ending node.
    filters : dict
        A dictionary of filters to pass to NetworkSearchQuery.

    Returns
    -------
    list
        A list of lists of nodes.
    """
    api = IndraNetworkSearchAPI(graph, DiGraph())
    try:
        query = NetworkSearchQuery(source=start, target=end, **filters)
    except Exception as e:
        logger.exception(e)
        logger.warning("One or more filters are invalid")
        query = NetworkSearchQuery(source=start, target=end)

    ssp_query = ShortestSimplePathsQuery(query)

    res = api.shortest_simple_paths(ssp_query, is_signed=False)
    return res.get_results()


def run_bfs(graph, start: str, reverse: bool, **filters):
    """Runs the breadth first search algorithm on the graph.

    Parameters
    ----------
    graph : indra.network.Graph
        The graph to run the algorithm on.
    start : str
        The starting node.
    reverse : bool
        Whether to reverse the graph.
    filters : dict
        A dictionary of filters to pass to NetworkSearchQuery.

    Returns
    -------
    list
        A list of lists of nodes.
    """
    api = IndraNetworkSearchAPI(graph, DiGraph())
    start_node = {"source" if not reverse else "target": start}
    try:
        query = NetworkSearchQuery(**start_node, **filters)
    except Exception as e:
        logger.exception(e)
        logger.warning("One or more filters are invalid")
        query = NetworkSearchQuery(**start_node)

    bfs_query = BreadthFirstSearchQuery(query)

    res = api.breadth_first_search(bfs_query, is_signed=False)
    return res.get_results()


def run_open_dijkstra(graph, start: str, reverse: bool, **filters):
    api = IndraNetworkSearchAPI(graph, DiGraph())
    start_node = {"source" if not reverse else "target": start}
    try:
        query = NetworkSearchQuery(**start_node, **filters)
    except Exception as e:
        logger.exception(e)
        logger.warning("One or more filters are invalid")
        query = NetworkSearchQuery(**start_node)

    # Create a dijkstra query
    dijk_query = DijkstraQuery(query)

    # Run the dijkstra query
    res = api.dijkstra(dijk_query, is_signed=False)
    return res.get_results()


def run_shared_interactors(graph, node1: str, node2: str, downstream: bool, **filters):
    api = IndraNetworkSearchAPI(graph, DiGraph())

    try:
        query = NetworkSearchQuery(source=node1, target=node2, **filters)
    except Exception as e:
        logger.exception(e)
        logger.warning("One or more filters are invalid")
        query = NetworkSearchQuery(source=node1, target=node2)

    # Create a SharedInteractorsQuery and run it
    if downstream:
        query = SharedTargetsQuery(query)
        res = api.shared_targets(query, is_signed=False)
    else:
        query = SharedRegulatorsQuery(query)
        res = api.shared_regulators(query, is_signed=False)

    # Return the results
    return res.get_results()


def get_subgraph_edges(g, start_node, max_edges=10000):
    edges = {(start_node, s) for s in g.succ[start_node]}

    # Now get successors of the successors
    new_edges = set()
    for _, n in edges:
        for s in g.succ[n]:
            new_edges.add((n, s))
            if len(edges.union(new_edges)) >= max_edges:
                return edges.union(new_edges)

    # Continue to the next level if necessary
    edges = edges.union(new_edges)
    new_edges = set()
    print(f"Only got {len(edges)} edges, going deeper")
    for _, n in edges:
        for s in g.succ[n]:
            new_edges.add((n, s))
            if len(edges.union(new_edges)) >= max_edges:
                return edges.union(new_edges)
    # Return now
    print(f"Got {len(edges)} edges, returning")
    return edges.union(new_edges)


def get_subgraph(g, start_node: str, max_edges=10000):
    """Get a subgraph of the graph starting from the given node."""
    edges = get_subgraph_edges(g, start_node, max_edges=max_edges)
    nodes = set()
    for edge in edges:
        nodes.update(edge)

    # Following recipe in nx docs
    new_g = g.__class__()
    # Add nodes with their metadata
    new_g.add_nodes_from((n, g.nodes[n]) for n in nodes)
    # Add edges with their metadata
    new_g.add_edges_from((u, v, d) for (u, v), d in g.edges.items() if (u, v) in edges)
    new_g.graph.update(g.graph)

    print(f'Made new graph with {len(new_g.edges)} edges and '
          f'{len(new_g.nodes)} nodes')
    return new_g

