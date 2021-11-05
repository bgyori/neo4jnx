"""Tests that compare a neo4jnx graph to a networkx graph"""

from random import choice, choices
from neo4jnx.tests.util import n4_g, nx_g
from neo4jnx.run_pathfinding_algs import *
from indra_network_search.tests.util import basemodels_equal


# Testing g.nodes, g.edges
def test_nodes_edges_count():
    # Test that the number of nodes and edges are the same
    assert len(n4_g.nodes) == len(nx_g.nodes)
    assert len(n4_g.edges) == len(nx_g.edges)


def test_node_attributes():
    nodes_to_test = choices(list(nx_g.nodes), k=100)
    for n in nodes_to_test:
        nx_node_data = nx_g.nodes[n]
        n4_node_data = n4_g.nodes[n]
        assert n4_node_data["ns"] == nx_node_data["ns"]
        assert n4_node_data["id"] == nx_node_data["id"]


def test_edge_attributes():
    # {'statements': [{'stmt_hash': 29290247440853755,
    #    'stmt_type': 'Inhibition',
    #    'evidence_count': 2,
    #    'belief': 0.86,
    #    'source_counts': {'reach': 2},
    #    'english': 'RTN3 inhibits cell cycle.',
    #    'position': None,
    #    'weight': 0.15082288973458366,
    #    'residue': None,
    #    'curated': False}],
    #  'belief': 0.8599999999999999867,
    #  'weight': 0.15082288973458365064,
    #  'z_score': 0,
    #  'corr_weight': 1.0}
    edges_to_test = choices(list(nx_g.edges), k=100)
    for e in edges_to_test:
        n4_data = n4_g.edges[e]
        nx_data = nx_g.edges[e]

        _assert_edge_data_equal(n4_data, nx_data)


def test_graph_attributes_nodes():
    node = choice(list(nx_g.nodes))

    # Test g.succ
    assert len(n4_g.succ[node]) == len(nx_g.succ[node])
    assert set(n4_g.succ[node]) == set(nx_g.succ[node])

    # Test g.successors
    assert set(n4_g.successors(node)) == set(nx_g.successors(node))

    # Test g.pred
    assert len(n4_g.pred[node]) == len(nx_g.pred[node])
    assert set(n4_g.pred[node]) == set(nx_g.pred[node])

    # Test g.predecessors
    assert set(n4_g.predecessors(node)) == set(nx_g.predecessors(node))

    # Test g.in_edges
    assert set(n4_g.in_edges(node)) == set(nx_g.in_edges(node))

    # Test g.__getitem__
    # Make sure that the node has successors
    while len(nx_g.succ[node]) == 0:
        node = choice(list(nx_g.nodes))

    b = list(nx_g[node])[0]
    nx_data = nx_g[node][b]
    n4_data = n4_g[node][b]
    _assert_edge_data_equal(n4_data, nx_data)

    # Test g.__contains__
    assert node in n4_g
    assert node in nx_g

    # Test g.__len__
    assert len(n4_g) == len(nx_g)

    # Test g.__iter__
    assert {n for n in n4_g} == {n for n in nx_g}


def test_graph_attributes_edges():
    # Pick an edge at random
    edge = choice(list(nx_g.edges))

    # Test g.edges.__getitem__
    nx_edge_data = nx_g.edges[edge]
    n4_edge_data = n4_g.edges[edge]
    _assert_edge_data_equal(n4_edge_data, nx_edge_data)

    # Test g.edges.__contains__
    assert edge in n4_g.edges
    assert edge in nx_g.edges

    # Test g.edges.__iter__
    assert {e for e in n4_g.edges} == {e for e in nx_g.edges}

    # Test that nx_g.edges() behaves the same as nx_g.edges()
    random_nodes = set(choices(list(nx_g.nodes), k=10))
    nx_edge_iter = []
    for e, data in nx_g.edges(nbunch=random_nodes, data=True):
        nx_edge_iter.append((e, data))
    n4_edge_iter = []
    for e, data in n4_g.edges(nbunch=random_nodes, data=True):
        n4_edge_iter.append((e, data))

    assert len(n4_edge_iter) == len(nx_edge_iter)

    for (a, a_data), (b, b_data) in zip(n4_edge_iter, nx_edge_iter):
        _assert_edge_data_equal(a_data, b_data)


# Test algorithms
def test_bfs_search():
    # Run bfs on a random node that has successors
    node = choice(list(nx_g.nodes))
    while len(nx_g.succ[node]) == 0:
        node = choice(list(nx_g.nodes))

    path_res_nx = run_bfs(graph=nx_g, start=node, reverse=False)
    path_res_n4j = run_bfs(graph=n4_g, start=node, reverse=False)
    assert basemodels_equal(path_res_nx, path_res_n4j, any_item=True)


def test_shortest_simple_paths():
    # Run ssp between two random nodes that have successors/predecessors
    node = choice(list(nx_g.nodes))
    while len(nx_g.succ[node]) == 0:
        node = choice(list(nx_g.nodes))
    end = choice(list(nx_g.succ[node]))
    while end == node or len(nx_g.pred[node]) == 0:
        end = choice(list(nx_g.succ[node]))

    path_res_nx = run_shortest_simple_paths(graph=nx_g, start=node, end=end)
    path_res_n4j = run_shortest_simple_paths(graph=n4_g, start=node, end=end)
    assert basemodels_equal(path_res_nx, path_res_n4j, any_item=True)


def test_dijkstra():
    # Run dijkstra on a random node that has successors
    node = choice(list(nx_g.nodes))
    while len(nx_g.succ[node]) == 0:
        # Try again if the picked node is a leaf node
        node = choice(list(nx_g.nodes))

    path_res_nx = run_open_dijkstra(graph=nx_g, start=node, reverse=False)
    path_res_n4j = run_open_dijkstra(graph=n4_g, start=node, reverse=False)
    assert basemodels_equal(path_res_nx, path_res_n4j, any_item=True)


def test_shared_interactors():
    # Run shared_interactors

    # Pick two nodes that has at least one node in common downstream
    nodes_list = list(nx_g.nodes)
    node1 = choice(nodes_list)
    node2 = choice(nodes_list)
    while node1 == node2 or not set(nx_g.pred[node1]) & set(nx_g.pred[node2]):
        node1 = choice(nodes_list)
        node2 = choice(nodes_list)

    interactors_nx = run_shared_interactors(graph=nx_g, node1=node1,
                                            node2=node2, downstream=True)
    interactors_n4j = run_shared_interactors(graph=n4_g, node1=node1,
                                             node2=node2, downstream=True)
    assert basemodels_equal(interactors_nx, interactors_n4j, any_item=True)


def _close_enough(a: float, b: float):
    """Test that two floats are close enough"""
    return abs(a - b) < 1e-9


def _assert_edge_data_equal(a: dict, b: dict):
    """Test that two edge data dicts are equal"""
    for attr in [
        "weight",
        "belief",
        "z_score",
        "corr_weight",
    ]:
        assert _close_enough(a[attr], b[attr])

    assert len(a["statements"]) == len(b["statements"])

    for a_stmt, b_stmt in zip(a["statements"], b["statements"]):
        assert _stmt_data_equal(a_stmt, b_stmt)


def _stmt_data_equal(a: dict, b: dict):
    """Test that two statement data dictionaries are equal"""
    # Test attributes that can be checked for equality
    for attr in [
        "stmt_hash",
        "stmt_type",
        "evidence_count",
        "position",
        "english",
        "residue",
        "curated",
    ]:
        try:
            assert a[attr] == b[attr]
        except KeyError:  # Handle ontology edges with missing position/residue
            assert (
                (attr == "residue" or attr == "position")
                and a["stmt_type"] == b["stmt_type"] == "fplx"
            )

    # Test attributes with float values
    for attr in ["belief", "weight"]:
        assert _close_enough(a[attr], b[attr])

    # Check 'source_counts'
    assert all(
        b["source_counts"][src] == count for src, count in a["source_counts"].items()
    )
    return True
