from typing import Optional, Union, List, Any
import networkx as nx
import neo4j
from neo4j import GraphDatabase
from networkx import NetworkXError


def extract_properties(properties, property_loaders):
    return {k: extract_value(v, property_loaders.get(k))
            for k, v in properties.items()}


def extract_value(v, loader):
    if loader:
        return loader(v)
    else:
        return v


class Neo4jDiGraph(nx.DiGraph):
    def __init__(self, neo4j_url, neo4j_auth, property_loaders,
                 *args, **kwargs):
        self.driver = GraphDatabase.driver(neo4j_url, auth=neo4j_auth)
        self.session = None
        self.property_loaders = property_loaders
        super().__init__(*args, **kwargs)

    def __getitem__(self, n):
        query = """
            MATCH ({name: '%s'})-[r:Relation]-(t)
            RETURN r, t
        """ % n
        res = self.query_tx(query)
        return {r[1]['name']: extract_properties(r[0], self.property_loaders)
                for r in res}

    @property
    def nodes(self):
        # Lazy View creation, like in networkx
        nodes = NodeView(self)
        self.__dict__["nodes"] = nodes
        return nodes

    @property
    def edges(self):
        edges = EdgeView(self)
        self.__dict__["edges"] = edges
        return edges

    @property
    def pred(self):
        return PredView(self)

    def predecessors(self, n):
        try:
            return iter(self.pred[n])
        except KeyError as e:
            raise NetworkXError(f"The node {n} is not in the digraph.") from e

    @property
    def succ(self):
        return SuccView(self)

    def successors(self, n):
        try:
            return iter(self.succ[n])
        except KeyError as e:
            raise NetworkXError(f"The node {n} is not in the digraph.") from e

    def __iter__(self):
        return iter(self.nodes)

    def __contains__(self, n):
        return n in self.nodes

    def __len__(self):
        return len(self.nodes)

    def number_of_nodes(self):
        return len(self.nodes)

    def query_tx(self, query: str) -> Union[List[List[Any]], None]:
        """Run a read-only query and return the results.

        Parameters
        ----------
        query :
            The query string to be executed.

        Returns
        -------
        values :
            A list of results where each result is a list of one or more
            objects (typically neo4j nodes or relations).
        """
        tx = self.get_session().begin_transaction()
        try:
            res = tx.run(query)
        except Exception as e:
            tx.close()
            return
        values = res.values()
        tx.close()
        return values

    def get_session(self, renew: Optional[bool] = False) -> neo4j.work.simple.Session:
        """Return an existing session or create one if needed.

        Parameters
        ----------
        renew :
            If True, a new session is created. Default: False

        Returns
        -------
        session
            A neo4j session.
        """
        if self.session is None or renew:
            sess = self.driver.session()
            self.session = sess
        return self.session


class NodeView:
    def __init__(self, graph: Neo4jDiGraph):
        self.graph = graph

    def __iter__(self):
        return iter(self.__call__())

    def __len__(self):
        with self.graph.driver.session() as session:
            query = """MATCH (:Node) RETURN count(*)"""
            return self.graph.query_tx(query)[0][0]

    def __getitem__(self, index):
        try:
            with self.graph.driver.session() as session:
                query = """MATCH (node:Node {`name`: $value }) RETURN node"""
                n = session.run(query, {"value": index}).single()["node"]
                data = {k: n[k] for k in n.keys() if k != 'name'}
                return data
        except TypeError:
            raise KeyError(index)

    def __call__(self, data=False, default=None):
        with self.graph.driver.session() as session:
            query = """MATCH (node:`%s`) RETURN node"""
            nodes = [r["node"] for r in session.run(query).data()]
            if not data:
                for n in nodes:
                    yield n['name']
            elif isinstance(data, bool):
                for n in nodes:
                    rdata = {k: n[k] for k in n.keys() if k != 'name'}
                    yield n['name'], rdata
            else:
                for n in nodes:
                    yield n['name'], n.get(data, default)

    def __contains__(self, n):
        with self.graph.driver.session() as session:
            query = """MATCH (node:Node {`name`: $value }) RETURN node"""
            n = session.run(query, {"value": n}).single()
            return True if n else False

    def get(self, index, default=None):
        try:
            return self.__getitem__(index)
        except KeyError:
            return default


class EdgeView:
    def __init__(self, graph):
        self.graph = graph

    def __iter__(self):
        return iter(self.__call__())

    def __len__(self):
        query = """MATCH (u:Node)-[r:Relation]->(v:Node)
                   RETURN COUNT(r)"""
        return self.graph.query_tx(query)[0][0]

    def __getitem__(self, edge):
        s, t = edge
        # return lookup of edge specifically; looks exactly like
        # AtlasView's __getitem__ ~95 ms, but here that method is exposed
        # directly
        query = """MATCH (u:Node)-[r:Relation]->(v:Node)
                   WHERE u.name = '%s' AND v.name = '%s'
                   RETURN r""" % (s, t)
        return extract_properties(self.graph.query_tx(query)[0][0],
                                  self.graph.property_loaders)

    def __call__(self, data=False, default=None):
        with self.graph.driver.session() as session:
            query = """MATCH (u:Node)-[r:Relation]->(v:Node)
                       RETURN u.Node AS u, v.Node AS v"""
            edges = [(r["u"], r["v"], r["edge"].properties) for r in
                     session.run(query)]
            if not data:
                for u, v, _ in edges:
                    yield u, v
            elif isinstance(data, bool):
                for u, v, d in edges:
                    yield u, v, d
            else:
                for u, v, d in edges:
                    yield u, v, d.get(data, default)


class AdjacencyView:
    def __init__(self, graph):
        self.graph = graph

    def __getitem__(self, n):
        return AtlasView(self.graph, n, 'out')


class PredView(AdjacencyView):
    def __init__(self, graph):
        super().__init__(graph)

    def __getitem__(self, n):
        return AtlasView(self.graph, n, 'in')


class SuccView(AdjacencyView):
    def __init__(self, graph):
        super().__init__(graph)

    def __getitem__(self, n):
        return AtlasView(self.graph, n, 'out')


class AtlasView:
    def __init__(self, graph, n, direction):
        self.graph = graph
        self.n = n
        self.direction = direction

    def __iter__(self):
        if self.direction == 'out':
            query = """MATCH (u:Node)-[r:Relation]->(v:Node)
                    WHERE u.name = '%s'
                    RETURN v.name""" % self.n
        else:
            query = """MATCH (u:Node)-[r:Relation]->(v:Node)
                    WHERE v.name = '%s'
                    RETURN u.name""" % self.n
        res = self.graph.query_tx(query)
        for r in res:
            yield r[0]

    def __contains__(self, n):
        return True if self[n] else False

    def __getitem__(self, n):
        if self.direction == 'out':
            return self.relation_from_source_target(self.n, n)
        else:
            return self.relation_from_source_target(n, self.n)

    def relation_from_source_target(self, s, t):
        query = """MATCH (u:Node)-[r:Relation]->(v:Node)
                   WHERE u.name = '%s' AND v.name = '%s'
                   RETURN r""" % (s, t)
        return extract_properties(self.graph.query_tx(query)[0][0],
                                  self.graph.property_loaders)