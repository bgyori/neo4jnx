1. Dump networkx graph into nodes/relations TSV

```
from neo4jnx.tsv import graph_to_tsv
g = pklload('indranet_dir_graph.pkl')
graph_to_tsv(g, 'docker/nodes.tsv.gz', 'docker/edges.tsv.gz')
```

2. Run Docker build for neo4j which loads dumped TSVs

```
cd docker
docker build --tag neo4j_test:latest .
```

3. Launch Docker container

```
docker run -d -it -p 7475:7474 -p 7688:7687 neo4j_test:latest
```

4. Instantiate Neo4jDiGraph object with config matching the
    launched container

```
from neo4jnx import Neo4jDiGraph
g = Neo4jDiGraph(neo4j_url='neo4j://localhost:7688',
                 neo4j_auth=('neo4j', 'admin'),
                 property_loaders={'statements': json.loads,
                                   'belief': float, 'weight': float,
                                   'z_score': float, 'corr_weight': float})
```
