import csv
import gzip
import json


def get_data_value(data, key):
    val = data.get(key)
    if not val:
        return ""
    elif isinstance(val, (list, dict)):
        return json.dumps(val)
    elif isinstance(val, str):
        return val.replace('\n', ' ')
    else:
        return val


def canonicalize(s):
    return s.replace('\n', ' ')


def graph_to_tsv(g, nodes_path, edges_path):
    metadata = sorted(set(key for node, data in g.nodes(data=True)
                          for key in data))
    header = "name:ID", ':LABEL', *metadata
    node_rows = (
        (canonicalize(node), 'Node',
         *[get_data_value(data, key) for key in metadata])
        for node, data in g.nodes(data=True)
    )

    with gzip.open(nodes_path, mode="wt") as fh:
        node_writer = csv.writer(fh, delimiter="\t")  # type: ignore
        node_writer.writerow(header)
        node_writer.writerows(node_rows)

    metadata = sorted(set(key for u, v, data in g.edges(data=True)
                          for key in data))
    edge_rows = (
        (
            canonicalize(u), canonicalize(v), 'Relation',
            *[get_data_value(data, key) for key in metadata],
        )
        for u, v, data in g.edges(data=True)
    )

    with gzip.open(edges_path, "wt") as fh:
        edge_writer = csv.writer(fh, delimiter="\t")  # type: ignore
        header = ":START_ID", ":END_ID", ":TYPE", *metadata
        edge_writer.writerow(header)
        edge_writer.writerows(edge_rows)

