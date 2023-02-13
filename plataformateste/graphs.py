import graphviz
import networkx as nx
import pandas as pd


def generate_markov_chain(n_clusters: int, matrix: pd.DataFrame) -> nx.MultiDiGraph:
    states = list(matrix.columns.values)
    # equals transition probability matrix of changing states given a state
    q_df = pd.DataFrame(columns=states, index=states)
    q_df = matrix

    for p in range(n_clusters):
        q_df.loc[states[p]].reset_index = matrix.iloc[p]

    def _get_markov_edges(q):
        edges = {}
        for col in q.columns:
            for idx in q.index:
                edges[(idx, col)] = q.loc[idx, col]
        return edges

    edges_wts = _get_markov_edges(q_df)
    # pprint(edges_wts)
    # create graph object
    graph = nx.MultiDiGraph()

    # nodes correspond to states
    graph.add_nodes_from(states)

    # edges represent transition probabilities
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        graph.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

    # remove edges below threshold
    threshold = 0.2
    graph.remove_edges_from(
        [(n1, n2) for n1, n2, w in graph.edges(data="weight") if w < threshold]
    )
    return graph


def save(graph: nx.MultiDiGraph, filename: str) -> None:
    nx.drawing.nx_pydot.write_dot(graph, filename)


def show_file(filename: str) -> None:
    s = graphviz.Source.from_file(filename)
    s.view()
