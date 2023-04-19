import graphviz
import networkx as nx
import pandas as pd

from pathlib import Path
from graphviz import Source
RESULTS_FOLDER = Path(__file__).parent.parent / "results"

def generate_markov_chain(n_clusters: int, matrix: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    threshold = 0.20  # set the threshold value
    for i in range(len(matrix.columns)):
        for j in range(len(matrix.index)):
            if matrix.iloc[j,i] > threshold:
                G.add_edge(matrix.columns[i],matrix.index[j], weight=matrix.iloc[j,i], label=matrix.iloc[j,i], dir='back') # direção oposta
    # Set the node colors based on their position in the graph
    colors = []
    for node in G.nodes():
        if node in matrix.columns[:n_clusters]:  # nodes for user
            colors.append('black')
        else:  # remaining nodes are blue
            colors.append('orange')
    # Create a dictionary of Graphviz node attributes with the colors
    node_attrs = {
        node: {'color': colors[i]} for i, node in enumerate(G.nodes())
    }
    
    nx.set_node_attributes(G, node_attrs)

    # Set the rankdir attribute to 'BT' to arrange nodes from bottom to top -> to SOD be the first one and EOD the last one
    G.graph['rankdir'] = 'TB'

    # Create a Graphviz object from the NetworkX graph and attributes
    #gv = nx.nx_agraph.to_agraph(G)
    #gv.graph_attr.update(graph_attrs)
    #for node, attrs in node_attrs.items():
    #    n = gv.get_node(node)
    #    n.attr.update(attrs)
    return G

def generate_markov_chain_separately(n_clusters_user: int, n_clusters_system: int, matrix: pd.DataFrame) -> nx.MultiDiGraph:
    n_clustersBoth = n_clusters_user + n_clusters_system
    G = nx.MultiDiGraph()
    threshold = 0.20  # set the threshold value
    for i in range(len(matrix.columns)):
        for j in range(len(matrix.index)):
            if matrix.iloc[j,i] > threshold:
                G.add_edge(matrix.columns[i],matrix.index[j], weight=matrix.iloc[j,i], label=matrix.iloc[j,i], dir='back') # direção oposta
    # Set the node colors based on their position in the graph
    colors = []
    for node in G.nodes():
        if node in matrix.columns[:n_clusters_user]:  # nodes for user
            colors.append('red')
        elif node in matrix.columns[n_clusters_user:n_clustersBoth]:  # nodes for system
            colors.append('blue')
        else:  # remaining nodes are blue
            colors.append('orange')
    # Create a dictionary of Graphviz node attributes with the colors
    node_attrs = {
        node: {'color': colors[i]} for i, node in enumerate(G.nodes())
    }
    nx.set_node_attributes(G, node_attrs)
    G.graph['graph'] = {'rankdir': 'BT'}
    return G

#nx.drawing.nx_agraph.write_dot(G, "graph.dot", graph_attr={"rankdir": "LR"} rankdir BT => Bottom -> Top (aplicar este terceito argumento e consegue-se colocar a direção do grafo ))
def save(graph: nx.MultiDiGraph, filename: str) -> None:
    nx.drawing.nx_pydot.write_dot(graph, filename) 


def show_file(filename: str) -> None:
    s = graphviz.Source.from_file(filename)
    s.view()
