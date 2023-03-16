import graphviz
import networkx as nx
import pandas as pd

from pathlib import Path
from graphviz import Source
RESULTS_FOLDER = Path(__file__).parent.parent / "results"

def generate_markov_chain(n_clusters: int, matrix: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    threshold = 0.2  # set the threshold value
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
    # Set the Graphviz graph attributes for layout and style
    graph_attrs = {
        'layout': 'dot',
        #'overlap': 'false',
        #'rankdir': 'LR'
    }
    # Set the rankdir attribute to 'BT' to arrange nodes from bottom to top -> to SOD be the first one and EOD the last one
    G.graph['rankdir'] = 'BT'
    # Create a Graphviz object from the NetworkX graph and attributes
    gv = nx.nx_agraph.to_agraph(G)
    gv.graph_attr.update(graph_attrs)
    for node, attrs in node_attrs.items():
        n = gv.get_node(node)
        n.attr.update(attrs)
    return G

def generate_markov_chain_separately(n_clusters: int, matrix: pd.DataFrame) -> nx.MultiDiGraph:
    n_clustersBoth = n_clusters + n_clusters
    G = nx.MultiDiGraph()
    threshold = 0.2  # set the threshold value
    for i in range(len(matrix.columns)):
        for j in range(len(matrix.index)):
            if matrix.iloc[j,i] > threshold:
                G.add_edge(matrix.columns[i],matrix.index[j], weight=matrix.iloc[j,i], label=matrix.iloc[j,i], dir='back') # direção oposta
    # Set the node colors based on their position in the graph
    colors = []
    for node in G.nodes():
        if node in matrix.columns[:n_clusters]:  # nodes for user
            colors.append('red')
        elif node in matrix.columns[n_clusters:n_clustersBoth]:  # nodes for system
            colors.append('blue')
        else:  # remaining nodes are blue
            colors.append('orange')
    # Create a dictionary of Graphviz node attributes with the colors
    node_attrs = {
        node: {'color': colors[i]} for i, node in enumerate(G.nodes())
    }
    nx.set_node_attributes(G, node_attrs)
    G.graph['rankdir'] = 'BT'

    # Set the Graphviz graph attributes for layout and style
    #graph_attrs = {
        #'layout': 'dot',
    #}


    # Set the rankdir attribute to 'BT' to arrange nodes from bottom to top -> to SOD be the first one and EOD the last one
    #G.graph['rankdir'] = 'BT'

    # Create a Graphviz object from the NetworkX graph and attributes
    #gv = nx.nx_agraph.to_agraph(G)
    #gv.graph_attr.update(graph_attrs)
    #for node, attrs in node_attrs.items():
        #n = gv.get_node(node)
        #n.attr.update(attrs)

    # Write the Graphviz object to a .dot file
    #gv.layout('dot')
    #gv.draw('markov_model.png')
    #gv.write('markov_model.dot')

    #gv é do tipo agraph (objeto de G) quando é feito o print de gv mostra as cores e a direção correta, mas não se conseguiu guardar o gv em dot

    return G

#nx.drawing.nx_agraph.write_dot(G, "graph.dot", graph_attr={"rankdir": "LR"} rankdir BT => Bottom -> Top (aplicar este terceito argumento e consegue-se colocar a direção do grafo ))
def save(graph: nx.MultiDiGraph, filename: str) -> None:
    nx.drawing.nx_pydot.write_dot(graph, filename) 


def show_file(filename: str) -> None:
    s = graphviz.Source.from_file(filename)
    s.view()
