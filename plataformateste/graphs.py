import graphviz
import networkx as nx
import pandas as pd

from pathlib import Path
from graphviz import Source
import numpy as np
import pickle
import sys
RESULTS_FOLDER = Path(__file__).parent.parent / "results"

def generate_markov_chain(n_clusters: int, matrix: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    threshold = 0.20  # set the threshold value
    for i in range(len(matrix.columns)):
        for j in range(len(matrix.index)):
            if matrix.iloc[j,i] > threshold:
                G.add_edge(matrix.columns[i],matrix.index[j], weight=matrix.iloc[j,i], label=matrix.iloc[j,i]) 
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
    return G


def traverse(dataframe, threshold, InputGraph, df_final):
    listi=['SOD']
    non_visited=[]
    visited=['EOD']
    for c in dataframe.columns:
        non_visited.append(c)
    while len(listi) != 0:
        for i in listi:
            k=0
            flag = 0
            for j in dataframe.loc[i]:
                if j > threshold:
                    print(i,"->", dataframe.columns[k], "weight ",j)
                    InputGraph.add_edge(i,dataframe.columns[k],weight=j, label=j, sentiment=df_final.iloc[k]['avg_sentiment'])
                    if dataframe.columns[k] != 'EOD' and dataframe.columns[k] in non_visited and dataframe.columns[k] not in listi:
                        flag = 1
                        listi.append(dataframe.columns[k])
                k=k + 1
            listi.remove(i)
            non_visited.remove(i)
            if flag == 1:
                visited.append(i)
            if len(listi) == 0:
                return visited

def generate_markov_chain_separately(n_clusters_user: int, n_clusters_system: int, matrix: pd.DataFrame, df_final) -> nx.MultiDiGraph:
    n_clustersBoth = n_clusters_user + n_clusters_system
    G = nx.MultiDiGraph()
    threshold = 0.10  # set the threshold value

    visited=traverse(matrix, threshold, G, df_final)
    print(G)
    #tirar todos os arcos que passam por nós que não foram visitados
    to_remove=[]
        
    for i in list(G.nodes):
        if i not in visited:
            to_remove.append(i)
        
    # for n in to_remove:
    #     try:
    #         G.remove_node(n)
    #     except KeyError:
    #         print("node ", n, "does not exist")
                    
    print(G)

    #for i in range(len(matrix.columns)):
    #    for j in range(len(matrix.index)):
    #        if matrix.iloc[j,i] > threshold:
    #            G.add_edge(matrix.columns[i],matrix.index[j], weight=matrix.iloc[j,i], label=matrix.iloc[j,i], sentiment=df_final.iloc[j]['avg_sentiment'], dir='back') # direção oposta
    # Set the node colors based on their position in the graph
    colors = []
    for node in G.nodes():
        if node in matrix.columns[:n_clusters_user]:  # nodes for user
            colors.append('turquoise')
        elif node in matrix.columns[n_clusters_user:n_clustersBoth]:  # nodes for system
            colors.append('#04099C')
        else:  # remaining nodes are blue
            colors.append('orange')
    # Create a dictionary of Graphviz node attributes with the colors
    node_attrs = {
        node: {'color': colors[i]} for i, node in enumerate(G.nodes())
    }
    nx.set_node_attributes(G, node_attrs)
    #G.graph['graph'] = {'rankdir': 'BT'}
    return G

#nx.drawing.nx_agraph.write_dot(G, "graph.dot", graph_attr={"rankdir": "LR"} rankdir BT => Bottom -> Top (aplicar este terceito argumento e consegue-se colocar a direção do grafo ))
def save(graph: nx.MultiDiGraph, filename: str) -> None:
    nx.drawing.nx_pydot.write_dot(graph, filename) 


def show_file(filename: str) -> None:
    s = graphviz.Source.from_file(filename)
    s.view()
