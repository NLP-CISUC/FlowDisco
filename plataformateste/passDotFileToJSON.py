import networkx as nx
import pydotplus
from networkx.readwrite import json_graph
import json
# Parse .dot file
dot_graph = pydotplus.graph_from_dot_file('file.dot')
# Convert dot to networkx graph
nx_graph = nx.drawing.nx_pydot.from_pydot(dot_graph)
# Convert networkx graph to json
data = json_graph.node_link_data(nx_graph)
# Write json to file
with open('file2.json', 'w') as f:
    json.dump(data, f)