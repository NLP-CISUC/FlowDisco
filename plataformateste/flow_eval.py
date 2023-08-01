import pickle

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

import networkx as nx
import itertools
from networkx.readwrite import json_graph
import json

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

from Dialog import Dialog

GRAPH_JSON = 'graph.json'
GRAPH_MULTIWOZ_TRAIN = 'graphMultiWoz.json'
GRAPH_TWITTER = 'twitter.json'

KMEANS_SYS = 'kmeans_system.pkl'
KMEANS_USR = 'kmeans_user.pkl'

TEST_DIALOGS_SAMPLE = 'mw_test_sample.txt'
MW_TEST_DIALOGS = 'mw_test.csv'


def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)


def read_dialogs(filename, encoder):
    dialogs = []
    with open(filename, "r") as f:
        lines = f.readlines()
        dialog_id = 1
        curr_dialog = None
        for line in lines:

            if len(line.strip()) == 0:
                dialogs.append(curr_dialog)
                curr_dialog = None
                continue

            if not curr_dialog:
                curr_dialog = Dialog(dialog_id)
                dialog_id += 1

            cols = line.strip().split(': ')
            encoding = model.encode(cols[1])
            curr_dialog.add_turn(cols[0], cols[1], encoding)

    print(str(len(dialogs)), 'dialogs loaded... ')
    return dialogs


def read_dialogs_csv(filename, encoder, top=None):
    dialogs = []
    data = pd.read_csv(filename, sep=';', header=0)

    curr_dialog = None
    dialog_id = 0

    for i, item in data.iterrows():
        #print(item['speaker'], item['utterance'])

        if top and dialog_id == top:
            break

        if item['turn_id'] == 0:

            if curr_dialog:
                dialogs.append(curr_dialog)

            curr_dialog = Dialog(dialog_id)
            dialog_id += 1

            if dialog_id % 100 == 1:
                print('\tLoaded dialogues:', len(dialogs))

        encoding = model.encode(item['utterance'])
        curr_dialog.add_turn(item['speaker'], item['utterance'], encoding)

    if curr_dialog:
        dialogs.append(curr_dialog)

    print(len(dialogs), 'dialogs loaded!')
    return dialogs


def dialogues_to_transitions(dialogs, usr, sys):
    transitions = []

    for d in dialogs:
        previous = 'SOD'

        for i, t in enumerate(d.turns):
            #print(t.speaker, t.utterance)
            df = pd.DataFrame(data=[t.encoding])

            if t.speaker == 'USER':
                cl = 'U' + str(usr.predict(df)[0])
            else:
                cl = 'S' + str(sys.predict(df)[0])

            transitions.append((previous, cl))
            previous = cl

        transitions.append((previous, 'EOD'))

    return transitions


def eval(dialogs, km_usr, km_sys, flow, min_threshold=None):

    if min_threshold:
        flow = remove_transitions(flow, min_threshold=min_threshold)

    print('Nodes:', len(flow))
    print('Edges:', len(flow.edges))
    transitions = dialogues_to_transitions(dialogs, km_usr, km_sys)

    previstas = 0
    for t in transitions:
        if flow.has_edge(t[0], t[1]):
            previstas += 1

    acc = float(previstas) / len(transitions)
    print('Accuracy =', acc)
    return acc, len(flow)


def remove_transitions(graph, min_threshold=None):
    edges_keep = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > min_threshold]
    g = nx.Graph(edges_keep)

    #if g.has_node('SOD') and g.has_node('EOD'):
    if nx.has_path(g, 'SOD', 'EOD'):
        all_path_nodes = set(itertools.chain(*list(nx.all_simple_paths(g, source='SOD', target='EOD', cutoff=len(graph)/2))))
        #sem cutoff, nunca mais acaba... len(graph) / 2 foi apenas uma heurística, mas para grafos maiores pode ter de se mudar!

        h = g.subgraph(all_path_nodes)
    else:
        h = nx.Graph()

    return h

print('Reading and embedding dialogues...')
dialogs = read_dialogs_csv(MW_TEST_DIALOGS, model, top=1100)
#dialogs = read_dialogs_csv(MW_TEST_DIALOGS, model, top=50)
#dialogs = read_dialogs(TEST_DIALOGS_SAMPLE, model)

print('Reading K-Means...')
with open(KMEANS_SYS, "rb") as ks:
    km_sys = pickle.load(ks)
with open(KMEANS_USR, "rb") as ku:
    km_usr = pickle.load(ku)

print('Reading flow...')
graph = read_json_file(GRAPH_MULTIWOZ_TRAIN)

print('Prob >= 0.05', len(graph))
eval(dialogs, km_usr, km_sys, graph)

#graph = read_json_file(GRAPH_TWITTER)
print('Testing with other thresholds...')
thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
for t in thresholds:
    print('Prob >=', t)
    (acc, nos) = eval(dialogs, km_usr, km_sys, graph, min_threshold=t)
    if nos == 0: #se o grafo ficar vazio com um threshold, também vai ficar com thresholds maiores
        break
