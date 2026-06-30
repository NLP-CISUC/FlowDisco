# %% [markdown]
# # Requisitos

# %%
#!pip install -r requirements.txt
#pip freeze > r.txt
#python3.11 -m venv venv
#source venv/bin/activate
#rm -rf venv

import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
import pandas as pd
import pickle
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, pickle
import networkx.drawing.nx_pydot as nx_pydot
import threading
import spacy
import numpy as np
import statistics
import seaborn as sns
from bertopic import BERTopic
import itertools
from pathlib import Path
import random
import sys
from datetime import datetime
import joblib
import pydot
import random
import json
import re
import math
from collections import Counter
import time
import hdbscan
import flask
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from sklearn.cluster import HDBSCAN, DBSCAN, KMeans, OPTICS, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from kneed import KneeLocator
import networkx as nx
from networkx.readwrite import json_graph
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from rank_bm25 import BM25Okapi
from rake_nltk import Rake
import optuna
from optuna.storages import RDBStorage
import graphviz
from graphviz import Source
#from langchain.llms.ollama import Ollama
from langchain_community.llms import Ollama
import matplotlib
matplotlib.use('Agg')
#import flask-cors
import networkx.drawing.nx_pydot as nx_pydot
import Levenshtein
from itertools import islice
import unicodedata


import warnings
warnings.filterwarnings("ignore")

# ### Modelos de Sentence Transformers

# %%
nltk.download("stopwords")
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words("portuguese")

#Português
MODEL_ML = "paraphrase-multilingual-MiniLM-L12-v2"
#MODEL_ML = "rufimelo/bert-large-portuguese-cased-sts"
#MODEL_ML = "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder"
#MODEL_ML = "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder"

#Inglês
#MODEL_ML = "all-MiniLM-L6-v2"
#MODEL_ML ="all-mpnet-base-v2"
#MODEL_ML ="multi-qa-mpnet-base-cos-v1"
#MODEL_ML ="gtr-t5-base"
#MODEL_ML ="all-MiniLM-L12-v2"
#MODEL_ML ="gtr-t5-large"

# ### Parâmetros

# Algoritmo
algorithm = "kmeans"
#algorithm = "keywords"
#algorithm = "bertopic"
#algorithm = "hieraquico"
#algorithm = "dbscan"
#algorithm = "none"

# Labelling
#labelling = "kbert"
#labelling = "keywords"
#labelling = "verbs"
#labelling = "closest"
labelling = "llm"
#labelling = "none"

labelling_for_excel = [
    #"kbert",
    "llm",
    #"verbs",
    #"closest",
    # "none"
]

# Métrica a otimizar
metric_to_optimize = "silhouette"
#metric_to_optimize = "vmeasure"
#metric_to_optimize = "NONE"

# Datasets de treino
#filename = "DAs_train_emo.csv"
#filename = "MRDA_train.csv"
#filename = "Estado_Da_Nacao_21_julho_2021_falas_separadas.xlsx"
#filename = "Estado_da_Nacao_2025_CERTO.xlsx"
#filename = "Estado_Da_Nacao_17_julho_2025.xlsx"
filename = "24_TREINO_new.xlsx"
#filename = "Estado_Da_Nacao_17_julho_2024.xlsx"
#filename = "Estado_Da_Nacao_20_julho_2023.xlsx"
#filename = "Estado_Da_Nacao_20_julho_2022.xlsx"
#filename = "Estado_Da_Nacao_21_julho_2021.xlsx"
#filename = "24_TREINO.xlsx"
#filename = "DATASET_FINAL.xlsx"
#filename = "dataset_short_paper_FINAL.xlsx"

# Datasets de teste
#filename_test = "DAs_test_emo.csv"
#filename_test = "MRDA_test.csv"
#filename_test = "Estado_Da_Nacao_21_julho_2021_falas_separadas.xlsx"
#filename_test = "Estado_Da_Nacao_20_julho_2023.xlsx"
#filename_test = "Estado_Da_Nacao_20_julho_2022.xlsx"
#filename_test = "Estado_Da_Nacao_21_julho_2021.xlsx"
#filename_test = "Estado_Da_Nacao_17_julho_2025.xlsx"
filename_test = "24_TESTE_new.xlsx"
#filename_test = "24_TESTE.xlsx"
#filename_test = "DATASET_FINAL.xlsx"
#filename_test = "dataset_short_paper_FINAL.xlsx"

# Modos disponíveis: "GENERICO", "POLITICA" ou "POESIA"
TIPO_DATASET = "GENERICO"

#Valores de Threshold
#threshold = [0, 0.05, 0.1, 0.15, 0.2]
threshold = [0]

# Beta V-measure
beta = 1

# Número de trials para o optuna
n_trials = 5
MAPA_SPEAKERS = {}

#Se queremos colocar na label a contagem do número de utterances por cluster
count_utterances_label = False

#Sentimento do system (True = Transições com sentimento (cor)); False: Transições sem sentimento (a preto))
sentiment_system = True

#Se queremos colocar na label a contagem do número de utterances, o turno e tempo por cluster
turns_label = False
time_start_label = False
time_previous_label = False

#cluster = qd a cor do sentimento tem a ver com o sentimento médio destino
#transição = qd o sentimento médio é por transição origem->destino
#sentimento no fluxo (Opções: False | cluster | transition)
sentiment_in_flow = True

# Número de utterances passadas a considerar para o contexto (1 = sem contexto)
id_max = 1

# URL para o Ollama funcionar
llm_url = "http://localhost:11434"

#Diretoria
#diretoria = filename +'_'+ algorithm +'_'+ labelling +'_'+ metric_to_optimize +'_'+ str(threshold[0]) +'_'+ str(n_trials) +'_'+ str(datetime.now())
#diretoria = filename +'_'+ algorithm +'_'+ labelling +'_'+ metric_to_optimize +'_'+ str(threshold[0]) +'_'+ str(n_trials) +'_'+ str(datetime.now()).replace(':', '-')
diretoria = f"Res_{algorithm}_{datetime.now().strftime('%H%M%S')}"
os.makedirs('./Resultados/' + diretoria)
#print (diretoria)

amount_speakers = 0
#amount_speakers = ["USER", "SYSTEM"]
#amount_speakers = ["Rachel", "Ross", "Joey" , "Monica" , "Phoebe", "Chandler"]
#amount_speakers = ["Chandler"]
#amount_speakers = ["Presidente", "Primeiro-Ministro", "PSD", "PS", "CH"]
others = True
same_speakers = False
ENTIDADES_A_PROCESSAR = {}
BLOCO_MODERADOR = []
BLOCO_GOVERNO_SPEAKERS = []
INCLUIR_OUTROS = False

if TIPO_DATASET == "POLITICA":
    #Dividir por categorias
    ENTIDADES_A_PROCESSAR = {
        "Esquerda": ["BE", "PCP", "PEV", "L", "PS"],
        "Direita": ["IL", "PSD", "CDS-PP", "CH"],
        "Outros_Partidos": ["PAN", "JPP"]
    }

    # Meter só dois partidos
    #ENTIDADES_A_PROCESSAR = {
    #    "PS": ["PS"],
    #    "PSD": ["PSD"]
    #}

    # Todos os partidos separados
    #ENTIDADES_A_PROCESSAR = {
    #    "PS": ["PS"], 
    #    "PSD": ["PSD"], 
    #    "CH": ["CH"], 
    #    "IL": ["IL"],
    #    "BE": ["BE"], 
    #    "PCP": ["PCP"], 
    #    "L": ["L"], 
    #    "PAN": ["PAN"], 
    #    "CDS-PP": ["CDS-PP"], 
    #    "PEV": ["PEV"], 
    #    "JPP": ["JPP"]
    #}

    # Governo
    #BLOCO_MODERADOR = ["Presidente"]
    BLOCO_MODERADOR = ["Presidente", "Secretário", "Secretária"]
    BLOCO_GOVERNO_SPEAKERS = ["Primeiro-Ministro", "Ministro", "Ministra"]

    # Se 'True', qualquer speaker que não esteja em ENTIDADES_A_PROCESSAR
    # será agrupado num único nó chamado "Outros".
    # Se 'False', eles serão simplesmente removidos do grafo.
    #INCLUIR_OUTROS = False

n_nodes = 0

# Validation
val = {"validation":False, "val_filename":"24_TREINO.xlsx", "val_filename_test":"24_TREINO.xlsx", "val_model":"all-MiniLM-L6-v2"}


def normalize_dataset(df_initial, regex=None, removeGreetings=None, speaker=None):
    """Normalize turn_id by (all turn_id's/max turn_id) and some column names."""
    df = df_initial.copy()

    if "Tipo" in df.columns:
        df = df[df["Tipo"].astype(str).str.lower().eq("fala")].copy()
    else:
        print("Coluna 'Tipo' não encontrada.")

    # Também normaliza as ações
    if "Tipo" in df.columns and "Ação" in df.columns:
        mask_acoes = df["Tipo"].astype(str).str.lower() == "ação"
        if mask_acoes.any():
            df.loc[mask_acoes, "Ação"] = df.loc[mask_acoes, "Ação"].astype(str).str.lower().str.strip()

    # Padrões para substituição
    url_pattern = r"https?://\S+"
    url_placeholder = "xURLx"
    user_tags_pattern = "@\\S+"
    user_tags_placeholder = "xUSERNAMEx"

    #Renomear as colunas
    if "text" in df.columns:
        df.rename(columns={"text": "utterance"}, inplace=True)

    if "Utterance" in df.columns:
        df.rename(columns={"Utterance": "utterance"}, inplace=True)

    if "transcript" in df.columns:
        df.rename(columns={"transcript": "utterance"}, inplace=True)

    if "Msg" in df.columns:
        df.rename(columns={"Msg": "utterance"}, inplace=True)

    if "intent_title" in df.columns:
        df.rename(columns={"intent_title": "trueLabel"}, inplace=True)

    if "dialog_act" in df.columns:
        df.rename(columns={"dialog_act": "trueLabel"}, inplace=True)

    if "dialogue_act" in df.columns:
        df.rename(columns={"dialogue_act": "trueLabel"}, inplace=True)

    if "speaker" in df.columns:
        df.rename(columns={"speaker": "Speaker"}, inplace=True)

    if "user" in df.columns:
        df.rename(columns={"user": "Speaker"}, inplace=True)

    if "interlocutor" in df.columns:
        df.rename(columns={"interlocutor": "Speaker"}, inplace=True)

    if "trueLabel" in df.columns:
        df["trueLabel"] = df["trueLabel"].replace(" ", "_", regex=True)

    if 'trueLabel' in df.columns:
        df['utterance']= df['utterance'].apply(lambda x: x.lower())

    if 'trueLabel' in df.columns:
        df.trueLabel = df.trueLabel.fillna('none')
    
    if "Emotion" in df.columns:
        df.rename(columns={"Emotion": "Emotion"}, inplace=True)

    # Substituição de padrões usando expressões regulares
    if regex is True:
        df["utterance"] = df["utterance"].replace(
            to_replace=url_pattern, value=url_placeholder, regex=True
        )
        df["utterance"] = df["utterance"].replace(
            to_replace=user_tags_pattern, value=user_tags_placeholder, regex=True
        )

    if speaker == "both":
        df = df

    if 'Speaker' in df.columns:
        df['Speaker'] = df['Speaker'].replace('USR', 'USER')
        df['Speaker'] = df['Speaker'].replace('Cliente', 'USER')
        df['Speaker'] = df['Speaker'].replace('SERVICE', 'SYSTEM')
        df['Speaker'] = df['Speaker'].replace('SYS', 'SYSTEM')
        df['Speaker'] = df['Speaker'].replace('Automaise', 'SYSTEM')

    df['Speaker'] = df['Speaker'].str.strip()

    # Criação de 'dialogue_id' com base no 'turn_id'
    if 'dialogue_id' not in df.columns and 'turn_id' in df.columns:
        dialog = 0
        result = []
        i_anterior = -1
        for i in df['turn_id']:
            if i_anterior == -1 or i > i_anterior:
                i_anterior = i
            else:
                dialog = dialog + 1
                i_anterior = -1
            result.append(dialog)
        df['dialogue_id'] = result

    # Criação de 'turn_id' com base no 'dialogue_id'
    if 'turn_id' not in df.columns and 'dialogue_id' in df.columns:
        df['turn_id'] = df.groupby('dialogue_id').cumcount()

    #create variable which is a incremental sequence by number of utterances
    df['sequence'] = [i for i in range(len(df))]

    return df


def compute_weighted_mean(df, id_max=99999, sep=False, opt=1):
    df = df.sort_values(by=['dialogue_id', 'turn_id']).reset_index(drop=True)

    weighted_vectors = [None] * len(df)  # Initialize the list with None to maintain alignment

    if sep:
        grouped = df.groupby(['dialogue_id', 'Speaker'])
    else:
        grouped = df.groupby('dialogue_id')

    for name, group in grouped:
        sum_weighted_vectors = []

        for idx, row in group.iterrows():
            vector = row['vectors']
            if len(sum_weighted_vectors) >= id_max:
                sum_weighted_vectors.pop(0)
            sum_weighted_vectors.append(vector)

            sum_weights = range(1, len(sum_weighted_vectors) + 1)
            if opt == 2:
                weighted_mean_vector = np.concatenate(np.mean(sum_weighted_vectors[:-1], axis=0), sum_weighted_vectors[-1])
            else:
                weighted_mean_vector = sum(sum_weighted_vectors[g] * sum_weights[g] / sum(sum_weights) for g in range(len(sum_weighted_vectors)))
            weighted_vectors[idx] = weighted_mean_vector

    df['vectors_weight'] = weighted_vectors
    return df

#Para a política
def mapear_entidade_dinamica(row, ano, gov_partido, entidades_map, incluir_outros=True):
    speaker_nome = str(row.get('Speaker', ''))
    partido_nome = str(row.get('Partido', ''))
    tipo = str(row.get('Tipo', ''))

    # 1. Tratar de Ações
    if tipo.lower() == 'ação':
        return 'Acao' 

    # 2. Tratar do Governo (pelo nome do Speaker)
    for gov_speaker in BLOCO_GOVERNO_SPEAKERS:
        if gov_speaker in speaker_nome:
            return "GOV"
            
    # 3. Tratar do Presidente
    if any(mod in speaker_nome for mod in BLOCO_MODERADOR):
        return "Presidente" 
        
    # 4. Tratar dos Partidos
    partido_real = partido_nome if pd.notna(partido_nome) and partido_nome else speaker_nome
    
    partido_real_upper = partido_real.upper()

    for nome_no, lista_partidos in entidades_map.items():
        for p in lista_partidos:
            # Cria um padrão que procura a palavra exata. Assim "PS" não é encontrado dentro de "PSD" por ex
            padrao = r'\b' + re.escape(p.upper()) + r'\b'
            
            if re.search(padrao, partido_real_upper):
                return nome_no # Retorna a CHAVE (ex: PS, PSD)

    return "Outros" if incluir_outros else None

cores_personalizadas = {}
def atribuir_cor_friends(speaker):
    cores_fixas = {
        "Rachel": "#FF6B6B", "Joey": "#FF9966", "Monica": "#66CDAA",
        "Ross": "#6A5ACD", "Chandler": "#B266FF", "Phoebe": "#FF69B4",
        "Others": "#40E0D0", "User": "#33CCFF", "System": "#3300FF",
        "SOD": "#FFD700", "EOD": "#FFD700"
    }

    if speaker in cores_fixas:
        return cores_fixas[speaker]
    if speaker in cores_personalizadas:
        return cores_personalizadas[speaker]

    # gera cor nova
    cores_usadas = set(cores_fixas.values()).union(cores_personalizadas.values())
    while True:
        cor_aleatoria = "#" + "".join(random.choices("0123456789ABCDEF", k=6))
        if cor_aleatoria not in cores_usadas:
            cores_personalizadas[speaker] = cor_aleatoria
            return cor_aleatoria

def use_sentence_transformer(sentences, model):
    # As frases são codificadas quando chamamos o  model.encode()
    embeddings = model.encode(sentences)
    vectors = np.array(embeddings)
    return vectors

#Silhouette Score
def silhouette_method(data, min_k, max_k, incr):
    number_clusters = 0
    atual_silhouette = 0.000
    # Prepare the scaler
    scale = StandardScaler().fit(data)

    # Fit the scaler
    scaled_data = pd.DataFrame(scale.fit_transform(data))

    # Para o método de silhouette, k precisa começar a partir de 2
    n_clusters_axis = range(min_k, max_k, incr)
    silhouettes = []

    # Ajustar o modelo
    for k in n_clusters_axis:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=2)
        kmeans.fit(scaled_data)
        score = metrics.silhouette_score(scaled_data, kmeans.labels_)
        silhouettes.append(score)
        if score > atual_silhouette:
            number_clusters = k
            atual_silhouette = score

    #Gerar o grafo de dispersão
    scatter_plot = sns.scatterplot(
        x=n_clusters_axis, y=silhouettes)

    #Usa a função get_figure e armazena o gráfico em uma variável (scatter_fig)
    scatter_fig = scatter_plot.get_figure()

    # Guarda o gráfico
    scatter_fig.savefig('scatterplot.png')

    return number_clusters

# #### K-means
# Função para treinar o modelo KMeans e guardar os resultados num ficheiro pickle
def clustering_kmeans(vectors, n_clusters, nomeFichPickle, max_iters=2500):
    # Verifica se o modelo KMeans já foi treinado e guardado
    if not os.path.exists(nomeFichPickle):
        # Se o modelo ainda não foi treinado, executar o KMeans
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=max_iters, random_state=2)
        kmeans.fit(vectors)

        # Obtém rótulos de cluster e centros de cluster
        labels_kmeans = kmeans.labels_
        centers_kmeans = kmeans.cluster_centers_

        print("\nInternal Evaluation:\nSilhouette Score: ", metrics.silhouette_score(vectors, labels_kmeans))
        print("Davies-Bouldin Index (DBI): ", metrics.davies_bouldin_score(vectors, labels_kmeans))

        # Guarda o modelo KMeans treinado num ficheiro pickle
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'wb') as file:
            pickle.dump((labels_kmeans, centers_kmeans), file)
    else:
        # Se o modelo já foi treinado e guardado, carrega-o do ficheiro pickle
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'rb') as file:
            labels_kmeans, centers_kmeans = pickle.load(file)

    return labels_kmeans, centers_kmeans

# ### Topic Modeling

def bertopic_modeling(utterances, model_ml, n_topics, min_topic_size):
    model = BERTopic(embedding_model=model_ml, nr_topics=n_topics)
    topics, probabilities = model.fit_transform(utterances)

    return model, topics, probabilities

# %%
def objective_bertopic(trial, vectors, utterances):
    utterances = [str(doc) for doc in utterances if isinstance(doc, str) and doc.strip()]
    n_topics = trial.suggest_int("n_topics", 2, 10)
    min_topic_size = trial.suggest_int("min_topic_size", 2, 10)

    # Criação do modelo BERTopic
    topic_model = BERTopic(nr_topics=n_topics, min_topic_size=min_topic_size)

    # Ajuste do modelo aos dados
    topics, probs = topic_model.fit_transform(utterances)

    # Cálculo do Silhouette Score usando os tópicos atribuídos
    silhouette = silhouette_score(vectors, topics)
    return silhouette


def clustering_bertopic_optuna(vectors, utterances, role, metric, nomeFichPickle, n_trials=5):
    labels_topic_model = None
    silhouette_topic_model = None
    n_topics = None
    best_params = None

    # Configurar a base de dados para armazenar trials
    storage_path = "sqlite:///optuna_studies.db"
    storage = RDBStorage(url=storage_path)

    # Limpeza dos utterances para garantir que sejam apenas strings
    cleaned_utterances = [str(doc) for doc in utterances if isinstance(doc, str) and doc.strip()]

    # Se o modelo já foi treinado e guardado, carrega-o do ficheiro pickle
    if os.path.exists(nomeFichPickle):
        with open(nomeFichPickle, 'rb') as file:
            topic_model = pickle.load(file)

            # Precisamos passar os documentos (cleaned_utterances) para get_document_info()
            topics = topic_model.get_document_info(cleaned_utterances)["Topic"]

            silhouette_topic_model = silhouette_score(vectors, topics)
            params = topic_model.get_params()

        print(f"Modelo já existente para {role}. Carregando os resultados guardados.")
        print(f"Hiperparâmetros carregados do pickle para {role}: {params}")
        print(f"Melhor {metric} Score carregado do pickle para {role}: {silhouette_topic_model}")
    else:
        # Se não existe um modelo, realiza a otimização com o Optuna
        study = optuna.create_study(direction="maximize", storage=storage)
        study.optimize(lambda trial: objective_bertopic(trial, vectors, cleaned_utterances), n_trials=n_trials)
        best_trial = study.best_trial  # Obtém a melhor tentativa

        best_params = best_trial.params
        n_topics = best_params['n_topics']
        min_topic_size = best_params['min_topic_size']

        # Criação do modelo com os melhores parâmetros
        topic_model = BERTopic(nr_topics=n_topics, min_topic_size=min_topic_size)

        # Ajuste do modelo usando as utterances
        topics, probs = topic_model.fit_transform(cleaned_utterances)

        silhouette_topic_model = silhouette_score(vectors, topics)

        # Guarda o modelo no ficheiro pickle
        with open(nomeFichPickle, 'wb') as file:
            pickle.dump(topic_model, file)

        print(f"Melhores parâmetros encontrados para {role} na Função Silhouette:")
        print(best_params)
        print(f"Melhor Silhouette para {role}: {silhouette_topic_model}")

        labels_topic_model = topics

    return labels_topic_model, silhouette_topic_model, n_topics, best_params

#CLUSTERING -> K-MEANS
def objective_kmeans_silhouette(trial, vectors):
    # Codifica as frases usando o modelo selecionado
    n_clusters = trial.suggest_int('n_clusters', 3, 10)
    init_method = trial.suggest_categorical('init_method', ['k-means++', 'random'])
    n_init = trial.suggest_int('n_init', 1, 30)
    tol = trial.suggest_float('tol', 1e-4, 1e-1, log=True)
    algorithm = trial.suggest_categorical('algorithm', ['lloyd', 'elkan'])

    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_method,
        n_init=n_init,
        tol=tol,
        algorithm=algorithm,
        random_state=2
    )
    kmeans.fit(vectors)

    silhouette = metrics.silhouette_score(vectors, kmeans.labels_)
    return silhouette


def clustering_kmeans_silhouette_optuna(vectors, role, metric, nomeFichPickle):
    labels_kmeans = None
    centers_kmeans = None
    silhouette_kmeans = None
    n_clusters = None
    best_params = None
    parameters_trial = None
    study = None

    # Configurar a base de dados para armazenar trials
    storage_path = f"sqlite:///{os.path.join('./Resultados/' + diretoria, 'optuna_studies.db')}"
    storage = RDBStorage(storage_path)

    # Se o modelo já foi treinado e guardado, carrega-o do ficheiro pickle
    if os.path.exists(nomeFichPickle):
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'rb') as file:
            kmeans = pickle.load(file)
            print("kmeans", kmeans)
            labels_kmeans, centers_kmeans = kmeans.labels_, kmeans.cluster_centers_
            params = kmeans.get_params()
        silhouette_kmeans = metrics.silhouette_score(vectors, labels_kmeans)
        print(f"Modelo já existente para {role}. Carregando os resultados guardados.")
        print(f"Hiperparâmetros carregados do pickle para {role}: {params}")
        print(f"Melhor {metric} Score carregado do pickle para {role}: {silhouette_kmeans}")

        if params is not None:
            n_clusters = params["n_clusters"]
    else:
        # Se o ficheiro pickle não existe, aplica o optuna
        study_name = f"study_{role}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
        )

        study.optimize(lambda trial: objective_kmeans_silhouette(trial, vectors), n_trials=n_trials)
        best_trial = study.best_trial  # Obtém a melhor tentativa

        best_params = best_trial.params
        n_clusters = best_params['n_clusters']
        init = best_params['init_method']
        n_init = best_params['n_init']
        tol = best_params['tol']
        algorithm = best_params['algorithm']

        parameters_trial = {
            'n_clusters': n_clusters,
            'init_method': init,
            'n_init': n_init,
            'tol': tol,
            'algorithm': algorithm
        }

        kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, tol=tol, algorithm=algorithm, random_state=2)
        kmeans.fit(vectors)

        silhouette_kmeans = metrics.silhouette_score(vectors, kmeans.labels_)

        # Guarda todos os resultados no ficheiro pickle
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'wb') as file:
            pickle.dump(kmeans, file)

        print(f"Melhores parâmetros encontrados para {role} na Função Silhouette:")
        print(best_params)
        print(f"Melhor Silhouette para {role}: {silhouette_kmeans}")

        labels_kmeans, centers_kmeans = kmeans.labels_, kmeans.cluster_centers_

    return labels_kmeans, centers_kmeans, silhouette_kmeans, n_clusters, study

#V-measure

#PARA DIALOGUE ACTS
def objective_kmeans_vmeasure(trial, vectors, normalized_df):
    # Defino os parâmetros do KMeans a serem otimizados
    n_clusters = trial.suggest_int('n_clusters', 3, 10)
    init_method = trial.suggest_categorical('init_method', ['k-means++', 'random'])
    n_init = trial.suggest_int('n_init', 1, 30)
    tol = trial.suggest_float('tol', 1e-4, 1e-1, log=True)
    algorithm = trial.suggest_categorical('algorithm', ['lloyd', 'elkan'])

    # Configurar kmeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_method,
        n_init=n_init,
        max_iter=1000,
        tol=tol,
        algorithm=algorithm,
        random_state=2
    )
    kmeans.fit(vectors)

    if 'trueLabel' in normalized_df.columns:
        true_labels = []
        pred_labels = []

        for idx in range(len(kmeans.labels_)):
            # Obter o valor da coluna trueLabel para o índice idx
            true_label_entry = normalized_df.iloc[idx]['trueLabel']

            # Verificar se true_label_entry é uma string válida
            if isinstance(true_label_entry, str):
                utterance_true_labels = true_label_entry.split(',')
            else:
                raise ValueError(f"O 'trueLabel' na linha {idx} não está no formato esperado.")

            cluster_label = kmeans.labels_[idx]

            true_labels.extend(utterance_true_labels)
            pred_labels.extend([cluster_label] * len(utterance_true_labels))

        # Calcular as métricas
        v_measure = metrics.v_measure_score(true_labels, pred_labels, beta=1.0)
        silhouette_score = metrics.silhouette_score(vectors, kmeans.labels_)
        adjusted_rand_score = metrics.adjusted_rand_score(true_labels, pred_labels)
        completeness_score = metrics.completeness_score(true_labels, pred_labels)
        homogeneity_score = metrics.homogeneity_score(true_labels, pred_labels)
        calinski_harabasz_score = metrics.calinski_harabasz_score(vectors, kmeans.labels_)
        davies_bouldin_score = metrics.davies_bouldin_score(vectors, kmeans.labels_)

        # Adicionar os valores calculados aos atributos do trial
        trial.set_user_attr(key='v_measure', value=float(v_measure))
        trial.set_user_attr(key='silhouette_score', value=float(silhouette_score))
        trial.set_user_attr(key='adjusted_rand_score', value=float(adjusted_rand_score))
        trial.set_user_attr(key='completeness_score', value=float(completeness_score))
        trial.set_user_attr(key='homogeneity_score', value=float(homogeneity_score))
        trial.set_user_attr(key='calinski_harabasz_score', value=float(calinski_harabasz_score))
        trial.set_user_attr(key='davies_bouldin_score', value=float(davies_bouldin_score))

        print(f" Trial {trial.number}:")
        print(f"  Silhouette Score: {silhouette_score}")
        print(f"  V-Measure: {v_measure}")
        print(f"  Adjusted Rand Score: {adjusted_rand_score}")
        print(f"  Completeness Score: {completeness_score}")
        print(f"  Homogeneity Score: {homogeneity_score}")
        print(f"  Calinski-Harabasz Score: {calinski_harabasz_score}")
        print(f"  Davies-Bouldin Score: {davies_bouldin_score}")

        return v_measure
    else:
        silhouette_score = metrics.silhouette_score(vectors, kmeans.labels_)
        print("Como o dataset não está anotado, não é possível calcular a V-measure e foi retornada a Silhouette score.")
        return silhouette_score


def clustering_kmeans_vmeasure_optuna(vectors, normalized_df, role, nomeFichPickle, val={}):
    labels_kmeans = None
    centers_kmeans = None
    n_clusters = None
    study = None
    # Configurar a base de dados para armazenar trials
    storage_path = f"sqlite:///{os.path.join('./Resultados/' + diretoria, 'optuna_studies.db')}"
    storage = RDBStorage(storage_path)

    # Se o ficheiro pickle não existe, aplicar o optuna
    # Se for para validar configurações de um dataset diferente
    if val["validation"]:
        # Colocar path do melhor modelo
        best_model_path = f'models/{val["val_model"]}/{val["val_filename"]}/vmeasure/kmeans_{role}.pkl'
        with open(os.path.join('./Resultados/' + diretoria, best_model_path), 'rb') as file:
            kmeans = pickle.load(file)
            labels_kmeans, centers_kmeans = kmeans.labels_, kmeans.cluster_centers_

        best_params = kmeans.get_params()
        n_clusters = best_params['n_clusters']
        init = best_params['init']
        n_init = best_params['n_init']
        tol = best_params['tol']
        algorithm= best_params['algorithm']

        kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, tol=tol, algorithm=algorithm, random_state=2)

        kmeans.fit(vectors)

        silhouette_kmeans = metrics.silhouette_score(vectors, kmeans.labels_)

        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'wb') as file:
            pickle.dump(kmeans, file)

        print(f"Melhores parâmetros encontrados para {role}:")
        print(best_params)
        print(f"Melhor Silhouette para {role}: {silhouette_kmeans}")
        print(f"Modelo já existente para {role}. Carregando os resultados guardados.")
        labels_kmeans, centers_kmeans = kmeans.labels_, kmeans.cluster_centers_
    elif os.path.exists(nomeFichPickle):
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'rb') as file:
            kmeans = pickle.load(file)
            print("kmeans", kmeans)
            labels_kmeans, centers_kmeans = kmeans.labels_, kmeans.cluster_centers_
        print(f"Modelo já existente para {role}. Carregando os resultados guardados.")
    else:
        study_name = f"study_{role}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
        )

        study.optimize(lambda trial: objective_kmeans_vmeasure(trial, vectors, normalized_df), n_trials=n_trials)

        best_trial = study.best_trial
        best_params = best_trial.params
        n_clusters = best_params['n_clusters']
        init = best_params['init_method']
        n_init = best_params['n_init']
        tol = best_params['tol']
        algorithm = best_params['algorithm']

        parameters_trial = {
            'n_clusters': n_clusters,
            'init_method': init,
            'n_init': n_init,
            'tol': tol,
            'algorithm': algorithm
        }

        kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, tol=tol, algorithm=algorithm, random_state=2)
        kmeans.fit(vectors)
        print (vectors)

        print(f"Melhores parâmetros encontrados para {role} na Otimização da V-measure:")
        print(best_params)

        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'wb') as file:
            pickle.dump(kmeans, file)

        labels_kmeans, centers_kmeans = kmeans.labels_, kmeans.cluster_centers_

    return labels_kmeans, centers_kmeans, n_clusters, study

#Outra métrica

# %%
#CLUSTERING -> K-MEANS
def objective_kmeans_nova_func(trial, vectors, n_clusters_ant, best_silhouette):
    n_clusters_ideal = 5
    n_clusters = trial.suggest_int('n_clusters', 3, 10)
    init_method = trial.suggest_categorical('init_method', ['k-means++', 'random'])
    n_init = trial.suggest_int('n_init', 1, 30)
    tol = trial.suggest_float('tol', 1e-4, 1e-1, log=True)
    algorithm = trial.suggest_categorical('algorithm', ['lloyd', 'elkan'])
    print("n_clusters 1", n_clusters)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_method,
        n_init=n_init,
        tol=tol,
        algorithm=algorithm,
        random_state=2
    )
    kmeans.fit(vectors)

    # Se o num de clusters que vem da função objetivo da silhouette for igual ao num de clusters ideal (5), passamos o num de clusters ideal para 6
    objective_value = best_silhouette - (math.log(1 + (abs(n_clusters_ant - n_clusters_ideal))) / n_clusters_ideal)

    return objective_value

def clustering_kmeans_nova_func_optuna(vectors, role, metric, nomeFichPickle, n_clusters_ant, best_silhouette):
    labels_kmeans = None
    centers_kmeans = None
    silhouette_kmeans = None
    n_clusters = None
    best_params = None
    parameters_trial = None

    # Se o modelo já foi treinado e guardado, carrega-o do ficheiro pickle
    if os.path.exists(nomeFichPickle):
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'rb') as file:
            data = pickle.load(file)
            labels_kmeans, centers_kmeans, params, silhouette_kmeans = data.get('labels'), data.get('centers'), data.get('params'), data.get('silhouette')
        print(f"Modelo já existente para {role}. Carregando os resultados guardados.")
        print(f"Hiperparâmetros carregados do pickle para {role}: {params}")
        print(f"Melhor Silhouette Score carregado do pickle para {role}: {silhouette_kmeans}")

    else:
        # Se o ficheiro pickle não existe, aplica o optuna
        study = optuna.create_study(direction="maximize")  # Queremos o melhor valor
        objective_func = lambda trial: objective_kmeans_nova_func(trial, vectors, n_clusters_ant, best_silhouette)
        study.optimize(objective_func, n_trials=1)

        best_trial = study.best_trial  # Obtém a melhor tentativa

        best_params = best_trial.params
        n_clusters = best_params['n_clusters']
        init = best_params['init_method']
        n_init = best_params['n_init']
        tol = best_params['tol']
        algorithm = best_params['algorithm']

        kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, tol=tol, algorithm=algorithm, random_state=2)
        kmeans.fit(vectors)

        silhouette_kmeans = metrics.silhouette_score(vectors, kmeans.labels_)

        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'wb') as file:
            pickle.dump({'labels': kmeans.labels_, 'centers': centers_kmeans, 'params': best_params,'silhouette': silhouette_kmeans}, file)

        print(f"Melhores parâmetros encontrados para {role} na Nova Função de Maximização:")
        print(best_params)
        print(f"Nova Silhouette para {role}: {silhouette_kmeans}")

        labels_kmeans, centers_kmeans = kmeans.labels_, kmeans.cluster_centers_

    return labels_kmeans, centers_kmeans, silhouette_kmeans, n_clusters, best_params

#CLUSTERING -> HIERÁRQUICO
def objective_hierarquico_silhouette(trial, vectors):
    n_clusters = trial.suggest_int('n_clusters', 2, 20)
    linkage_method = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])

    if linkage_method == 'ward':
        metric = 'euclidean'  # Se a linkage for 'ward', a métrica é sempre 'euclidean'
    else:
        metric = trial.suggest_categorical('metric', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'])

    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        metric=metric
    )
    hierarchical.fit(vectors)

    silhouette = metrics.silhouette_score(vectors, hierarchical.labels_)
    return silhouette


# Função de clustering Hierárquico com Optuna
def clustering_hierarquico_silhouette_optuna(vectors, role, nomeFichPickle):
    labels_hierarchical = None
    study = None
    silhouette_hierarquico = None
    n_clusters = None

    # Se o modelo já foi treinado e guardado, carrega-o do ficheiro pickle
    if os.path.exists(nomeFichPickle):
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'rb') as file:
            data = pickle.load(file)
            labels_hierarchical, params, silhouette_hierarquico = data.get('labels'), data.get('params'), data.get('silhouette')
        print(f"Modelo já existente para {role}. Carregando os resultados guardados.")
        print(f"Hiperparâmetros carregados do pickle para {role}: {params}")
        print(f"Melhor Silhouette Score carregado do pickle para {role}: {silhouette_hierarquico}")
    else:
        # Se o ficheiro pickle não existe, aplica o optuna
        # Garante que é executado até que os resultados estejam disponíveis
        while labels_hierarchical is None:
            study = optuna.create_study(direction="maximize")

            objective_func = lambda trial: objective_hierarquico_silhouette(trial, vectors)
            study.optimize(objective_func, n_trials=5)

            best_params = study.best_params
            n_clusters = best_params['n_clusters']
            linkage_method = best_params['linkage']

            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            hierarchical.fit(vectors)
            silhouette_hierarquico = metrics.silhouette_score(vectors, hierarchical.labels_)

            with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'wb') as file:
                pickle.dump({'labels': hierarchical.labels_, 'params': best_params,'silhouette': silhouette_hierarquico}, file)

            print(f"Melhores parâmetros encontrados para {role} na Função Silhouette:")
            print(best_params)
            print(f"Melhor Silhouette para {role}: {silhouette_hierarquico}")

            labels_hierarchical = hierarchical.labels_

    return labels_hierarchical, silhouette_hierarquico, n_clusters


#CLUSTERING -> HIERÁRQUICO
def objective_hierarquico_nova_func(trial, vectors, n_clusters_ant, best_silhouette):
    n_clusters_ideal = 5
    n_clusters = trial.suggest_int('n_clusters', 2, 20)
    linkage_method = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])

    if linkage_method == 'ward':
        metric = 'euclidean'  # Se a linkage for 'ward', a métrica é sempre 'euclidean'
    else:
        metric = trial.suggest_categorical('metric', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'])

    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        metric=metric
    )
    hierarchical.fit(vectors)

    objective_value = best_silhouette - (math.log(1 + (abs(n_clusters_ant - n_clusters_ideal))) / n_clusters_ideal)

    return objective_value

def clustering_hierarquico_nova_func_optuna(vectors, role, nomeFichPickle, n_clusters_ant, best_silhouette):
    labels_hierarchical = None
    study = None

    # Se o modelo já foi treinado e guardado, carrega-o do ficheiro pickle
    if os.path.exists(nomeFichPickle):
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'rb') as file:
            data = pickle.load(file)
            labels_hierarchical, params, silhouette_hierarquico = data.get('labels'), data.get('params'), data.get('silhouette')
            print(f"Modelo já existente para {role}. Carregando os resultados guardados.")
            print(f"Hiperparâmetros carregados do pickle para {role}: {params}")
            print(f"Melhor Silhouette Score carregado do pickle para {role}: {silhouette_hierarquico}")
    else:
        # Se o ficheiro pickle não existe, aplica o optuna
        # Garante que é executado até que os resultados estejam disponíveis
        while labels_hierarchical is None:
            study = optuna.create_study(direction="maximize")

            objective_func = lambda trial: objective_hierarquico_nova_func(trial, vectors, n_clusters_ant, best_silhouette)
            study.optimize(objective_func, n_trials=1)

            best_params = study.best_params
            n_clusters = best_params['n_clusters']
            linkage_method = best_params['linkage']

            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            labels_hierarchical = hierarchical.fit_predict(vectors)
            silhouette_hierarquico = metrics.silhouette_score(vectors, labels_hierarchical)

            with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'wb') as file:
                pickle.dump({'labels': labels_hierarchical, 'params': best_params, 'silhouette': silhouette_hierarquico}, file)

            print(f"Melhores parâmetros encontrados para {role} na Nova Função de Maximização:")
            print(best_params)
            print(f"Nova Silhouette para {role}: {silhouette_hierarquico}")

    return labels_hierarchical, silhouette_hierarquico

#CLUSTERING -> DBSCAN
def objective_dbscan_silhouette(trial, vectors):
    eps = trial.suggest_float('eps', 0.1, 1.0)
    min_samples = trial.suggest_int('min_samples', 1, 20)
    #min_samples = trial.suggest_int('min_samples', 3, 100)
    metric = trial.suggest_categorical('metric', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    leaf_size = trial.suggest_int('leaf_size', 10, 50)

    if algorithm == 'ball_tree' or algorithm == 'kd_tree' and metric == 'cosine':
        # Se o algoritmo for ball_tree ou kd_tree, a métrica não pode ser cosine
        metric = 'euclidean'

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size)
    dbscan.fit(vectors)


    try:
        silhouette = metrics.silhouette_score(vectors, dbscan.labels_)
    except ValueError:
        silhouette = -1.0

    return silhouette

def clustering_dbscan_silhouette_optuna(vectors, nomeFichPickle, role="None"):
    labels_dbscan = None
    study_dbscan = None
    silhouette_dbscan = None
    n_clusters = None
    # Se o modelo já foi treinado e guardado, carrega-o do ficheiro pickle
    if os.path.exists(nomeFichPickle):
        # Se o modelo já foi treinado e guardado, carrega-o do ficheiro pickle
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'rb') as file:
            dbscan = pickle.load(file)
            params = dbscan.get_params()
        silhouette = metrics.silhouette_score(vectors, dbscan.labels_)
        print(f"Modelo já existente para {role}. Carregando os resultados guardados.")
        print(f"Hiperparâmetros carregados do pickle para {role}: {params}")
        print(f"Melhor Silhouette Score carregado do pickle para {role}: {silhouette_dbscan}")
    else:
        # Se o ficheiro pickle não existe, aplica o optuna
        # Garante que é executado até que os resultados estejam disponíveis
        study_dbscan = optuna.create_study(direction='maximize')

        objective_func = lambda trial: objective_dbscan_silhouette(trial, vectors)
        study_dbscan.optimize(objective_func, n_trials=n_trials)

        best_params = study_dbscan.best_params
        eps = best_params['eps']
        min_samples = best_params['min_samples']
        metric = best_params['metric']
        algorithm = best_params['algorithm']
        leaf_size = best_params['leaf_size']

        if algorithm == 'ball_tree' or algorithm == 'kd_tree' and metric == 'cosine':
            # Se o algoritmo for ball_tree, a métrica não pode ser cosine
            metric = 'euclidean'

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size)
        dbscan.fit(vectors)
        silhouette_dbscan = metrics.silhouette_score(vectors, dbscan.labels_)

        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'wb') as file:
            pickle.dump(dbscan, file)

        print(f"Melhores parâmetros encontrados para {role} na Função Silhouette:")
        print(best_params)
        print(f"Melhor Silhouette para {role}: {silhouette_dbscan}")

    labels_dbscan = dbscan.labels_
    n_clusters = len(set(labels_dbscan))

    return labels_dbscan, silhouette_dbscan, n_clusters, study_dbscan

# V-measure

def objective_dbscan_vmeasure(trial, vectors, normalized_df):

    eps = trial.suggest_float('eps', 0.1, 1.0)
    min_samples = trial.suggest_int('min_samples', 20, 100)
    metric = trial.suggest_categorical('metric', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    leaf_size = trial.suggest_int('leaf_size', 10, 50)

    if algorithm == 'ball_tree' or algorithm == 'kd_tree' and metric == 'cosine':
        # Se o algoritmo for ball_tree ou kd_tree, a métrica não pode ser cosine
        metric = 'euclidean'

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size)
    dbscan.fit(vectors)

    normalized_df = normalized_df.assign(cluster_label=dbscan.labels_)

    try:
        v_measure = metrics.v_measure_score(normalized_df['trueLabel'], normalized_df['cluster_label'], beta=1)
    except ValueError:
        silhouette = -20.0

    return v_measure

# Função para treinar o modelo DBSCAN e guardar os resultados em um ficheiro pickle
def clustering_dbscan_vmeasure_optuna(vectors, nomeFichPickle, normalized_df, role="None"):
    """Receive vectors and do the clustering in DBSCAN (-1 cluster represents outliers)."""
    labels_dbscan = None
    study_dbscan = None
    n_clusters = None
    if not os.path.exists(nomeFichPickle):
        study_dbscan = optuna.create_study(direction="maximize")  # Queremos o melhor silhouette
        objective_func = lambda trial: objective_dbscan_vmeasure(trial, vectors, normalized_df)
        study_dbscan.optimize(objective_func, n_trials=n_trials, gc_after_trial=True)

        best_trial = study_dbscan.best_trial  # Obtém a melhor tentativa
        print(best_trial)

        best_params = best_trial.params

        eps=best_params['eps']
        min_samples=best_params['min_samples']
        metric=best_params['metric']

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        dbscan.fit(vectors)

        print(f"Melhores parâmetros encontrados para {role}:")
        print(best_params)

        # Save the model to a file
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'wb') as f:
            pickle.dump(dbscan, f)
            f.close()
    else:
        # Se o modelo já foi treinado e guardado, carrega-o do ficheiro pickle
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'rb') as file:
            dbscan = pickle.load(file)
        print(f"\nModelo já existente para {role}. Carregando os resultados guardados.")

    labels_dbscan = dbscan.labels_
    n_clusters = len(set(labels_dbscan))
    return labels_dbscan, n_clusters, study_dbscan

#CLUSTERING -> DBSCAN
def objective_dbscan_nova_func(trial, vectors, n_clusters_ant, best_silhouette):
    n_clusters_ideal = 5
    n_clusters_ant = n_clusters_ant if n_clusters_ant else 6  # Se n_clusters_ant é None, use 6 como valor padrão

    eps = trial.suggest_float('eps', 0.1, 1.0)
    min_samples = trial.suggest_int('min_samples', 20, 100)
    metric = trial.suggest_categorical('metric', ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    leaf_size = trial.suggest_int('leaf_size', 10, 50)

    if algorithm == 'ball_tree' or algorithm == 'kd_tree' and metric == 'cosine':
        # Se o algoritmo for ball_tree ou kd_tree, a métrica não pode ser cosine
        metric = 'euclidean'

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size)
    dbscan.fit(vectors)

    objective_value = best_silhouette - (math.log(1 + (abs(n_clusters_ant - n_clusters_ideal))) / n_clusters_ideal)

    return objective_value


def clustering_dbscan_nova_func_optuna(vectors, role, nomeFichPickle, n_clusters_ant, best_silhouette):
    labels_dbscan = None
    study_dbscan = None

    # Se o modelo já foi treinado e guardado, carrega-o do ficheiro pickle
    if os.path.exists(nomeFichPickle):
        with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'rb') as file:
            data = pickle.load(file)
            labels_dbscan, params, silhouette_dbscan = data.get('labels'), data.get('params'), data.get('silhouette')
        print(f"Modelo já existente para {role}. Carregando os resultados guardados.")
        print(f"Hiperparâmetros carregados do pickle para {role}: {params}")
        print(f"Melhor Silhouette Score carregado do pickle para {role}: {silhouette_dbscan}")
    else:
        # Se o ficheiro pickle não existe, aplica o optuna
        # Garante que é executado até que os resultados estejam disponíveis
        while labels_dbscan is None:
            study_dbscan = optuna.create_study(direction='maximize')

            objective_func = lambda trial: objective_dbscan_nova_func(trial, vectors, n_clusters_ant, best_silhouette)
            study_dbscan.optimize(objective_func, n_trials=n_trials)

            # Obtém os melhores parâmetros sugeridos pelo Optuna
            best_params = study_dbscan.best_params
            eps = best_params['eps']
            min_samples = best_params['min_samples']
            metric = best_params['metric']
            algorithm = best_params['algorithm']
            leaf_size = best_params['leaf_size']

            if algorithm == 'ball_tree' or algorithm == 'kd_tree' and metric == 'cosine':
                # Se o algoritmo for ball_tree, a métrica não pode ser cosine
                metric = 'euclidean'

            # Treina o modelo de clustering DBSCAN com os melhores parâmetros encontrados
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size)
            labels_dbscan = dbscan.fit_predict(vectors)

            # Verifica se o número de clusters é maior que 1
            if len(np.unique(labels_dbscan)) > 1:
                silhouette_dbscan = metrics.silhouette_score(vectors, labels_dbscan)
            else:
                # Se o número de clusters for 1, atribui rótulos padrão
                labels_dbscan = np.zeros_like(labels_dbscan)
                silhouette_dbscan = -1  #

            with open(os.path.join('./Resultados/' + diretoria, nomeFichPickle), 'wb') as file:
                pickle.dump(dbscan, file)

            print(f"Melhores parâmetros encontrados para {role} na Nova Função de Maximização:")
            print(best_params)
            print(f"Nova Silhouette para {role}: {silhouette_dbscan}")

    return labels_dbscan, silhouette_dbscan

# ##### Predict

# %%
# Igual ao kmeans.predict mas para o dbscan
def dbscan_predict(vectors, dbscan_model, new_data):
    y_new = np.ones(shape=len(new_data), dtype=int)*-1

    neigh = NearestNeighbors(n_neighbors=1).fit(vectors[dbscan_model.core_sample_indices_])
    (distances, indices) = neigh.kneighbors(new_data)
    for i, dist  in enumerate(distances):
        if dist < dbscan_model.eps:
            y_new[i] = dbscan_model.labels_[indices[i]]

    return y_new


# ### Labelling
# #### KeyBERT

def describe_clusters_kBERT(n_clusters, y_predicted, normalized_df, model, stopwords, n_grams, n_entries):
    # Inicializa o modelo KeyBERT
    kw_model = KeyBERT(model=model)

    # Cria um DataFrame vazio para armazenar as labels do cluster e palavras-chave do kBERT
    df_teste = normalized_df.copy()
    df_teste["predicted"] = y_predicted
    df_teste["utterance"] = df_teste["utterance"].astype(str)

    # Lista para armazenar as labels do cluster e palavras-chave do kBERT
    list_kBERT = []
    list_clusters = []  # clusters

    # ids dos clusters, por ordem crescente
    cluster_ids = sorted(list(set(y_predicted)))

    # Percorre os clusters
    for cluster_id in range(n_clusters):
        predicted_df = df_teste[df_teste["predicted"] == cluster_id]
        corpus = predicted_df['utterance']
        corpus = [str(c) for c in corpus]

        # Extrai palavras-chave - KeyBERT
        keywords = kw_model.extract_keywords(" ".join(corpus), keyphrase_ngram_range=(1, n_grams), stop_words=stopwords, highlight=False, top_n=n_entries)
        keywords_list = [w[0] for w in keywords]
        keywords_set = set(keywords_list)
        keywords_list = list(keywords_set)

        # Adiciona "Outros {cluster_id}" se a lista de palavras-chave estiver vazia
        if not keywords:
            keywords = [(f"Outros {cluster_id}", 1.0)]

        print(f"Cluster {cluster_id} - Keywords: {keywords}")

        # Armazena apenas as palavras-chave e não os scores
        keywords_list = [w[0] for w in keywords]
        list_kBERT.append(keywords_list)
        list_clusters.append(f"Cluster {cluster_id}")

    # Cria DataFrames a partir das listas e concatena-os
    df1 = pd.DataFrame(list_clusters, columns=["clusters"])
    df2 = pd.DataFrame(list_kBERT, columns=["labels"])
    df_labels = pd.concat([df1, df2], axis=1)

    print(df_labels)
    return df_labels

# #### Verb Phrases
def describe_clusters_verbs(nlp, n_clusters, normalized_df, y_predicted):
    # Cria uma cópia do DataFrame normalizado
    df_teste = normalized_df.copy()
    df_teste["predicted"] = y_predicted

    # Converte as utterances para strings
    df_teste["utterance"] = df_teste["utterance"].astype(str)

    # Listas para armazenar as labels do cluster, verbos e sintagmas
    list_clusters = []
    list_verbs = []
    list_sintagmas = []

    for i, utt in enumerate(list(df_teste["utterance"])):
        doc = nlp(utt)
        sintagma = " "
        for token in doc:
            # Verificar se o token é uma palavra ou um nome de utilizador
            if token.text[0] >= 'A' and token.text[0] <= 'z' or token.text[0] == '@':
                if token.dep_ == 'ROOT':
                    sintagma = sintagma + " " + token.lemma_
                    for child in token.children:
                         # Verifica se o filho é uma palavra ou um nome de utilizador e não é um sujeito
                        if (
                            (child.text[0] >= 'A'
                            and child.text[0] <= 'z' or child.text[0] == '@')
                            and child.dep_ != "nsubj"
                        ):
                            sintagma = sintagma + " " + child.lemma_

        list_sintagmas.append(sintagma)

    # Atualiza as utterances no DataFrame
    df_teste["utterance"] = list_sintagmas
    # Remove linhas com utterances vazias
    df_teste["utterance"].replace(" ", np.nan, inplace=True)
    df_teste.dropna()

    for p in range(n_clusters):
        predicted_df = df_teste[df_teste['predicted'] == p]
        corpus = predicted_df['utterance']

        df_corpus = pd.DataFrame(corpus)
        df_corpus.columns = ['docs_in_cluster']

        # Verificar se a lista não está vazia antes de aceder o primeiro elemento
        count_values = df_corpus.value_counts().index.tolist()
        if count_values:
            count_values = count_values[0][0]
        else:
            count_values = "N/A"
        # Adiciona a label do cluster à lista
        list_clusters.append(f"Cluster {p}")
        # Adiciona contagem de verbos à lista
        list_verbs.append(count_values)


    # Cria DataFrames a partir das listas e concatena-os
    df1 = pd.DataFrame(list_clusters, columns=["clusters"])
    df2 = pd.DataFrame(list_verbs, columns=["labels"])
    df2["labels"] = df2["labels"].str.cat(
        df2.groupby("labels").cumcount().astype(str).str.replace("0", ""), sep=" "
    )

    df_labels = pd.concat([df1, df2], axis=1)

    print(f"Clusters Verbs: {df_labels}")
    print(f"Labels Verbs: {df_labels['labels'].to_dict()}")

    return df_labels

# Mais perto do centro
def describe_clusters_closest(normalized_df, y_predicted, vectors, centers, n_clusters):
    docs = normalized_df["utterance"]
    order_centroids = centers.argsort()[:, ::-1]
    closest, _ = metrics.pairwise_distances_argmin_min(
        centers, vectors, metric="cosine"
    )
    print("closest", closest)

    df_teste = pd.DataFrame()  # utterance and y_predicted
    df_teste["predicted"] = y_predicted
    df_teste["corpus"] = normalized_df["utterance"]

    list_clusters = []  # clusters
    list_closest_doc = []  # docs mais proximos do centroide
    for p in range(n_clusters):
        predicted_df = df_teste[df_teste["predicted"] == p]
        corpus = predicted_df["corpus"]

        closest_index = closest[p]
        closest_doc = docs.iloc[closest_index]
        list_closest_doc.append(closest_doc)

        list_clusters.append(f"Cluster {p}")

    df1 = pd.DataFrame(list_clusters, columns=["clusters"])
    df2 = pd.DataFrame(list_closest_doc, columns=["labels"])
    df2["labels"] = df2["labels"].str.cat(
        df2.groupby("labels").cumcount().astype(str).str.replace("0", ""), sep=" "
    )

    df_labels = pd.concat([df1, df2], axis=1)

    print("Cluster Describe Closest Document")
    print(df_labels)
    return df_labels

def describe_clusters_LLM(y_predicted, normalized_df, ollama_url):
    df_teste = normalized_df.copy()
    df_teste["predicted"] = y_predicted
    df_teste["utterance"] = df_teste["utterance"].astype(str)

    list_LLM = []
    list_clusters = [] 

    cluster_ids = sorted(list(set(y_predicted)))
    
    # Inicializa o LLM fora do loop para ser mais rápido
    llm = Ollama(base_url=ollama_url, model="llama3", temperature=0.1, num_predict=50)
    
    print(f"\n[LLM] A gerar rótulos para {len(cluster_ids)} clusters...")

    for cluster_id in cluster_ids:
        predicted_df = df_teste[df_teste["predicted"] == cluster_id]
        corpus = predicted_df['utterance'].tolist()
        corpus = [str(c).replace("'","") for c in corpus]
        
        subset_length = min(30, len(corpus))
        subset = random.sample(corpus, subset_length)

        prompt = "Por favor, atribua um rótulo (label), em PT-PT, que capture as ações principais deste grupo de falas: '"
        prompt_end = "'. Responda apenas com o rótulo, use este formato Rótulo: {seu_rótulo}, e use no máximo três palavras."
        data = ';'.join(subset)
        
        try:
            response = llm.generate([prompt + data + prompt_end])
            keywords = response.generations[0][0].text

            pattern = r"Label:\s*(.+?)(?:[\n.:;,]|$)"
            pattern2 = r"\s*(.+?)(?:[\n.:;,]|$)"
            label = keywords
            match = re.search(pattern, keywords)
            split_keywords = keywords.split(':', 1)

            if match:
                label = match.group(1).strip().replace('"', '')
            elif len(split_keywords) > 1:
                label_part = split_keywords[1]
                match2 = re.search(pattern2, label_part)
                if match2:
                    label = match2.group(1).strip().replace('"', '')
                else:
                    label = label_part.strip().replace('"', '') 
            else:
                label = label.strip().replace('"', '')
                
        except Exception as e:
            print(f"Erro ao comunicar com Ollama no cluster {cluster_id}: {e}")
            label = f"Tópico {cluster_id}"

        # -----------------------------------------------------
        # O TRUQUE ANTI-LOOP INFINITO
        # Se a label já existir, acrescentamos um número à frente
        # -----------------------------------------------------
        base_label = label
        counter = 2
        while [label] in list_LLM:
            label = f"{base_label} {counter}"
            counter += 1

        list_LLM.append([label])
        list_clusters.append(cluster_id)
        
        print(f" -> Cluster {cluster_id}: {label}")

    df1 = pd.DataFrame({"clusters": [f"Cluster {x}" for x in list_clusters]})
    df2 = pd.DataFrame(list_LLM, columns=["labels"])
    df_labels = pd.concat([df1, df2], axis=1).sort_values(by=['clusters'])
    
    return df_labels

# Accuracy
def calcular_accuracy_transicoes(normalized_df, flow, names_speaker):
    dialogue_ids = normalized_df["dialogue_id"].tolist()

    num_dialogos = len(set(dialogue_ids))  # Número de diálogos distintos
    print("num_dialogos", num_dialogos)
    num_utterances = len(normalized_df.index)  # Número total de utterances
    print("num_utterances", num_utterances)

    transicoes_previstas = 0
    transicoes_totais = 0
    count_inter = 0
    count_eod = 0
    count_sod = 0
    for dialogue_id in set(dialogue_ids): # Percorrer todos os dialogos

        dialogue_utterances = normalized_df[normalized_df["dialogue_id"] == dialogue_id] # Linhas com os dados das utterances do dialogo atual
        dialogue_utterances = dialogue_utterances.sort_values(by='turn_id').reset_index() # Ordenar por turn para obter as transições corretas

        for i in range(len(dialogue_utterances["clusters_final"])-1): # Percorrer todas as utterances
            for speaker in speakers:
                if dialogue_utterances['Speaker'][i] == speaker:
                    #print(dialogue_utterances)
                    #print(len(dialogue_utterances))
                    #print(len(names_speaker))
                    label_atual = names_speaker[speaker][dialogue_utterances["clusters_final"][i]]

            for speaker in speakers:
                if dialogue_utterances['Speaker'][i + 1] == speaker:
                    label_proximo = names_speaker[speaker][dialogue_utterances["clusters_final"][i + 1]]

            # Se o label das utterances for -1 então a transição não pode ser prevista
            if dialogue_utterances["clusters_final"][i] == -1 or dialogue_utterances["clusters_final"][i + 1] == -1:
                transicoes_totais += 1
                continue

            # Verificar transição
            if flow.get_edge(("\"" + label_atual + "\""), ("\"" + label_proximo + "\"")):
                count_inter += 1
                transicoes_previstas += 1
            transicoes_totais += 1

        # Verificar que a primeira utterance não tem label -1 e se a transição SOD -> first é prevista
        if dialogue_utterances["clusters_final"][0] != -1:
            for speaker in speakers:
                if dialogue_utterances['Speaker'][0] == speaker:
                    #print("Speaker:", speaker)
                    #print("Dialogue utterances:", dialogue_utterances)
                    #print("Cluster index:", dialogue_utterances["clusters_final"][0])
                    #print("Speaker names list:", names_speaker.get(speaker, []))

                    first_utterance = names_speaker[speaker][dialogue_utterances["clusters_final"][0]]

            if flow.get_edge("SOD", ("\"" + first_utterance + "\"")):
                count_sod += 1
                transicoes_previstas += 1

        # Verificar que a ultima utterance não tem label -1 e se a transição last -> EOD é prevista
        if dialogue_utterances["clusters_final"][len(dialogue_utterances["clusters_final"])-1] != -1:
            
            for speaker in speakers:
                if dialogue_utterances['Speaker'][len(dialogue_utterances["clusters_final"])-1] == speaker:
                    last_utterance = names_speaker[speaker][dialogue_utterances["clusters_final"][len(dialogue_utterances["clusters_final"])-1]]
            

            if flow.get_edge(("\"" + last_utterance + "\""), "EOD"):
                count_eod += 1
                transicoes_previstas += 1

        transicoes_totais += 2

    print(count_sod, count_inter, count_eod)
    accuracy = transicoes_previstas / transicoes_totais

    return accuracy

def save_to_excel(output_file, role, metric, params, objective_value, clusters, create_new_sheet=True):
    # Cria um DataFrame
    data = {
        'Role': [role],
        'Metric': [metric],
        'Parameters': [str(params)],
        'Clusters': [clusters],
        'Objective Value': [objective_value]
    }

    df = pd.DataFrame(data)

    # Tenta ler o ficheiro Excel existente, se não existir, cria um novo
    try:
        existing_df = pd.read_excel(output_file, sheet_name='results')
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    # Adiciona ou cria uma nova planilha
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        if create_new_sheet or 'results' not in writer.sheets:
            df.to_excel(writer, sheet_name='results', index=False)
        else:
            df.to_excel(writer, sheet_name='results', index=False, startrow=len(existing_df)+1)

# Sentimento
def shade_color(color, intensity):
    try:
        named_color = mpl.colors.CSS4_COLORS[color]
        r, g, b = [int(named_color[i:i+2], 16) for i in (1, 3, 5)]
    except KeyError:
        # Supomos que o formato seja RGB
        r, g, b = [int(val) for val in color.split(",")]

    # Ajustar a intensidade para criar tom mais escuro ou mais claro
    r = max(0, min(255, int(r + intensity * (255 - r))))
    g = max(0, min(255, int(g + intensity * (255 - g))))
    b = max(0, min(255, int(b + intensity * (255 - b))))

    # Converter de volta para formato hexadecimal
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def colorize_sentiment(sentiment, sentiment_in_flow, sentiment_system, df_final, sentiment_cluster):
    # Se o sentimento estiver no fluxo
    if sentiment_in_flow == "cluster":
        # Se o sentimento do sistema for considerado (sentiment_system == True)
        if sentiment_system:
            # Verifica se os sentimentos originais do sistema foram armazenados e restaura-os
            if 'Original_Sentiment_System' in df_final.columns:
                df_final.loc[df_final['Speaker'] == 'SYSTEM', sentiment_cluster] = df_final['Original_Sentiment_System']
                # Remove a coluna temporária após restaurar os valores originais
                df_final.drop(columns=['Original_Sentiment_System'], inplace=True)

            if pd.notna(sentiment):  # Se o sentimento não for NaN
                sentiment = float(sentiment)

                # Definir cores com base no valor do sentimento
                if sentiment > 0.6:
                    intensity = min(max((sentiment - 0.6) * 2, 0.4), 1.0)
                    intensity = 1 - intensity
                    return shade_color("darkgreen", intensity)
                elif 0.4 <= sentiment <= 0.6:
                    return "gold"
                else:
                    intensity = min(max(sentiment, 0.0), 0.4)
                    return shade_color("darkred", intensity)

        # Caso o sentimento do sistema não seja considerado (sentiment_system == False)
        else:
            # Armazena os sentimentos originais, se ainda não foram armazenados
            if 'Original_Sentiment_System' not in df_final.columns:
                df_final['Original_Sentiment_System'] = df_final[sentiment_cluster]

            # Define o sentimento do sistema como 0
            df_final.loc[df_final['Speaker'] == 'SYSTEM', sentiment_cluster] = 0

            # Se o sentimento for None ou zero
            if pd.isna(sentiment) or sentiment == 0:
                return "black"

            sentiment = float(sentiment)

            # Definir cores com base no valor do sentimento
            if sentiment > 0.6:
                intensity = min(max((sentiment - 0.6) * 2, 0.4), 1.0)
                intensity = 1 - intensity
                return shade_color("darkgreen", intensity)
            elif 0.4 <= sentiment <= 0.6:
                return "gold"
            elif sentiment > 0 and sentiment <= 0.4:
                intensity = min(max(sentiment, 0.0), 0.4)
                return shade_color("darkred", intensity)

    # Se o sentimento não estiver no fluxo ou for None, retorna "black"
    return "black"

def colorize_sentiment_v2(sentiment):
    # Define o colormap "RdYlGn" que vai de vermelho (negativo) a verde (positivo)
    cmap = plt.get_cmap("RdYlGn")
    rgba_color = cmap(sentiment)

    # Converto RGBA para formato hexadecimal
    return "#{:02X}{:02X}{:02X}".format(int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255))


if filename[-4:] == ".csv":
    dados = pd.read_csv(filename, sep=',')
else:
    dados = pd.read_excel(filename)

if TIPO_DATASET == "GENERICO":
    MODELOS = ['gemma3:4b']
    LINGUA = 'English'
    TEMP = 0.8

    modelo_limpo = MODELOS[0].replace(":", "-") 
    NOME_DO_GRUPO = f'All Domains ({modelo_limpo} _ T={TEMP})'
    FILTRO_EXTRA_COLUNA = 'Dominio'
    FILTRO_EXTRA_VALOR = ['Desporto', 'Música', 'Cidades'] 

    # Filtra pelas colunas que só existem no dataset LLM
    if all(col in dados.columns for col in ['Modelo', 'Lingua', 'Temperatura']):
        dados = dados[
            (dados['Modelo'].isin(MODELOS)) &
            (dados['Lingua'] == LINGUA) &
            (dados['Temperatura'] == TEMP)
        ]
    else:
        print(">>> Aviso: Colunas Modelo, Lingua, Temp não encontradas. Filtro ignorado.")
        NOME_DO_GRUPO = "All Domains"

    if FILTRO_EXTRA_COLUNA != "" and FILTRO_EXTRA_COLUNA in dados.columns:
        dados = dados[dados[FILTRO_EXTRA_COLUNA].isin(FILTRO_EXTRA_VALOR)]

    # Sobrescreve o Speaker
    #dados['Speaker'] = NOME_DO_GRUPO

# ==============================================================

dados_test = dados.copy()
print("Colunas disponíveis:", dados.columns.tolist())

# Se o dataset não tiver a coluna 'Tipo', criamos uma e assumimos que é tudo 'fala'
if "Tipo" not in dados.columns:
    print("coluna 'Tipo' não encontrada. A criar coluna padrão (tudo = 'fala').")
    dados["Tipo"] = "fala"

# Se o dataset não tiver a coluna 'Ação', criamos vazia para não dar erro depois
if "Ação" not in dados.columns:
    dados["Ação"] = np.nan

# Filtrar apenas ações (agora seguro porque a coluna 'Tipo' existe sempre)
diag_acoes = dados[dados["Tipo"].astype(str).str.lower() == "ação"]

if not diag_acoes.empty:
    print("\nPrimeiras 5 linhas de AÇÕES (Raw Data):")
    # Tenta mostrar as colunas mais prováveis
    cols_to_show = [c for c in ['Speaker', 'Speaker_pessoa', 'Partido', 'Ação', 'acoes_simples'] if c in dados.columns]
    print(diag_acoes[cols_to_show].head(5))
    
    print("\nValores únicos na coluna 'Speaker' para ações:")
    if 'Speaker' in dados.columns:
        print(diag_acoes['Speaker'].unique())
else:
    print("Nenhuma linha do tipo 'ação' encontrada.")

print("----------------------------\n")

normalized_df = normalize_dataset(dados, regex=True, removeGreetings=False, speaker='both')

if TIPO_DATASET == "POLITICA":  
    # Detetar Ano no nome do ficheiro
    nome_ficheiro_lower = filename.lower()
    ano_detetado = re.search(r'(\d{4})', nome_ficheiro_lower)

    if ano_detetado:
        ano_str = ano_detetado.group(1)
        print(f"   Ano detetado: {ano_str}")
        if ano_str in ["2021", "2022", "2023"]:
            GOV_PARTIDO_DO_ANO = "PS"
        else: # 2024, 2025
            GOV_PARTIDO_DO_ANO = "PSD" 
    else:
        print("   AVISO: Ano não detetado. A assumir 'GOV' como PS.")
        GOV_PARTIDO_DO_ANO = "PS"
        ano_str = "ano_desconhecido"

    print(f"   Governo definido como: {GOV_PARTIDO_DO_ANO}")

    normalized_df['Speaker'] = normalized_df.apply(
        mapear_entidade_dinamica,
        axis=1,
        ano=ano_str,
        gov_partido=GOV_PARTIDO_DO_ANO,
        entidades_map=ENTIDADES_A_PROCESSAR,
        incluir_outros=INCLUIR_OUTROS
    )

    # Filtrar
    if not INCLUIR_OUTROS:
        normalized_df.dropna(subset=['Speaker'], inplace=True)

else:
    # Limpar
    normalized_df['Speaker'] = normalized_df['Speaker'].astype(str).str.strip()
    normalized_df = normalized_df[~normalized_df['Speaker'].isin(['nan', 'NaN', 'None', '', 'float'])]

    # Mapear (Automaise -> SYSTEM, Cliente -> USER)
    MAPA_GENERICO = {
        'Cliente': 'USER', 
        'User': 'USER',
        'Automaise': 'SYSTEM', 
        'Assistente': 'SYSTEM'
    }
    normalized_df['Speaker'] = normalized_df['Speaker'].replace(MAPA_GENERICO)

# Atualizar lista final (para os dois modos)
speakers = normalized_df['Speaker'].unique().tolist()
if 'Acao' in speakers: speakers.remove('Acao')

"""def group_consecutive_speakers(dialogue_df):
    dialogue_df["Speaker_Group"] = (dialogue_df["Speaker"] != dialogue_df["Speaker"].shift()).cumsum()
    grouped_df = dialogue_df.groupby(["dialogue_id", "Speaker_Group", "Speaker"]).agg({
        "utterance": " ".join,
        "Sentiment": "first",
        "Median_Binary": "first",
        "trueLabel": "first"
    }).reset_index()
    grouped_df = grouped_df.drop(columns=["Speaker_Group"], errors="ignore")
    return grouped_df """
print(normalized_df.head())

def group_consecutive_speakers(dialogue_df):
    dialogue_df["Speaker_Group"] = (dialogue_df["Speaker"] != dialogue_df["Speaker"].shift()).cumsum()
    
    # Lista de colunas que queremos agregar
    agg_dict = {
        "utterance": " ".join,
        "trueLabel": "first"
    }

    # Adiciona colunas opcionais se existirem
    optional_columns = ["Sentiment", "Median_Binary", "Emotion"]
    for col in optional_columns:
        if col in dialogue_df.columns:
            agg_dict[col] = "first"

    grouped_df = dialogue_df.groupby(["dialogue_id", "Speaker_Group", "Speaker"]).agg(agg_dict).reset_index()
    grouped_df = grouped_df.drop(columns=["Speaker_Group"], errors="ignore")
    return grouped_df


if same_speakers:
    grouped_dialogues = []
    for dialogue_id in normalized_df["dialogue_id"].unique():
        dialogue_df = normalized_df[normalized_df["dialogue_id"] == dialogue_id]
        
        grouped_dialogue_df = group_consecutive_speakers(dialogue_df)
        grouped_dialogues.append(grouped_dialogue_df)
        

    normalized_df = pd.concat(grouped_dialogues, ignore_index=True)
    print(normalized_df.head())
print(normalized_df.head())

normalized_df = normalize_dataset(normalized_df, regex=True, removeGreetings=False, speaker='both')


normalized_df_speaker= {}
utterances_speaker= {}

speakers = normalized_df[normalized_df['Speaker'] != 'Acao']['Speaker'].unique().tolist()

for speaker in speakers:
    normalized_df_speaker[speaker] = normalized_df[normalized_df['Speaker'] == speaker]
    utterances_speaker[speaker] = normalized_df_speaker[speaker]["utterance"].tolist()

utterances = normalized_df[normalized_df['Tipo'] == 'fala']["utterance"].tolist()

print (normalized_df.head())
acts = False
if 'trueLabel' in normalized_df.columns:
    acts = True

# Vetorização
model_ml = SentenceTransformer(MODEL_ML, device='cpu')

speaker_files = {}
vectors_speaker={}

for speaker in speakers:
    #speaker_files[speaker] = f'{MODEL_ML}_{algorithm}_{filename.split(".")[0]}_{metric_to_optimize}_{str(id_max)}_vectors_{speaker}.pkl'
    speaker_files[speaker] = f'vetores_{speaker}.pkl'

if not all(os.path.exists(os.path.join('./Resultados/' + diretoria, file)) for file in speaker_files.values()):
    for speaker in speakers:
        vectors_speaker[speaker]= use_sentence_transformer(utterances_speaker[speaker], model_ml)
    vectors = use_sentence_transformer(utterances, model_ml)
    
    normalized_df['vectors'] = [vectors[i] for i in range(len(vectors))]
    normalized_df = compute_weighted_mean(normalized_df, id_max)
    
    for speaker in speakers:
        normalized_df_speaker[speaker] = normalized_df[normalized_df['Speaker'] == speaker]
        vectors_speaker[speaker] = np.vstack(np.array(normalized_df_speaker[speaker]['vectors_weight']))

        with open(os.path.join('./Resultados/' + diretoria, speaker_files[speaker]), 'wb') as file:
                pickle.dump(vectors_speaker[speaker], file)
else:
    for speaker in speakers:
        with open(os.path.join('./Resultados/' + diretoria, speaker_files[speaker]), 'rb') as file:
            vectors_speaker[speaker] = pickle.load(file)

# Clustering

def reindex_labels(seq, dic, names_clusters):
        unique = {}
        new_seq = []
        current = 0

        for val in seq:
            if val not in unique:
                unique[val] = current
                current += 1
            new_seq.append(unique[val])
        

        for val in unique.keys():
            dic[names_clusters[val]] = unique[val]
        print ("dic",dic)
        
        for k,v in dic.items():
            if v > 21:
                dic[k] = unique[v]
        print ("dic2",dic)
        return new_seq, dic

# Inicialização de variáveis comuns
excel_speaker = {}
pickle_filename_speakers = {}
pickle_utterances_filenames_speaker = {}
y_predicted_speaker = {}
n_clusters_speaker = {}
params_speaker = {}
centers_speaker = {}
clusters_speaker = {}
metric_speaker = {}
labels_speaker = {}

# Definir nomes de ficheiros
for speaker in speakers:
    excel_speaker[speaker] = f'{algorithm}_{filename.split(".")[0]}_{speaker}_{metric_to_optimize}_{str(id_max)}.xlsx'
    pickle_filename_speakers[speaker] = f'{algorithm}_{speaker}_{filename.split(".")[0]}_{metric_to_optimize}_{str(id_max)}.pkl'
    pickle_utterances_filenames_speaker[speaker] = f'{algorithm}_{speaker}_{filename.split(".")[0]}_utt_{metric_to_optimize}_{str(id_max)}.pkl'

def create_clusters_dic(y_predicted, utterances, label):
    clusters_dic = {} 
    for i, u in enumerate(utterances):
        cid = y_predicted[i]
        if cid not in clusters_dic:
            clusters_dic[cid] = []
        clusters_dic[cid].append(u)
    return {k: v for k, v in clusters_dic.items()}

# MODO ATIVO: keywords
if labelling == "keywords":    
    for speaker in speakers:
        # 1. Obter dados e limpar coluna alvo
        df_atual = normalized_df_speaker[speaker].copy()
        coluna_alvo = 'speech_act_gerado'
        
        if coluna_alvo not in df_atual.columns:
            print(f"oluna '{coluna_alvo}' não encontrada para {speaker}. Usando 'Indefinido'.")
            df_atual[coluna_alvo] = "Indefinido"
        
        df_atual[coluna_alvo] = df_atual[coluna_alvo].fillna('Outros').astype(str).str.strip()
        
        # 2. Transformar categorias em números
        codes, uniques = pd.factorize(df_atual[coluna_alvo])
        
        # 3. Guardar nas variáveis globais
        y_predicted_speaker[speaker] = codes
        n_clusters_speaker[speaker] = len(uniques)
        
        # Cria o dicionário de Labels (ex: {0: "Cooperação", 1: "Conflito"})
        labels_speaker[speaker] = {i: label for i, label in enumerate(uniques)}
        
        # Atualizar DataFrame original
        normalized_df_speaker[speaker]['clusters_speaker_' + speaker] = codes
        
        params_speaker[speaker] = {"method": "keywords_excel", "n_clusters": len(uniques)}
        
        # 4. Criar dicionário de utterances e guardar pickle
        clusters_speaker[speaker] = create_clusters_dic(codes, utterances_speaker[speaker], speaker)
        
        with open(os.path.join('./Resultados/' + diretoria, pickle_utterances_filenames_speaker[speaker]), 'wb') as file:
            pickle.dump(clusters_speaker[speaker], file)
            
        print(f"   [{speaker}] Tipos identificados: {list(uniques)}")

# MODO ATIVO: CLUSTERING
else:    
    for speaker in speakers:
        # SE FOR BERTOPIC
        if algorithm == "bertopic":
            print(f"\n>>> [{speaker}] A usar BERTopic (Clustering + Labelling Automático)...")
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer
            from bertopic.representation import KeyBERTInspired
            
            # Ignorar palavras de ligação
            vectorizer_model = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2))
            
            # Forçar a escolha de palavras com verdadeiro significado (Keywords)
            representation_model = KeyBERTInspired()
            
            # Inicializa o modelo (com o filtro e o KeyBERT)
            topic_model = BERTopic(
                language="multilingual",
                min_topic_size=5, 
                nr_topics=20, # Força o modelo a tentar encontrar até 20 temas distintos
                vectorizer_model=CountVectorizer(stop_words=stopwords, ngram_range=(1, 2)),
                representation_model=representation_model
            )
            
            textos = utterances_speaker[speaker]
            textos_str = [str(doc) for doc in textos]
            
            # Obter o dataframe original deste speaker para servir de base
            df_speaker = normalized_df_speaker[speaker].copy()
            
            # 2. Treinar o modelo com TODAS as frases (sem filtrar nada antes)
            # Isto garante que o output tem o MESMO tamanho do input
            topics, probs = topic_model.fit_transform(textos_str)
            
            y_predicted_speaker[speaker] = np.array(topics)
            n_clusters_speaker[speaker] = len(set(topics)) - (1 if -1 in topics else 0)
            
            # Extrai os nomes dos tópicos automaticamente
            topic_info = topic_model.get_topic_info()
            labels_speaker[speaker] = {}
            for index, row in topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id != -1:
                    # Limpa o nome (ex: "0_fatura_pagamento_mes" -> "fatura - pagamento")
                    palavras_validas = [word for word in row['Name'].split('_')[1:4] if word.strip()]
                    nome_limpo = " - ".join(palavras_validas)
                    labels_speaker[speaker][topic_id] = nome_limpo
            
            # Removida a linha do [-1] para não desalinhar a matriz!
            centers_speaker[speaker] = None # BERTopic não usa centros matemáticos
            normalized_df_speaker[speaker] = df_speaker.assign(clusters_speaker=y_predicted_speaker[speaker])
            
            # Guarda as utterances
            clusters_speaker[speaker] = create_clusters_dic(y_predicted_speaker[speaker], textos_str, speaker)            
            with open(os.path.join('./Resultados/' + diretoria, pickle_utterances_filenames_speaker[speaker]), 'wb') as file:
                pickle.dump(clusters_speaker[speaker], file)

        # SE FOR KMEANS OU DBSCAN
        else:
            if metric_to_optimize == "vmeasure":
                if algorithm == "dbscan":
                    y_predicted_speaker[speaker], n_clusters_speaker[speaker], params_speaker[speaker] = clustering_dbscan_vmeasure_optuna(vectors_speaker[speaker], pickle_filename_speakers[speaker], normalized_df_speaker[speaker], role= speaker)
                else:
                    y_predicted_speaker[speaker], centers_speaker[speaker], n_clusters_speaker[speaker], params_speaker[speaker] = clustering_kmeans_vmeasure_optuna(vectors_speaker[speaker], normalized_df_speaker[speaker], role= speaker, nomeFichPickle=pickle_filename_speakers[speaker],val=val)
            elif metric_to_optimize == "silhouette":
                if algorithm == "dbscan":
                    y_predicted_speaker[speaker], metric_speaker[speaker], n_clusters_speaker[speaker], params_speaker[speaker] = clustering_dbscan_silhouette_optuna(vectors_speaker[speaker], nomeFichPickle=pickle_filename_speakers[speaker], role= speaker)
                else:
                    y_predicted_speaker[speaker], centers_speaker[speaker], metric_speaker[speaker], n_clusters_speaker[speaker], params_speaker[speaker] = clustering_kmeans_silhouette_optuna(vectors_speaker[speaker], role= speaker, metric='Silhouette', nomeFichPickle=pickle_filename_speakers[speaker])
            
            # Guardar clusters
            if os.path.exists(pickle_utterances_filenames_speaker[speaker]):
                with open(os.path.join('./Resultados/' + diretoria, pickle_utterances_filenames_speaker[speaker]), 'rb') as file:
                    data = pickle.load(file)
                    clusters_speaker[speaker] = data 
            else:
                clusters_speaker[speaker] = create_clusters_dic(y_predicted_speaker[speaker], utterances_speaker[speaker], speaker)
                with open(os.path.join('./Resultados/' + diretoria, pickle_utterances_filenames_speaker[speaker]), 'wb') as file:
                    pickle.dump(clusters_speaker[speaker], file)

def gerar_heatmap(matriz_df, diretorio_saida, nome_ficheiro, titulo="Transition Probability Matrix"):
    """
    Gera e guarda um Heatmap limpo e estético em inglês.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    print(f"\nA gerar Heatmap: {nome_ficheiro}...")

    # Fazer uma cópia para não estragar o grafo da interface!
    matriz_plot = matriz_df.copy()

    # Limpar os nomes gigantes (Ficar só com o que está depois do "->")
    clean_index = matriz_plot.index.to_series().apply(lambda x: str(x).split('->')[-1].strip())
    clean_columns = matriz_plot.columns.to_series().apply(lambda x: str(x).split('->')[-1].strip())
    
    matriz_plot.index = clean_index
    matriz_plot.columns = clean_columns

    # Filtrar linhas e colunas que sejam APENAS 0.00
    matriz_plot = matriz_plot.loc[(matriz_plot > 0).any(axis=1)] 
    matriz_plot = matriz_plot.loc[:, (matriz_plot > 0).any(axis=0)] 

    if matriz_plot.empty:
        print("Heatmap ignorado: A matriz ficou vazia após remover os zeros.")
        return

    tamanho_figura = max(8, len(matriz_plot.columns) * 0.8)
    plt.figure(figsize=(tamanho_figura, tamanho_figura)) 
    
    # 4. Desenhar o Heatmap
    ax = sns.heatmap(
        matriz_plot, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues",
        cbar_kws={'label': 'Transition Probability'},
        linewidths=1,
        linecolor='white',
        annot_kws={"size": 10},
        square=True
    )
    
    plt.title(titulo, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Target State", fontsize=13, fontweight='bold', labelpad=15)
    plt.ylabel("Source State", fontsize=13, fontweight='bold', labelpad=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    
    # Guardar Ficheiro
    caminho_completo = os.path.join('./Resultados', diretorio_saida, nome_ficheiro)
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Heatmap guardado com sucesso em: {caminho_completo}")

# CÁLCULO DE MÉTRICAS

cluster_counts_speaker = {}
silhouette_speaker = {}
aux_y_predicted_speaker = {}

# 1. Concatenar tudo num único DataFrame (df_final)
df_final = pd.concat([normalized_df_speaker[speaker] for speaker in speakers])

# 2. Calcular Contagens e Silhouette
for speaker in speakers:
    cluster_col_name = f'clusters_speaker_{speaker}'

    # Garantir que a coluna existe
    if cluster_col_name not in normalized_df_speaker[speaker].columns and speaker in y_predicted_speaker:
        normalized_df_speaker[speaker][cluster_col_name] = y_predicted_speaker[speaker]

    if cluster_col_name in normalized_df_speaker[speaker].columns:
        labels_atuais = normalized_df_speaker[speaker][cluster_col_name]
        cluster_counts_speaker[speaker] = labels_atuais.value_counts()
        # Só calculamos se NÃO for keywords E se houver mais de 1 cluster
        unique_labels = set(labels_atuais)
        if -1 in unique_labels: unique_labels.remove(-1)

        if labelling != "keywords" and len(unique_labels) > 1:
            try:
                silhouette_speaker[speaker] = metrics.silhouette_score(
                    vectors_speaker[speaker],
                    y_predicted_speaker.get(speaker, labels_atuais)
                )
            except Exception as e:
                print(f"Aviso: Erro ao calcular Silhouette para {speaker}: {e}")
                silhouette_speaker[speaker] = 0.0
        else:
            # Se for keywords ou só tiver 1 cluster, Silhouette é 0
            silhouette_speaker[speaker] = 0.0
            
    else:
        cluster_counts_speaker[speaker] = pd.Series()
        silhouette_speaker[speaker] = 0.0

print("Métricas calculadas e df_final atualizado.")

for speaker in speakers:
    normalized_df_speaker[speaker] = normalized_df_speaker[speaker].assign(clusters_speaker=y_predicted_speaker[speaker])

df_final = pd.concat([normalized_df_speaker[speaker] for speaker in speakers])

speaker_cluster_counts = {}
speaker_cluster_utts_mean = {}

for speaker in speakers:
    #CONTAGEM DO NÚMERO DE FALAS POR CLUSTER
    speaker_cluster_counts[speaker] = normalized_df_speaker[speaker]['clusters_speaker_'+ speaker].value_counts()
    #print("Counting the number of utterances per cluster - "+ speaker +":")
    #print(speaker_cluster_counts[speaker])
    
    # Calcular a média e o desvio padrão do número de falas por cluster
    speaker_cluster_utts_mean[speaker] = normalized_df_speaker[speaker].groupby('clusters_speaker_'+ speaker).size().agg(['mean', 'std'])
    #print("\nMédia e desvio padrão do número de falas por cluster - "+ speaker +":")
    #print(speaker_cluster_utts_mean[speaker])

with open(os.path.join('./Resultados/' + diretoria, 'N_falas_cluster.txt'), 'a') as f:
    f.write("---------START--------\n")
    for speaker in speakers:
        f.write("Counting the number of utterances per cluster - "+ speaker +": \n")
        f.write(speaker_cluster_counts[speaker].to_string(index=False)) 
        f.write("\nMédia e desvio padrão do número de falas por cluster - "+ speaker +":\n")
        f.write(speaker_cluster_utts_mean[speaker].to_string(index=False)) 
        f.write("\n----------------\n")
    f.write("---------END--------\n")

# ------------- MÉDIA DAS PALAVRAS POR FALA DE CADA CLUSTER --------------
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')

# Tokenizar e contar o número de tokens em cada 'utterance' para o df_final
df_final['word_count'] = df_final['utterance'].apply(lambda x: len(word_tokenize(x)))

# Calcular a média e o desvio padrão do número de palavras por cluster
stats_word_count_speaker = {}
for speaker in speakers:
    stats_word_count_speaker[speaker] = df_final.groupby('clusters_speaker_'+ speaker)['word_count'].agg(['mean', 'std']).reset_index()
    stats_word_count_speaker[speaker].columns = ['clusters_speaker_' + speaker, 'mean_word_count', 'std_word_count']
    #print("Média e Desvio Padrão do Número de Palavras por Cluster ("+ speaker +"):")
    #print(stats_word_count_speaker[speaker])
    
    
with open(os.path.join('./Resultados/' + diretoria, 'N_palavras_cluster.txt'), 'a') as f:
    f.write("---------START--------\n")
    for speaker in speakers:
        f.write("Média e Desvio Padrão do Número de Palavras por Cluster (" + speaker + "):\n")
        f.write(stats_word_count_speaker[speaker].to_string(index=False)) 
        f.write("\n----------------\n")
    f.write("---------END--------\n")

statistics = True

if statistics and 'Median_Binary' in df_final.columns:
    df_final['normalized_sentiment'] = df_final['Median_Binary']

    sentiment_distribution_speaker = {}
    sentiment_stats_speaker = {}
    fsc_speaker = {}

    for speaker in speakers:
        cluster_col = f'clusters_speaker_{speaker}'
        sentiment_distribution_speaker[speaker] = df_final.groupby([cluster_col, 'normalized_sentiment']).size().unstack(fill_value=0)
        sentiment_stats_speaker[speaker] = df_final.groupby(cluster_col)['normalized_sentiment'].agg(['mean', 'std']).reset_index()
        sentiment_stats_speaker[speaker].columns = [cluster_col, 'mean_sentiment', 'std_sentiment']
        #print(f"Distribuição de Sentimento por Cluster {speaker}:")
        #print(sentiment_distribution_speaker[speaker])
        #print(f"\nMédia e Desvio Padrão de Sentimento por Cluster {speaker}:")
        #print(sentiment_stats_speaker[speaker])

        fsc_speaker[speaker] = sentiment_stats_speaker[speaker]['std_sentiment'].mean()
        print(f"Flow Sentiment Cohesion (FSC) para clusters de {speaker}:", fsc_speaker[speaker])

    with open(os.path.join('./Resultados/' + diretoria, 'N_falas_cluster.txt'), 'a') as f:
        f.write("---------START--------\n")
        for speaker in speakers:
            f.write(f"Distribuição de Sentimento por Cluster {speaker}:\n")
            f.write(sentiment_distribution_speaker[speaker].to_string(index=False)) 
            f.write(f"\nMédia e Desvio Padrão de Sentimento por Cluster {speaker}:\n")
            f.write(sentiment_stats_speaker[speaker].to_string(index=False)) 
            f.write(f"\nFlow Sentiment Cohesion (FSC) para clusters de {speaker}:\n") 
            f.write(f"{fsc_speaker[speaker]:.3f}\n----------------\n")
        f.write("---------END--------\n")

else:
    print("Coluna 'Median_Binary' não encontrada. Estatísticas de sentimento não serão calculadas.")

#------------------ TEMPOS DE FALA --------------------
if 'time_since_start' in df_final.columns and 'time_since_previous' in df_final.columns:
    speaker_avg_time_since_start_per_fala = {}
    speaker_avg_time_since_previous_per_fala = {}
    speaker_std_time_since_start_per_fala = {}
    speaker_std_time_since_previous_per_fala = {}
    speaker_stats = {}

    for speaker in speakers:
        cluster_col = f'clusters_speaker_{speaker}'

        # Médias e desvios
        speaker_avg_time_since_start_per_fala[speaker] = df_final.groupby(cluster_col)['time_since_start'].mean()
        speaker_avg_time_since_previous_per_fala[speaker] = df_final.groupby(cluster_col)['time_since_previous'].mean()
        speaker_std_time_since_start_per_fala[speaker] = df_final.groupby(cluster_col)['time_since_start'].std()
        speaker_std_time_since_previous_per_fala[speaker] = df_final.groupby(cluster_col)['time_since_previous'].std()

        # Medianas e quartis
        speaker_stats[speaker] = df_final.groupby(cluster_col).agg({
            'time_since_start': ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
            'time_since_previous': ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        })

        speaker_stats[speaker].columns = [
            'median_time_since_start',
            'q1_time_since_start',
            'q3_time_since_start',
            'median_time_since_previous',
            'q1_time_since_previous',
            'q3_time_since_previous'
        ]

        print(f"{speaker} - Média 'time_since_start':\n", speaker_avg_time_since_start_per_fala[speaker])
        print(f"{speaker} - Média 'time_since_previous':\n", speaker_avg_time_since_previous_per_fala[speaker])
        print(f"{speaker} - Desvio padrão 'time_since_start':\n", speaker_std_time_since_start_per_fala[speaker])
        print(f"{speaker} - Desvio padrão 'time_since_previous':\n", speaker_std_time_since_previous_per_fala[speaker])

    with open(os.path.join('./Resultados/' + diretoria, 'N_falas_cluster.txt'), 'a') as f:
        f.write("---------START--------\n")
        for speaker in speakers:
            f.write(f"{speaker} - Média 'time_since_start':\n")
            f.write(speaker_avg_time_since_start_per_fala[speaker].to_string(index=False)) 
            f.write(f"\n{speaker} - Média 'time_since_previous':\n")
            f.write(speaker_avg_time_since_previous_per_fala[speaker].to_string(index=False)) 
            f.write(f"\n{speaker} - Desvio padrão 'time_since_start':\n") 
            f.write(speaker_std_time_since_start_per_fala[speaker].to_string(index=False)) 
            f.write(f"\n{speaker} - Desvio padrão 'time_since_previous':\n") 
            f.write(speaker_std_time_since_previous_per_fala[speaker].to_string(index=False)) 
            f.write("\n----------------\n")
        f.write("---------END--------\n")
else:
    print("As colunas 'time_since_start' e/ou 'time_since_previous' não estão presentes no dataset.")

# ### Labelling dos clusters
# Só corre os algoritmos de labelling se NÃO estivermos no modo 'keywords'
if labelling != "keywords" and algorithm != "bertopic":
#if labelling != "keywords":
    print(f"A executar labelling automático: {labelling}")
    n_grams = 3
    n_entries = 1
    nlp = spacy.load('pt_core_news_md')
    labels_tab_kbert = {}
    labels_tab_llm = {}
    labels_tab_verbs = {}
    labels_tab_closest = {}
    labels_tab_none = {}
    
    cluster_counts_speaker = {}
    listLen_speaker = {}

    # Calcular Contagens
    for speaker in speakers:
        if speaker in y_predicted_speaker:
            n_clusters_speaker[speaker] = len(set(y_predicted_speaker[speaker]))
            if len(y_predicted_speaker[speaker]) > 0:
                cluster_counts_speaker[speaker] = pd.Series(y_predicted_speaker[speaker]).value_counts().sort_index()
                listLen_speaker[speaker] = cluster_counts_speaker[speaker].tolist()
        else:
            n_clusters_speaker[speaker] = 0

    print(f"A calcular labels para o Excel: {labelling_for_excel}")

    for speaker in speakers:
        if n_clusters_speaker.get(speaker, 0) == 0: continue

        # --- kBERT ---
        if "kbert" in labelling_for_excel:
            print(f"[{speaker}] Gerando labels kBERT...")
            kBERT_result = describe_clusters_kBERT(n_clusters_speaker[speaker], y_predicted_speaker[speaker], normalized_df_speaker[speaker], model_ml, stopwords, n_grams, n_entries)
            labels_tab_kbert[speaker] = kBERT_result.rename(columns={'labels': 'label_kbert'})

        # --- LLM ---
        if "llm" in labelling_for_excel:
            print(f"[{speaker}] Gerando labels LLM...")
            LLM_result = describe_clusters_LLM(y_predicted_speaker[speaker],normalized_df_speaker[speaker],llm_url)
            labels_tab_llm[speaker] = LLM_result.rename(columns={'labels': 'label_llm'})

        # --- Verbs ---
        if "verbs" in labelling_for_excel:
            print(f"[{speaker}] Gerando labels Verbs...")
            verbs_result = describe_clusters_verbs(nlp, n_clusters_speaker[speaker], normalized_df_speaker[speaker], y_predicted_speaker[speaker])
            labels_tab_verbs[speaker] = verbs_result.rename(columns={'labels': 'label_verbs'})

        # --- Closest ---
        if "closest" in labelling_for_excel:
            print(f"[{speaker}] Gerando labels Closest...")
            centroide_result = describe_clusters_closest(normalized_df_speaker[speaker], y_predicted_speaker[speaker], vectors_speaker[speaker], centers_speaker[speaker], n_clusters_speaker[speaker])
            labels_tab_closest[speaker] = centroide_result.rename(columns={'labels': 'label_closest'})

        # --- None ---
        if "none" in labelling_for_excel and 'names_clusters' in locals():
            print(f"[{speaker}] Gerando labels 'None'...")
            aux_names_clusters = {}
            cont = 0
            cluster_keys = []
            for cluster in set(y_predicted_speaker[speaker]):
                aux_names_clusters[cont] = (names_clusters[cluster], cluster)
                cont += 1

            ordered_keys = sorted(list(set(y_predicted_speaker[speaker])))
            labels_aux = [aux_names_clusters[key][0] for key in ordered_keys]
            cluster_keys = [aux_names_clusters[key][1] for key in ordered_keys]
            display_clusters = [f'Cluster {i}' for i in range(len(cluster_keys))]

            labels_tab_none[speaker] = pd.DataFrame({
                'clusters': display_clusters,
                'label_none': labels_aux,
                'cluster_real': cluster_keys
            })

    print(f"A definir '{labelling}' como a label principal do grafo...")

    # Tenta preencher 'labels_speaker' a partir dos resultados já calculados (do Excel)
    if labelling == "kbert" and labels_tab_kbert:
        for speaker in speakers:
            if speaker in labels_tab_kbert:
                labels_speaker[speaker] = labels_tab_kbert[speaker]['label_kbert'].to_dict()
    elif labelling == "llm" and labels_tab_llm:
        for speaker in speakers:
            if speaker in labels_tab_llm:
                labels_speaker[speaker] = labels_tab_llm[speaker]['label_llm'].to_dict()
    elif labelling == "verbs" and labels_tab_verbs:
        for speaker in speakers:
            if speaker in labels_tab_verbs:
                labels_speaker[speaker] = labels_tab_verbs[speaker]['label_verbs'].to_dict()
    elif labelling == "closest" and labels_tab_closest:
        for speaker in speakers:
            if speaker in labels_tab_closest:
                labels_speaker[speaker] = labels_tab_closest[speaker]['label_closest'].to_dict()
    elif labelling == "none" and labels_tab_none:
        for speaker in speakers:
            if speaker in labels_tab_none:
                labels_speaker[speaker] = {row['cluster_real']: row['label_none'] for index, row in labels_tab_none[speaker].iterrows()}

    # Se o método escolhido no 'labelling' não estava na lista 'labelling_for_excel', corre agora
    if labelling == "kbert" and not labels_tab_kbert:
        for speaker in speakers:
            if n_clusters_speaker.get(speaker, 0) > 0:
                kBERT_result = describe_clusters_kBERT(n_clusters_speaker[speaker], y_predicted_speaker[speaker], normalized_df_speaker[speaker], model_ml, stopwords, n_grams, n_entries)
                labels_speaker[speaker] = kBERT_result['labels'].to_dict()

    elif labelling == "llm" and not labels_tab_llm:
        for speaker in speakers:
            if n_clusters_speaker.get(speaker, 0) > 0:
                LLM_result = describe_clusters_LLM(y_predicted_speaker[speaker],normalized_df_speaker[speaker],llm_url)
                labels_speaker[speaker] = LLM_result['labels'].to_dict()

    elif labelling == "verbs" and not labels_tab_verbs:
        for speaker in speakers:
            if n_clusters_speaker.get(speaker, 0) > 0:
                verbs_result = describe_clusters_verbs(nlp, n_clusters_speaker[speaker], normalized_df_speaker[speaker], y_predicted_speaker[speaker])
                labels_speaker[speaker] = verbs_result['labels'].to_dict()

    elif labelling == "closest" and not labels_tab_closest:
        for speaker in speakers:
            if n_clusters_speaker.get(speaker, 0) > 0:
                centroide_result = describe_clusters_closest(normalized_df_speaker[speaker], y_predicted_speaker[speaker], vectors_speaker[speaker], centers_speaker[speaker], n_clusters_speaker[speaker])
                labels_speaker[speaker] = centroide_result['labels'].to_dict()

    # --- TABELA DE COMPARAÇÃO (EXCEL) ---
    print("A gerar tabela de comparação de labels...")
    all_labels_merged = []
    
    for speaker in speakers:
        if speaker not in n_clusters_speaker or n_clusters_speaker[speaker] == 0:
            continue

        dfs_to_merge = []
        base_clusters = [f"Cluster {i}" for i in range(n_clusters_speaker[speaker])]
        df_base = pd.DataFrame(base_clusters, columns=["clusters"])
        dfs_to_merge.append(df_base)

        if "kbert" in labelling_for_excel and speaker in labels_tab_kbert:
            dfs_to_merge.append(labels_tab_kbert[speaker])
        if "llm" in labelling_for_excel and speaker in labels_tab_llm:
            dfs_to_merge.append(labels_tab_llm[speaker])
        if "verbs" in labelling_for_excel and speaker in labels_tab_verbs:
            dfs_to_merge.append(labels_tab_verbs[speaker])
        if "closest" in labelling_for_excel and speaker in labels_tab_closest:
            dfs_to_merge.append(labels_tab_closest[speaker])
        if "none" in labelling_for_excel and speaker in labels_tab_none:
            dfs_to_merge.append(labels_tab_none[speaker].drop(columns=['cluster_real'], errors='ignore'))
        
        if len(dfs_to_merge) <= 1: continue
            
        df_merged = dfs_to_merge[0]
        for df_to_merge in dfs_to_merge[1:]:
            df_merged = pd.merge(df_merged, df_to_merge, on="clusters", how="outer")

        # Adiciona exemplos
        def get_utterance_examples(cluster):
            if isinstance(cluster, str) and ' ' in cluster:
                try:
                    cluster_number = int(cluster.split()[1])
                    examples = clusters_speaker.get(speaker, {}).get(cluster_number, [])
                    if examples: return " | ".join(examples[:3])
                except: return ""
            return ""
            
        df_merged["ex_utterance"] = df_merged["clusters"].apply(get_utterance_examples).fillna("")

        speaker_row_data = {col: "" for col in df_merged.columns}
        speaker_row_data["clusters"] = speaker
        speaker_row = pd.DataFrame([speaker_row_data])
        
        df_merged = pd.concat([speaker_row, df_merged], ignore_index=True)
        all_labels_merged.append(df_merged)

    if all_labels_merged: 
        combined_labels = pd.concat(all_labels_merged, ignore_index=True)
        output_path = f'./Resultados/{diretoria}/labels_clusters.xlsx'
        combined_labels.to_excel(output_path, index=False)
        print(f"Tabela de comparação salva em: {output_path}")

else:
    print(">>> Modo KEYWORDS detetado: O labelling automático (KeyBERT/LLM) foi ignorado.")
    print("    As labels usadas serão as categorias do Excel ('Cooperação', 'Conflito').")

# CLUSTERING PARA AS AÇÕES
print("A processar ações (Speaker + Ação)...")

# 1. Carregar RAW data para garantir que temos os Speakers originais
df_acoes = dados[dados["Tipo"].astype(str).str.lower() == "ação"].copy()

if df_acoes.shape[0] > 0:

    #Função para garantir um Speaker válido
    def resolver_speaker(row):
        # Tenta coluna Speaker normal
        s1 = str(row.get('Speaker', ''))
        # Tenta coluna Speaker_person
        s2 = str(row.get('Speaker_person', ''))
        
        # Se 'Speaker' for válido, usa-o
        if s1.lower() not in ['nan', 'none', '', 'float', 'int']:
            return s1.strip()
        # Se não, tenta 'Speaker_person'
        elif s2.lower() not in ['nan', 'none', '', 'float', 'int']:
            return s2.strip()
        # Se ambos falharem (ex: timestamps), é Geral
        else:
            return 'Geral'

    #Aplicar a resolução do Speaker
    df_acoes['Speaker_Final'] = df_acoes.apply(resolver_speaker, axis=1)
    print(f"Exemplos de Speakers encontrados nas Ações: {df_acoes['Speaker_Final'].unique()[:10]}")

    #Garantir que acoes_simples tem texto
    df_acoes['acoes_simples'] = df_acoes['acoes_simples'].fillna('Outros').astype(str)

    # Isto é o que garante a separação: "PSD - Aplausos" vs "PS - Aplausos"
    df_acoes['acao_composta'] = df_acoes.apply(
        lambda row: f"{row['Speaker_Final']} - {row['acoes_simples']}" 
                    if row['Speaker_Final'] != 'Geral' 
                    else row['acoes_simples'], # Timestamps ficam só com a ação
        axis=1
    )

    #Clustering
    labels_acoes, labels_nomes = pd.factorize(df_acoes['acao_composta'])
    df_acoes['clusters_acoes'] = labels_acoes
    
    #Atualizar o Speaker oficial para o resto do pipeline
    df_acoes['Speaker'] = df_acoes['Speaker_Final']

    n_clusters_acoes = len(labels_nomes)
    
    # Dicionário de labels para o Excel/Grafo
    labels_acoes_dict = pd.DataFrame({
        'clusters': [f"Cluster {i}" for i in range(n_clusters_acoes)],
        'labels': labels_nomes
    })
    
    print(f"[Acoes] Clusters Criados (Exemplos): {list(labels_nomes)[:10]}")
    print(f"[Acoes] Total de clusters de ação: {n_clusters_acoes}")

else:
    print("Não foram encontradas ações.")
    n_clusters_acoes = 0
    labels_acoes_dict = {'labels': pd.Series(dtype=str)}

# Juntar ao dataframe final e reordenar
df_final = pd.concat([df_final, df_acoes], ignore_index=True)
df_final.sort_values(by=['dialogue_id', 'turn_id'], inplace=True)

count = 0
if 'listLen_speaker' not in locals() and 'listLen_speaker' not in globals():
    listLen_speaker = {}

if 'cluster_counts_speaker' not in locals() and 'cluster_counts_speaker' not in globals():
    cluster_counts_speaker = {}

print("A recalcular contagens de falas por cluster...")

listLen_speaker = {}
cluster_counts_speaker = {}

for speaker in speakers:
    # Nome da coluna onde estão os clusters (ex: 'clusters_speaker_PSD')
    col_name = f'clusters_speaker_{speaker}'
    
    # Se a coluna existir no dataframe desse speaker
    if col_name in normalized_df_speaker[speaker].columns:
        # Conta quantas vezes cada cluster aparece
        counts = normalized_df_speaker[speaker][col_name].value_counts().sort_index()
        
        # Guarda nas variáveis globais que o resto do script usa
        cluster_counts_speaker[speaker] = counts
        listLen_speaker[speaker] = counts.tolist()
    else:
        # Fallback se não encontrar
        listLen_speaker[speaker] = []

print("Contagens recuperadas com sucesso!")

# Definimos o nome para as Ações
ACAO_SPEAKER_NAME = 'Acao'

#SÓ ADICIONA O SPEAKER 'Acao' SE ELE TIVER CLUSTERS
if n_clusters_acoes > 0:
    # Adicionar 'Acao' à lista de 'speakers'
    if ACAO_SPEAKER_NAME not in speakers:
        speakers.append(ACAO_SPEAKER_NAME) 
        print(f"'{ACAO_SPEAKER_NAME}' foi adicionado à lista de speakers...")

    # Atribuir o número de clusters
    n_clusters_speaker[ACAO_SPEAKER_NAME] = n_clusters_acoes 

    # Atribuir labels
    labels_speaker[ACAO_SPEAKER_NAME] = labels_acoes_dict['labels'].to_dict()

    # Atribuir a contagem
    if df_acoes.shape[0] > 0 and 'clusters_acoes' in df_acoes.columns:
        cluster_counts_acoes = df_acoes['clusters_acoes'].value_counts().sort_index()
        cluster_counts_speaker[ACAO_SPEAKER_NAME] = cluster_counts_acoes
        listLen_speaker[ACAO_SPEAKER_NAME] = cluster_counts_acoes.tolist()
    else:
        cluster_counts_speaker[ACAO_SPEAKER_NAME] = pd.Series(dtype=int)
        listLen_speaker[ACAO_SPEAKER_NAME] = []
else:
    print("Não foram encontradas ações. O 'speaker' Acao não será adicionado.")

SOD_NAME = "SOD" 
EOD_NAME = "EOD"
SOD_LABEL = "SOD" 
EOD_LABEL = "EOD" 

# Isolar todas as ações
df_todas_acoes = df_final[df_final['Tipo'].astype(str).str.lower() == 'ação'].copy()

# Regex para encontrar (ex: "Eram 15 horas")
regex_padrao_tempo = r"^Eram \d+ horas"
# Regex para extrair (ex: "15 horas e 3 minutos")
regex_extracao_tempo = r"(\d+\s*horas?(?: e \d+\s*minutos?)?)"

if not df_todas_acoes.empty:
    # --- Encontrar o NOME DO SOD ---
    primeiras_acoes = df_todas_acoes.groupby('dialogue_id').first()
    primeiras_acoes_tempo = primeiras_acoes[
        primeiras_acoes['Ação'].astype(str).str.contains(regex_padrao_tempo, na=False)
    ]

    if not primeiras_acoes_tempo.empty:
        sod_time_string_full = primeiras_acoes_tempo['Ação'].mode().get(0)
        if sod_time_string_full:
            SOD_NAME = f"SOD ({sod_time_string_full})" 

            match = re.search(regex_extracao_tempo, sod_time_string_full)
            if match:
                time_curto = match.group(1) # Ex: "15 horas e 3 minutos"
                parts = re.findall(r'(\d+)', time_curto) # Ex: ['15', '3']

                if len(parts) == 2:
                    # Formata como "15h03" (com 'zfill' para adicionar o zero)
                    SOD_LABEL = f"SOD\n{parts[0]}h{parts[1].zfill(2)}" # \n é a quebra de linha
                elif len(parts) == 1:
                    SOD_LABEL = f"SOD\n{parts[0]}h00" # Ex: "15h00"
                else:
                    SOD_LABEL = f"SOD\n{time_curto}" # Fallback
            else:
                SOD_LABEL = SOD_NAME # Fallback se a extração falhar

    # --- Encontrar o NOME DO EOD ---
    ultimas_acoes = df_todas_acoes.groupby('dialogue_id').last()
    ultimas_acoes_tempo = ultimas_acoes[
        ultimas_acoes['Ação'].astype(str).str.contains(regex_padrao_tempo, na=False)
    ]
    
    if not ultimas_acoes_tempo.empty:
        eod_time_string_full = ultimas_acoes_tempo['Ação'].mode().get(0)
        if eod_time_string_full:
            EOD_NAME = f"EOD ({eod_time_string_full})" # O ID
            
            match = re.search(regex_extracao_tempo, eod_time_string_full)
            if match:
                time_curto = match.group(1)
                parts = re.findall(r'(\d+)', time_curto) # Ex: ['19', '24']
                if len(parts) == 2:
                     EOD_LABEL = f"EOD\n{parts[0]}h{parts[1].zfill(2)}" # Ex: "EOD\n19h24"
                elif len(parts) == 1:
                     EOD_LABEL = f"EOD\n{parts[0]}h00"
                else:
                     EOD_LABEL = f"EOD\n{time_curto}"
            else:
                EOD_LABEL = EOD_NAME

print(f"Nó de início (SOD) ID: {SOD_NAME} | Rótulo: {SOD_LABEL.replace(chr(10), ' // ')}")
print(f"Nó de fim (EOD) ID: {EOD_NAME} | Rótulo: {EOD_LABEL.replace(chr(10), ' // ')}")

# CONSTRUIR GRAFO

# Garantir que os labels estejam na mesma ordem
labels_type_speaker_ordered = {}
for speaker in speakers:
    if speaker in labels_speaker:
        labels_type_speaker_ordered[speaker] = [labels_speaker[speaker][i] for i in sorted(labels_speaker[speaker].keys())]

# Criação das labels finais
listValuesClusters = []

if count_utterances_label:
    for speaker in speakers:
        if speaker not in labels_type_speaker_ordered: continue
            
        for i, speaker_label in enumerate(labels_type_speaker_ordered[speaker]):
            count = 0
            if speaker in listLen_speaker and i < len(listLen_speaker[speaker]):
                count = listLen_speaker[speaker][i]
            
            if speaker == ACAO_SPEAKER_NAME:
                label_with_count = f"{speaker_label} ({count})"
            else:
                # Se for keywords, 'speaker_label' já é "Cooperação". 
                # O grafo fica "PS -> Cooperação (10)"
                prefix = speaker + " -> "
                label_with_count = f"{prefix}{speaker_label} ({count})"
            listValuesClusters.append(label_with_count)
else:
    # Lógica sem contagens
    temp_list_values = []
    for speaker in speakers:
        if speaker not in labels_speaker: continue
        speaker_labels = list(labels_speaker[speaker].values())

        if speaker == ACAO_SPEAKER_NAME:
            temp_list_values.append(speaker_labels)
        else:
            prefix = speaker + " -> "
            temp_list_values.append([prefix + l for l in speaker_labels])
        
    listValuesClusters = sum(temp_list_values, [])

print("Labels finais geradas.")


total_real_clusters = 0
for speaker in speakers: # 'speakers' agora inclui 'Acao'
    if speaker in n_clusters_speaker:
        total_real_clusters += n_clusters_speaker[speaker]

NA_CLUSTER_ID = total_real_clusters

NOME_NODO_NA = "Outros (Não Agrupados)"
names = listValuesClusters
names.append(NOME_NODO_NA) 
#names.extend(['SOD', 'EOD'])
names.extend([SOD_NAME, EOD_NAME])


with open(f'./Resultados/{diretoria}/DEBUG_names_list.txt', 'w', encoding='utf-8') as f:
    for idx, name in enumerate(names):
        f.write(f"{idx}: {name.replace(chr(10), ' ')}\n") # Salva o ID e o nome

ACAO_SPEAKER_NAME = 'Acao'
n_clusters_speaker_aux = 0 
df_final['n_clusters_final'] = -1

for speaker in speakers:
    
    # 1. Trata Ações
    if speaker == ACAO_SPEAKER_NAME:
        filtro = df_final['Tipo'].astype(str).str.lower() == "ação"
        if sum(filtro) > 0 and 'clusters_acoes' in df_final.columns:
             col_acoes = df_final.loc[filtro, 'clusters_acoes'].fillna(NA_CLUSTER_ID).astype(int)
             df_final.loc[filtro, 'n_clusters_final'] = col_acoes + n_clusters_speaker_aux
             df_final.loc[filtro, 'clusters_speaker_' + speaker] = col_acoes
        
    # Trata Speakers
    elif speaker in n_clusters_speaker:
        coluna_cluster_speaker = 'clusters_speaker_' + speaker
        filtro = df_final['Speaker'] == speaker
        
        if sum(filtro) > 0 and coluna_cluster_speaker in df_final.columns:
            # Pega na coluna de clusters
            col_speaker = df_final.loc[filtro, coluna_cluster_speaker]
            col_speaker_filled = col_speaker.fillna(NA_CLUSTER_ID)
            # Converte para int
            col_speaker_int = col_speaker_filled.astype(int)
            # Atribui o valor final
            df_final.loc[filtro, 'n_clusters_final'] = col_speaker_int + n_clusters_speaker_aux
            
    # Incrementa para o próximo 'speaker'
    n_clusters_speaker_aux += n_clusters_speaker[speaker]

# nClustersBoth agora inclui o cluster NA
nClustersBoth = n_clusters_speaker_aux + 1 # +1 para o cluster NA

# Mapear os -1 restantes (se houver) para o ID NA
df_final['n_clusters_final'] = df_final['n_clusters_final'].replace(-1, NA_CLUSTER_ID)
df_final['n_clusters_final'] = df_final['n_clusters_final'].astype(int)
df_final.sort_values(by=['dialogue_id', 'turn_id'], inplace=True)

print("df_final com 'n_clusters_final' (NaNs mapeados):")
print(df_final[['dialogue_id', 'turn_id', 'Tipo', 'Speaker', 'Ação', 'n_clusters_final']])

df_final.to_excel(f'./Resultados/{diretoria}/DEBUG_df_final_completo.xlsx', index=False)

NODO_NA_ID = NA_CLUSTER_ID
SOD_NODE_ID = nClustersBoth
EOD_NODE_ID = nClustersBoth + 1
numberLabelsMatrix = (nClustersBoth + 2) # Total de clusters + SOD + EOD

occurrence_matrix = np.zeros((numberLabelsMatrix, numberLabelsMatrix))
sentiment_transitions = {}
sentiment_cluster = 'Median_Binary'

if int(normalized_df['dialogue_id'].iat[-1])+1 > 0:
    for i in range(int(normalized_df['dialogue_id'].iat[-1])+1):
        
        clusters_neste_dialogo = df_final.loc[
            df_final.dialogue_id == i, 'n_clusters_final'
        ].tolist()
        
        if not clusters_neste_dialogo:
            continue

        # Transição SOD -> Primeiro Turno
        primeiro_cluster = clusters_neste_dialogo[0]
        occurrence_matrix[SOD_NODE_ID, primeiro_cluster] += 1
        #    Isto vai apanhar as transições (Fala -> Ação)
        for d_a, d in zip(clusters_neste_dialogo[:-1], clusters_neste_dialogo[1:]):           
            occurrence_matrix[d_a, d] += 1

            if sentiment_in_flow == "transition":
                pass
            
        # Transição Último Turno -> EOD
        ultimo_cluster = clusters_neste_dialogo[-1]
        occurrence_matrix[ultimo_cluster, EOD_NODE_ID] += 1

# --- Cálculo de Sentimento ---
if sentiment_cluster in df_final.columns:
    df_final[sentiment_cluster] = ((df_final[sentiment_cluster] + 1) / 2).fillna(0.5)
    df_final['avg_sentiment'] = 0.5
    for n in range(max(df_final['n_clusters_final']) + 1):
        cluster_df = df_final[df_final['n_clusters_final'] == n]
        if cluster_df.empty: continue
        avg_sent = cluster_df[sentiment_cluster].mean()
        df_final.loc[df_final['n_clusters_final'] == n, "avg_sentiment"] = avg_sent
        # std_sent_max/min
        std_max = (avg_sent + cluster_df[sentiment_cluster].std())
        std_min = (avg_sent - cluster_df[sentiment_cluster].std())
        df_final.loc[df_final['n_clusters_final'] == n, "std_sent_max"] = colorize_sentiment_v2(std_max)
        df_final.loc[df_final['n_clusters_final'] == n, "std_sent_min"] = colorize_sentiment_v2(std_min)


# --- Matriz de Transição ---
row_sums = occurrence_matrix.sum(axis=1)
# Evitar divisão por zero se uma linha for toda zero
row_sums[row_sums == 0] = 1
transition_matrix = np.divide(occurrence_matrix, row_sums[:, np.newaxis])
matrix = pd.DataFrame(transition_matrix, index=names, columns=names)
matrix = matrix.round(decimals = 2).fillna(0.00)

# GERAÇÃO DOS HEATMAPS
try:
    gerar_heatmap(
        matriz_df=matrix, 
        diretorio_saida=diretoria, 
        nome_ficheiro=f"Heatmap_Transitions_{filename.split('.')[0]}.png", 
        titulo=f"Transition Probability Matrix"
    )
except Exception as e:
    print(f"Erro ao gerar Heatmaps: {e}")
    
    # Heatmap das Ocorrências Absolutas (Quantas vezes aconteceu a transição)
    matrix_ocorrencias = pd.DataFrame(occurrence_matrix, index=names, columns=names)
    gerar_heatmap(
        matriz_df=matrix_ocorrencias, 
        diretorio_saida=diretoria, 
        nome_ficheiro=f"Heatmap_Ocorrencias_{filename.split('.')[0]}.png", 
        titulo=f"Matriz de Ocorrências Absolutas ({filename})"
    )
except Exception as e:
    print(f"Erro ao gerar Heatmaps: {e}")

# Verifica se existe coluna de sentimento em algum dos dataframes
colunas_totais = pd.concat([df_final, df_acoes]).columns

if "Median_Binary" in colunas_totais:
    print("A calcular sentimentos e cores...")
    
    # 1. Concatenar
    combined_df = pd.concat([df_final, df_acoes], ignore_index=True)
    
    # 2. Normalizar Sentimento
    combined_df['Median_Binary'] = ((combined_df['Median_Binary'] + 1) / 2).fillna(0.5)
    combined_df['avg_sentiment'] = 0.5
    
    # [O TRUQUE DA GENERICIDADE] Inicializar as colunas de cores explicitamente como TEXTO ('object')
    combined_df["std_sent_max"] = pd.Series(dtype='object')
    combined_df["std_sent_min"] = pd.Series(dtype='object')

    # Preenche NaNs com -1 e força a coluna inteira a ser inteiro
    combined_df['n_clusters_final'] = combined_df['n_clusters_final'].fillna(-1).astype(int)
    max_cluster_val = int(combined_df['n_clusters_final'].max())

    for n in range(max_cluster_val + 1):
        cluster_df = combined_df[combined_df['n_clusters_final'] == n]
        
        if cluster_df.empty:
            continue
            
        avg_sent = cluster_df[sentiment_cluster].mean()
        std_sent = cluster_df[sentiment_cluster].std() if len(cluster_df) > 1 else 0.0
        
        # Atribuir cores (Agora o Pandas já aceita as strings hexadecimais '#RRGGBB')
        combined_df.loc[cluster_df.index, "std_sent_max"] = colorize_sentiment_v2(avg_sent + std_sent)
        combined_df.loc[cluster_df.index, "std_sent_min"] = colorize_sentiment_v2(avg_sent - std_sent)
        combined_df.loc[cluster_df.index, "avg_sentiment"] = avg_sent

    # [GARANTIA] Devolver as colunas calculadas aos dataframes originais para o resto do código não falhar
    df_final = combined_df[combined_df['Tipo'].astype(str).str.lower() != 'ação'].copy()
    df_acoes = combined_df[combined_df['Tipo'].astype(str).str.lower() == 'ação'].copy()

    print("Cálculo de cores concluído.")

else:
    print("Coluna 'Median_Binary' não encontrada. A saltar o cálculo do sentimento e cores.")


def traverse(dataframe, threshold, InputGraph, df_final, SOD_NAME, EOD_NAME): 
    listi = [SOD_NAME]
    non_visited = []
    visited = [EOD_NAME] 
    n_nodes = 0

    for c in dataframe.columns:
        non_visited.append(c)
    
    # Loop principal
    while len(listi) != 0:
        # Criamos uma cópia da lista para iterar com segurança enquanto removemos itens
        current_nodes = list(listi) 
        
        for i in current_nodes:
            k = 0
            flag = 0
            
            # PROTEÇÃO 1: Se o nó não estiver no índice da matriz, salta fora
            if i not in dataframe.index:
                if i in listi: listi.remove(i)
                continue

            # Itera sobre as colunas (destinos) para a linha i (origem)
            for j in dataframe.loc[i]:
                target_node = dataframe.columns[k]
                
                try:
                    val = float(j)
                except (ValueError, TypeError):
                    val = 0.0 # Se for texto ou lixo, assume probabilidade 0
                # -------------------------------------------------------

                # Usa 'val' (número limpo) nas comparações
                if (val > threshold) or (i == SOD_NAME and val > 0) or (target_node == EOD_NAME and val > 0):
                    
                    sentiment = None
                    std_max = None
                    std_min = None
                    
                    # Lógica de Sentimento
                    if i != EOD_NAME: 
                        if 'Median_Binary' in df_final.columns: # Usar nome da coluna direto ou variável global
                            if target_node == EOD_NAME: 
                                sentiment = None
                                # Tenta encontrar o último cluster
                                try:
                                    # Lógica simplificada de busca para evitar erros de índice
                                    last_rows = df_final[df_final['n_clusters_final'] == names.index(i)]
                                    if not last_rows.empty:
                                        sentiment = last_rows.iloc[0]['avg_sentiment']
                                        std_max = last_rows.iloc[0]['std_sent_max']
                                        std_min = last_rows.iloc[0]['std_sent_min']
                                except:
                                    pass
                            else:
                                try:
                                    # Lógica normal
                                    target_idx = names.index(target_node)
                                    target_rows = df_final[df_final['n_clusters_final'] == target_idx]
                                    if not target_rows.empty:
                                        sentiment = target_rows.iloc[0]['avg_sentiment']
                                        std_max = target_rows.iloc[0]['std_sent_max']
                                        std_min = target_rows.iloc[0]['std_sent_min']
                                except:
                                    pass

                    # Definir cor
                    color = "black"
                    if sentiment is not None:
                        color = colorize_sentiment_v2(sentiment) # Assume que esta função existe

                    InputGraph.add_edge(
                        i, target_node, 
                        weight=val, 
                        label=val, 
                        sentiment=sentiment, 
                        color=color, 
                        std_max=std_max, 
                        std_min=std_min
                    )

                    if target_node != EOD_NAME and target_node in non_visited and target_node not in listi: 
                        flag = 1
                        listi.append(target_node)
                
                k += 1
            
            # Limpeza das listas de controlo
            if i in listi: listi.remove(i)
            if i in non_visited: non_visited.remove(i)
            
            if flag == 1:
                visited.append(i)
            
            # Se a lista esvaziar, terminamos e retornamos
            if len(listi) == 0:
                return visited, n_nodes
                
    return visited, n_nodes


def metrics_approach(G, labels_speaker):
  probs = [e for e in G.edges.data('label')]
  sentiments = [e for e in G.edges.data('sentiment')]
  print("probs", probs)
  print("sentiments", sentiments)
  sentiment_EOD = []
  sentiment_SOD = []
  probabilidade_EOD = []
  probabilidade_SOD = []
  n_utterances_SOD = []
  n_utterances_EOD = []
  visited_SOD = []
  visited_EOD = []
  n_clusters_EOD = []
  n_clusters_SOD = []
  true_probabilidade_EOD = []
  true_n_utterances_EOD = []
  true_sentiment_EOD = []
  true_n_clusters_EOD = []
  true_visited_EOD = []
  n_utt_eod_anterior = []
  visited_anterior_EOD = []
  n_utt_sod_anterior = []
  n_utt_true_eod_anterior = []
  n_utterances_anterior_EOD = []
  n_clusters_anterior_EOD = []
  n_clusters_anterior_SOD = []
  n_utterances_anterior_SOD = []

  for i, o_value in enumerate(probs):
    sent = sentiments[i][2]
    start = o_value[0]
    end = o_value[1]
    p = o_value[2]

    if start == 'SOD':
      if p == 1.0:
        found = 0

        #1
        offset = 0
        for speaker, labels in labels_speaker.items():
            for c, l in labels.items():
                if l in end: 
                  key_primeiro = c + offset
                  found = 1
                  offset += len(labels_speaker[speaker])
                  break

        for i, value in enumerate(probs):
          sent_after = sentiments[i][2]
          start_after = value[0]
          end_after = value[1]
          p_after = value[2]
          if end == start_after:
            found = 0

            if sent_after is None or pd.isna(sent_after):
                sent_after = 0.5
            sentiment_SOD.append(sent_after)
            probabilidade_SOD.append(p_after)
            # Vamos buscar num cluster associada aquela label


            #2
            offset = 0
            for speaker, labels in labels_speaker.items():
                for c, l in labels.items():
                    if l in end_after: 
                      key = c + offset
                      found = 1
                      offset += len(labels_speaker[speaker])
                      break
                    
            n_utterances_SOD.append(df_final[df_final['n_clusters_final'] == key].shape[0])
            print("n_utterances_SOD", n_utterances_SOD)
            n_utterances_anterior_SOD.append(df_final[df_final['n_clusters_final'] == key_primeiro].shape[0])
            n_clusters_anterior_SOD.append(key_primeiro)

            n_clusters_SOD.append(key)

      else:
        n_utterances_anterior_SOD.append(df_final[df_final['turn_id'] == 0].shape[0])
        print("n_utterances_anterior_SOD", n_utterances_anterior_SOD)
        n_clusters_anterior_SOD.append(-1)

        found = 0
        sentiment_SOD.append(sent)
        probabilidade_SOD.append(p)
        # Vamos buscar num cluster associada aquela label


        #3
        offset=0
        for speaker, labels in labels_speaker.items():
                for c, l in labels.items():
                    if l in end: 
                      key = c + offset
                      found = 1
                      offset += len(labels_speaker[speaker])
                      break

        if key in visited_SOD:
          n_utterances_SOD.append(0)
          n_clusters_SOD.append(key)
          continue

        n_utterances_SOD.append(df_final[df_final['n_clusters_final'] == key].shape[0])
        print("n_utterances_SOD", n_utterances_SOD)
        n_clusters_SOD.append(key)

        visited_SOD.append(key)

    if end == 'EOD':
      for i, value in enumerate(probs):
        sent_after = sentiments[i][2]
        start_after = value[0]
        end_after = value[1]
        p_after = value[2]
        if end_after == start:
          for x, aa_value in enumerate(probs):
            sent_after_after = sentiments[x][2]
            start_after_after = aa_value[0]
            end_after_after = aa_value[1]
            p_after_after = aa_value[2]

            if start_after == end_after_after:
              if [start_after_after, end_after_after] not in visited_EOD:
                found = 0
                if sent_after_after is None or pd.isna(sent_after_after):
                    sent_after_after = 0.5

                sentiment_EOD.append(sent_after_after)
                probabilidade_EOD.append(p_after_after)


                #4
                offset=0
                for speaker, labels in labels_speaker.items():
                        for c, l in labels.items():
                            if l in end_after_after: 
                              key = c + offset
                              found = 1
                              offset += len(labels_speaker[speaker])
                              break

                n_utterances_EOD.append(df_final[df_final['n_clusters_final'] == key].shape[0])
                n_clusters_EOD.append(key)
                visited_EOD.append([start_after_after, end_after_after])

                found = 0


                #5
                offset=0
                for speaker, labels in labels_speaker.items():
                        for c, l in labels.items():
                            if l in start_after_after: 
                              key = c + offset
                              found = 1
                              offset += len(labels_speaker[speaker])
                              break

                n_utterances_anterior_EOD.append(df_final[df_final['n_clusters_final'] == key].shape[0])
                n_clusters_anterior_EOD.append(key)

      found = 0
      true_sentiment_EOD.append(sent)
      true_probabilidade_EOD.append(p)

      offset=0
      
      #6
      for speaker, labels in labels_speaker.items():
        for c, l in labels.items():
          if l in start: 
            key = c + offset
            found = 1
            offset += len(labels_speaker[speaker])
            break

      if key in true_visited_EOD:
        true_n_utterances_EOD.append(0)
        true_n_clusters_EOD.append(key)
        continue

      true_n_utterances_EOD.append(df_final[df_final['n_clusters_final'] == key].shape[0])
      print("True - n_utterances_EOD", true_n_utterances_EOD)

      true_n_clusters_EOD.append(key)
      true_visited_EOD.append(key)

  n_clusters_sod_visited = []
  for i_cluster_sod, n_cluster_sod in enumerate(n_clusters_anterior_SOD):
    if n_cluster_sod not in n_clusters_sod_visited:
      n_clusters_sod_visited.append(n_cluster_sod)
      n_utt_sod_anterior.append(n_utterances_anterior_SOD[i_cluster_sod])
    else:
      n_clusters_sod_visited.append(n_cluster_sod)
      n_utt_sod_anterior.append(0)
  print("n_clusters_anterior_SOD", n_clusters_anterior_SOD)
  print("n_utt_sod_anterior", n_utt_sod_anterior)

  n_clusters_eod_true_visited = []
  for i_cluster_eod, n_cluster_eod in enumerate(n_clusters_EOD):
    if n_cluster_eod not in n_clusters_eod_true_visited:
      n_clusters_eod_true_visited.append(n_cluster_eod)
      n_utt_true_eod_anterior.append(n_utterances_EOD[i_cluster_eod])
    else:
      n_clusters_eod_true_visited.append(n_cluster_eod)
      n_utt_true_eod_anterior.append(0)
  print("n_utt_true_eod_anterior", n_utt_true_eod_anterior)

  n_clusters_eod_visited = []
  for i_cluster_eod, n_cluster_eod in enumerate(n_clusters_anterior_EOD):
    if n_cluster_eod not in n_clusters_eod_visited:
      n_clusters_eod_visited.append(n_cluster_eod)
      n_utt_eod_anterior.append(n_utterances_anterior_EOD[i_cluster_eod])
    else:
      n_clusters_eod_visited.append(n_cluster_eod)
      n_utt_eod_anterior.append(0)
  print("n_clusters_anterior_eod", n_clusters_anterior_EOD)
  print("n_utt_eod_anterior", n_utt_eod_anterior)

  return sentiment_EOD, sentiment_SOD, probabilidade_EOD, probabilidade_SOD, n_utterances_SOD, n_utterances_EOD, n_clusters_EOD, n_clusters_SOD, true_probabilidade_EOD, true_visited_EOD, true_n_clusters_EOD,  true_sentiment_EOD, true_n_utterances_EOD, n_utt_true_eod_anterior, n_utt_sod_anterior, n_utt_eod_anterior, n_utterances_anterior_SOD, n_clusters_anterior_SOD, n_utterances_anterior_EOD

def dot_with_std(filename):
  with open( filename, "r") as f:
    lines = f.readlines()
  #print(lines)
  new_lines = []
  transitions = []
  visited = ["EOD", "SOD"]

  n = 0
  for i, line in enumerate(lines):
    if i == 0:
      new_lines.append(line)
      new_lines.append("nodesep=0.5;\nranksep=\"1.2 equally\";")
    if "key" not in line and line != "}\n" and i > 0:
      if "EOD" in line or "SOD" in line:
        new_lines.append(line)
    elif "key" in line and line != "}\n":
      
      info = line.split("[")
      
      #print("ini")
      #print(info)
      #print("fin")
      transition = info[0].strip()
      color = info[1].split("color=")[1].split(",")[0].strip("\"")
      key = info[1].split(",")[1].split("=")[1]
      label = info[1].split(",")[2].split("=")[1]
      sentiment = info[1].split(",")[3].split("=")[1]
      std_max = info[1].split("std_max=")[1].split(",")[0].strip("\"")
      std_min = info[1].split("std_min=")[1].split("]")[0].strip("\"")
      weight = info[1].split(",")[2].split("=")[1]
      transition_split = transition.split("\"")

      if "EOD" in transition_split[-1]:
        aux = transition_split[-1].strip().split("->")[1].strip()
        nome_cluster= aux
      else:
        nome_cluster = "\"" + transition_split[-2].strip() + "\""

      # se ja criou o cluster, adiciona so a transicao. caso contratio, adiciona a lista de visitados e cria.
      if nome_cluster not in visited:
        visited.append(nome_cluster)
      else:
        transition_line = "\n" + transition + "[color=\"" + color + "\",key=" + key + ",label=" + label + ",sentiment=" + sentiment + ",weight=" + weight + "];"
        transitions.append(transition_line)
        continue

      border = "black"
      #if "sys" in nome_cluster or "Speaker 2" in nome_cluster:
        #border = "blue"
      #elif "ross" in nome_cluster or "Speaker 1" in nome_cluster:
        #border = "red"
      #else:
        #border = "turquoise"

      new_line = "subgraph cluster_" + str(n) + " {\n\tstyle=striped;\n\tcolor=lightgrey;\n\tfillcolor=\"" + std_min + ";.15:" + color + ";.7:" + std_max + "\";\n\tnode [style=filled," + "color=" + border + ",fillcolor=white];\n\t" + nome_cluster + ";\n}"

      transition_line = "\n" + transition + "[color=\"" + color + "\",key=" + key + ",label=" + label + ",sentiment=" + sentiment + ",weight=" + weight + "];"
      transitions.append(transition_line)

      new_lines.append(new_line)
      n+=1

    elif line == "}\n":

      for t in transitions:
        new_lines.append(t)

      new_lines.append(line)
      n+=1
  filename_std = filename.split('.')[0] + "_std.dot"
  print("transitions:",transitions)
  with open(os.path.join('./Resultados/' + diretoria, filename_std), "w") as f:
    for line in new_lines:
      f.write(line)
  return filename_std

avg_SOD = None
avg_EOD = None
density = None

dataframe_cluster_map = {name: idx for idx, name in enumerate(names)}

#sentiment_transitions = compute_sentiment_transitions(df_final)
def generate_markov_chain_separately(filename, n_clusters_speaker, matrix: pd.DataFrame, df_final, threshold) -> nx.MultiDiGraph:
    global avg_SOD, avg_EOD, density
    print("entrei em generate_markov_chain_separately")
    
    G = nx.MultiDiGraph()
    matrix = pd.DataFrame(matrix)
    
    # 1. Gerar Arestas e Nós base
    visited, n_nodes = traverse(matrix, threshold, G, df_final, SOD_NAME, EOD_NAME)

    # --- ADICIONAR NÓS DAS AÇÕES ---
    if "Tipo" in df_final.columns:
        df_acoes = df_final[df_final["Tipo"].astype(str).str.lower() == "ação"]
    else:
        df_acoes = pd.DataFrame()

    print(f"\n[DEBUG] Número total de ações encontradas: {len(df_acoes)}")

    for idx, row in df_acoes.iterrows():
        cluster_id = row.get('clusters_acoes', None)
        acao_label = None

        if pd.notna(cluster_id):
            acao_label = f"Ação cluster {int(cluster_id)}"
        elif pd.notna(row.get('Ação')):
            acao_label = str(row['Ação']).strip()
        else:
            acao_label = f"Ação_{idx}"

        if acao_label not in G:
            G.add_node(
                acao_label,
                style="filled",       
                fillcolor="white",    
                color="#228B22",
                fontcolor="black",
                penwidth="1.0",
                shape="box",          
                tipo="acao"
            )

    # --- ADICIONAR ARESTAS DAS AÇÕES ---
    if not df_acoes.empty and "KeyBERT_label" in df_final.columns:
        pass 

    print("A aplicar cores de sentimento às transições...")
    
    for u, v, k, data in G.edges(keys=True, data=True):
        # Obter o sentimento da transição (criado no traverse)
        sent = data.get('sentiment')
        
        # Se tiver sentimento válido, aplicar cor (Vermelho-Amarelo-Verde)
        if sent is not None:
            try:
                sent_float = float(sent)
                cor_aresta = colorize_sentiment_v2(sent_float)
                data['color'] = cor_aresta
            except:
                pass # Se der erro na conversão, mantém a cor original (preto)

    print("A aplicar estilos aos nós...")

    for node in G.nodes():
        if "tipo" in G.nodes[node] and G.nodes[node]["tipo"] == "acao":
            continue

        node_str = str(node).upper()

        attrs = {
            'style': 'filled',
            'fillcolor': 'white',
            'fontname': 'Arial',
            'fontsize': '10',
            'fontcolor': 'black',
            'penwidth': '1.0'
        }

        try:
            if node in dataframe_cluster_map:
                c_idx = dataframe_cluster_map[node]
                sent_val = df_final[df_final['n_clusters_final'] == c_idx]['avg_sentiment'].mean()
                if pd.notna(sent_val):
                    attrs['sentiment'] = str(round(sent_val, 2))
                else:
                    attrs['sentiment'] = "0.00"
            else:
                attrs['sentiment'] = "0.00"
        except:
            attrs['sentiment'] = "0.00"

        if "SOD" in node_str:
            attrs['color'] = "#FFD700"
            attrs['shape'] = "circle"
        
        elif "EOD" in node_str:
            attrs['color'] = "#FFD700"
            attrs['shape'] = "circle"

        elif "SYSTEM" in node_str or "AGENTE" in node_str or "AUTOMAISE" in node_str:
            attrs['color'] = "#00008B"
            attrs['shape'] = "ellipse"

        elif "USER" in node_str or "CLIENTE" in node_str:
            attrs['color'] = "#33CCFF"
            attrs['shape'] = "ellipse"

        else:
            attrs['color'] = "black"
            attrs['shape'] = "ellipse"

        nx.set_node_attributes(G, {node: attrs})

    # ESTATÍSTICAS
    num_nodes = len(G.nodes)
    num_edges = G.number_of_edges()
    density = nx.density(G)

    print(f"Número de nós: {num_nodes}")
    print(f"Número de arestas: {num_edges}")
    print(f"Densidade do grafo: {density:.4f}")

    # Guarda um ficheiro "metricas_densidade.txt" na pasta atual
    with open("metricas_densidade.txt", "w", encoding="utf-8") as file:
        file.write("--- ESTATÍSTICAS DA REDE ---\n")
        file.write(f"Numero de nos: {num_nodes}\n")
        file.write(f"Numero de arestas: {num_edges}\n")
        file.write(f"Densidade do grafo: {density:.4f}\n\n")

    # Outras Estatísticas 
    try:
        in_degree_centrality = nx.in_degree_centrality(G)
        out_degree_centrality = nx.out_degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Anexa ao ficheiro que já criámos
        with open("metricas_densidade.txt", "a", encoding="utf-8") as file:
            file.write("--- CENTRALIDADES ---\n")
            file.write("Calculadas com sucesso. (Podes imprimir aqui se precisares)\n")
            
    except Exception as e:
        print(f"Aviso: Não foi possível calcular centralidades ({e})")

    return G, num_nodes

def generate_markov_chain_separately_no(filename, n_clusters_speaker, matrix: pd.DataFrame, df_final, threshold, ACAO_SPEAKER_NAME, NOME_NODO_NA, SOD_NAME, EOD_NAME, SOD_LABEL, EOD_LABEL) -> nx.MultiDiGraph:
    print("entrei em generate_markov_chain_separately_no")
    
    G = nx.MultiDiGraph()
    matrix = pd.DataFrame(matrix)
    
    # 1. Ordenação
    df_final = df_final.sort_values(by=['dialogue_id', 'turn_id'])
    
    # 2. Gerar Grafo
    visited, n_nodes = traverse(matrix, threshold, G, df_final, SOD_NAME, EOD_NAME)
    
    # 3. Limpeza
    to_remove = []
    for i in list(G.nodes):
        if i not in visited:
            to_remove.append(i)
    PARTY_COLORS = {
        "BE": "#B22222", "PCP": "#FF0000", "PEV": "#008000", "L": "#9ACD32",
        "PS": "#FF69B4", "PAN": "#20B2AA", "JPP": "#32CD32", "GOV": "#000000",
        "IL": "#00BFFF", "PSD": "#FFA500", "CDS-PP": "#1E90FF", "CH": "#000080",
        "Presidente": "#708090", "SOD": "#FFD700", "EOD": "#FFD700",
        "Outros": "#A9A9A9", "Geral": "#A9A9A9"
    }

    # Função auxiliar interna para resolver a cor
    def get_color_safe(speaker_name):
        if speaker_name in PARTY_COLORS:
            return PARTY_COLORS[speaker_name]
        for key, color in PARTY_COLORS.items():
            if key in speaker_name:
                return color
        # 3. Fallback (Cinzento se não encontrar)
        return "#CCCCCC"

    node_attrs = {}
    
    for node in G.nodes():
        node_str = str(node)

        atributos = {
            'fontname': 'Arial', 'fontsize': '12',
            'style': 'filled', 'fillcolor': 'white', 'fontcolor': 'black'
        }

        if node_str == SOD_NAME:
            c = get_color_safe('SOD')
            atributos.update({'color': c, 'fillcolor': c, 'shape': 'circle', 'label': str(SOD_LABEL)})
            
        elif node_str == EOD_NAME:
            c = get_color_safe('EOD')
            atributos.update({'color': c, 'fillcolor': c, 'shape': 'circle', 'label': str(EOD_LABEL)})

        elif " -> " in node_str:
            prefix = node_str.split(" -> ")[0]
            c = get_color_safe(prefix)
            atributos.update({'color': c, 'fillcolor': 'white', 'shape': 'ellipse', 'penwidth': '1.5'})

        elif " - " in node_str:
            prefix = node_str.split(" - ")[0]
            c = get_color_safe(prefix)
            atributos.update({'color': c, 'fillcolor': 'white', 'shape': 'box', 'penwidth': '1.5'})

        else:
            atributos.update({'color': '#808080', 'shape': 'box', 'penwidth': '1.5'}) 

        node_attrs[node] = atributos

    nx.set_node_attributes(G, node_attrs)    
    try:
        duracao_str = "N/A"
        df_tempo = df_final[df_final['Tipo'].astype(str).str.lower() == 'ação'].copy()
        # Procura linhas de tempo ignorando maiúsculas/minúsculas
        timestamps = df_tempo[df_tempo['Ação'].astype(str).str.contains(r"Eram.*?horas", case=False, regex=True)]
        
        if not timestamps.empty:
            import re
            def get_mins(t):
                t = str(t).lower()
                h = int(re.search(r'(\d+)\s*horas?', t).group(1)) if re.search(r'(\d+)\s*horas?', t) else 0
                m = int(re.search(r'(\d+)\s*minutos?', t).group(1)) if re.search(r'(\d+)\s*minutos?', t) else 0
                return h*60 + m

            ini = get_mins(timestamps.iloc[0]['Ação'])
            fim = get_mins(timestamps.iloc[-1]['Ação'])
            if fim < ini: fim += 1440
            d = fim - ini
            duracao_str = f"{d//60}h {d%60}m"

        # Estatísticas
        df_m = df_final.copy()
        
        def make_lbl(row):
            if row['Tipo'].lower() == 'ação': return str(row.get('acao_composta', 'Ação'))
            return f"{row['Speaker']} -> {str(row.get('speech_act_gerado', 'Indef'))}"
        
        if 'Node_Label' not in df_m.columns: df_m['Node_Label'] = df_m.apply(make_lbl, axis=1)

        U = len(df_m[df_m['Tipo'].str.lower()=='fala'])
        A = len(df_m[df_m['Tipo'].str.lower()=='ação'])
        D = df_m['dialogue_id'].nunique()
        E = G.number_of_edges()

        # Clusters (Limpeza de nomes para match)
        raw_nodes = [str(n) for n in G.nodes() if n not in [SOD_NAME, EOD_NAME]]
        clean_nodes = [n.rsplit(" (", 1)[0] if " (" in n and n.endswith(")") else n for n in raw_nodes]
        
        unique_nodes = set(clean_nodes)
        df_filt = df_m[df_m['Node_Label'].isin(unique_nodes)]
        counts = df_filt.groupby('Node_Label').size()
        
        lbl_fala = [n for n in counts.index if "->" in n]
        lbl_acao = [n for n in counts.index if "->" not in n]
        
        Cd, Ca = len(lbl_fala), len(lbl_acao)
        C = Cd + Ca
        
        uc_mean = counts[lbl_fala].mean() if Cd > 0 else 0
        uc_std = counts[lbl_fala].std(ddof=1) if Cd > 1 else 0
        ac_mean = counts[lbl_acao].mean() if Ca > 0 else 0
        ac_std = counts[lbl_acao].std(ddof=1) if Ca > 1 else 0

        # Diálogos
        all_d = df_m['dialogue_id'].unique()
        cf = df_m[df_m['Tipo'].str.lower()=='fala'].groupby('dialogue_id').size().reindex(all_d, fill_value=0)
        ca = df_m[df_m['Tipo'].str.lower()=='ação'].groupby('dialogue_id').size().reindex(all_d, fill_value=0)
        
        ud_mean, ud_std = cf.mean(), cf.std(ddof=1) if D > 1 else 0
        ad_mean, ad_std = ca.mean(), ca.std(ddof=1) if D > 1 else 0

        # --- NOVAS MÉTRICAS DE COMPORTAMENTO ---
        # 1. Calcular Taxa de Repetição (Self-Loops)
        num_self_loops = sum(1 for u, v in G.edges() if u == v)
        taxa_self_loops = (num_self_loops / E) * 100 if E > 0 else 0.0

        # 2. Encontrar o Nó Central (Ignorando SOD e EOD)
        nos_poesia = [n for n in G.nodes() if n not in [SOD_NAME, EOD_NAME]]
        no_principal = "N/A"
        if nos_poesia:
            # Encontra o nó com mais trânsito (maior grau de entrada + saída)
            no_pesado = max(nos_poesia, key=lambda n: G.degree(n))
            # Limpa o texto para ficar só a categoria (ex: tira o "José Pinhal -> ")
            no_principal = str(no_pesado).split(" -> ")[-1] if " -> " in str(no_pesado) else str(no_pesado)
        # ---------------------------------------

        # Tabela
        density = nx.density(G) # <--- Calcula a densidade aqui!

        # Criamos uma variável com a tabela inteira
        tabela = f"""{'Símbolo':<10} | {'Valor':<22} | {'Descrição'}
---------------------------------------------------------------------------
|U|        | {U:<22} | Total Falas
|A|        | {A:<22} | Total Ações
|D|        | {D:<22} | Total Diálogos
ΔT         | {duracao_str:<22} | Duração
---------------------------------------------------------------------------
|U|/|D|    | {ud_mean:.2f} ± {ud_std:.2f}        | Falas/Diálogo
|A|/|D|    | {ad_mean:.2f} ± {ad_std:.2f}        | Ações/Diálogo
---------------------------------------------------------------------------
|C|        | {C:<22} | Total Estados (Grafo)
|C_d|      | {Cd:<22} | Nós Fala
|C_a|      | {Ca:<22} | Nós Ação
---------------------------------------------------------------------------
|U|/|C|    | {uc_mean:.2f} ± {uc_std:.2f}        | Falas/Estado
|A|/|C|    | {ac_mean:.2f} ± {ac_std:.2f}        | Ações/Estado
|E|        | {E:<22} | Transições (Arestas)
Densidade  | {density:<22.4f} | Densidade do Grafo
Auto-Loops | {taxa_self_loops:>20.2f} % | Taxa de repetição no mesmo estado
Nó Central | {no_principal:<22} | Categoria dominante do texto
============================================================"""

        print(tabela, flush=True)

        # Vai usar o mesmo nome do ficheiro .dot, mas muda para _metricas.txt
        txt_filename = str(filename).replace('.dot', '') + '_metricas.txt'
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(tabela)
        
        print(f"Tabela de métricas guardada em: {txt_filename}\n")

    except Exception as e:
        print(f"ERRO MÉTRICAS: {e}")
        import traceback
        traceback.print_exc()

    return G, n_nodes

# ### Gerar fluxo e calcular métricas (sentimento)

# %%
file_result_flow = f'flow1111e_{algorithm}_{filename}_{metric_to_optimize}_{str(id_max)}.pkl'

"""if os.path.exists('./Resultados/' + diretoria + "/" +file_result_flow):
        with open(os.path.join('./Resultados/' + diretoria + "/" + file_result_flow), 'rb') as file:
            result_list = pickle.load(file)
        i = 0
        for speaker in speakers:
          if i < len(result_list):  
            labels_speaker[speaker] = result_list[i]
            i += 1  
else:"""
print ("df_final", df_final)
if sentiment_cluster in df_final.columns:
        result, n_nodes = generate_markov_chain_separately(filename, n_clusters_speaker, matrix, df_final, threshold[0])

        sentiment_EOD, sentiment_SOD, probabilidade_EOD, probabilidade_SOD, n_utterances_SOD, n_utterances_EOD, n_clusters_EOD, n_clusters_SOD, true_probabilidade_EOD, true_visited_EOD, true_n_clusters_EOD,  true_sentiment_EOD, true_n_utterances_EOD, n_utt_true_eod_anterior, n_utt_sod_anterior, n_utt_eod_anterior,  n_utterances_anterior_SOD, n_clusters_anterior_SOD, n_utterances_anterior_EOD = metrics_approach(result, labels_speaker)
        print("sentiment_EOD", sentiment_EOD)
        print("sentiment_SOD", sentiment_SOD)
        print("probabilidade_SOD", probabilidade_SOD)
        print("probabilidade_EOD", probabilidade_EOD)
        print("n_utterances_SOD", n_utterances_SOD)
        print("n_utterances_EOD", n_utterances_EOD)
        print("n_clusters_EOD", n_clusters_EOD)
        print("n_clusters_SOD", n_clusters_SOD)
        print("true_probabilidade_EOD", true_probabilidade_EOD)
        print("true_visited_EOD", true_visited_EOD)
        print("true_n_clusters_EOD", true_n_clusters_EOD)
        print("true_sentiment_EOD", true_sentiment_EOD)
        print("true_n_utterances_EOD", true_n_utterances_EOD)

        # CALCULAR MÉDIAS PARA O HTML
        import numpy as np # Garantir que temos numpy
        
        # Filtrar None e NaNs antes de calcular a média
        clean_sod = [s for s in sentiment_SOD if s is not None and not np.isnan(s)]
        clean_eod = [s for s in sentiment_EOD if s is not None and not np.isnan(s)]

        if clean_sod:
            avg_SOD = np.mean(clean_sod)
        else:
            avg_SOD = 0.5 # Valor neutro por defeito

        if clean_eod:
            avg_EOD = np.mean(clean_eod)
        else:
            avg_EOD = 0.5 # Valor neutro por defeito
            
        print(f"Médias Calculadas -> SOD: {avg_SOD:.4f} | EOD: {avg_EOD:.4f}")

else:
        #result, n_nodes = generate_markov_chain_separately_no(filename, n_clusters_speaker, matrix, df_final, threshold[0])
        result, n_nodes = generate_markov_chain_separately_no(filename, n_clusters_speaker, matrix, df_final, threshold[0], ACAO_SPEAKER_NAME, NOME_NODO_NA, SOD_NAME, EOD_NAME,
                                                      SOD_LABEL, EOD_LABEL)
        for i in result:
          print(result[i])

result_list = [result, (labels_speaker)]
with open(os.path.join('./Resultados/' + diretoria, file_result_flow), 'wb') as file:
        pickle.dump(result_list, file)

#variable_name = f'{Path(filename).stem}_{algorithm}_{metric_to_optimize}_{str(id_max)}'
# nome_modelo_limpo = MODELOS[0].replace(":", "_")
# variable_name = f"Fluxo_{nome_modelo_limpo}_{FILTRO_EXTRA_VALOR}_{LINGUA}_T{TEMP}"

if TIPO_DATASET == "GENERICO":
    nome_modelo_limpo = MODELOS[0].replace(":", "_")
    variable_name = f"Fluxo_{nome_modelo_limpo}_AllDomains_{LINGUA}_T{TEMP}"
else:
    # Para Política, usa o nome do ficheiro Excel (sem o .xlsx)
    nome_base = str(filename).split('.')[0]
    variable_name = f"Fluxo_{nome_base}_{algorithm}"

def save(graph: nx.MultiDiGraph, variable_name: str) -> str:
    #current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 
    full_filename = './Resultados/' + diretoria + '/' + f"{variable_name}_graph.dot"

    # Use write_dot diretamente
    nx.drawing.nx_pydot.write_dot(graph, full_filename)
    print("full_filename", full_filename)

    return full_filename

import requests
import os

# Guardar o .dot
dot_filename = save(result, variable_name)

# %%
#def show_file(filename: str) -> None:
#    s = graphviz.Source.from_file(filename, encoding='utf-8')
#    s.view()
#show_file(dot_filename)
#show_file('./Resultados/' + diretoria + '/'+ filename_std)

import shutil

def show_file(dot_filename):
    from graphviz import Source
    s = Source.from_file(dot_filename)
    # Verifica se o executável "dot" existe no PATH
    if shutil.which("dot") is None:
        print(" xecutável `dot` não encontrado .")
        return
    s.view()

# FUNÇÃO TSNE
def apply_tsne(vectors, perplexity=30, n_iter=1000):
    # Quantas linhas tem este speaker
    n_samples = vectors.shape[0]
    
    # Se tiver menos de 2 linhas, o t-SNE não funciona (precisa de vizinhos)
    if n_samples < 2:
        return np.zeros((n_samples, 2))

    # Se tiver poucos dados (ex: 10 linhas), baixamos a perplexidade para 9
    current_perplexity = perplexity
    if n_samples <= perplexity:
        current_perplexity = max(1, n_samples - 1)

    tsne_model = TSNE(n_components=2, perplexity=current_perplexity, max_iter=n_iter, random_state=2)
    tsne_result = tsne_model.fit_transform(vectors)
    return tsne_result

nomeFichtsne_speaker={}
tsne_result_speaker={}

# 1. Definir nomes dos ficheiros
for speaker in speakers:
    nomeFichtsne_speaker[speaker] = f'{MODEL_ML}_{algorithm}_{filename.split(".")[0]}_{metric_to_optimize}_{str(id_max)}_tsne_{speaker}.pkl'

# 2. Verificar se já existem ou se é preciso calcular
caminho_dir = os.path.join('./Resultados', diretoria)
os.makedirs(caminho_dir, exist_ok=True) # Garante que a pasta existe

files_exist = all(os.path.exists(os.path.join(caminho_dir, file)) for file in nomeFichtsne_speaker.values())

if not files_exist:
    print("A calcular t-SNE para todos os speakers...")
    for speaker in speakers:
        if speaker in vectors_speaker and len(vectors_speaker[speaker]) > 0:
            tsne_result_speaker[speaker] = apply_tsne(vectors_speaker[speaker])
            
            # Guardar pickle
            with open(os.path.join(caminho_dir, nomeFichtsne_speaker[speaker]), 'wb') as file:
                pickle.dump(tsne_result_speaker[speaker], file)
        else:
            print(f"em vetores para {speaker}, a saltar t-SNE.")
else:
    print("A carregar t-SNE já existente...")
    for speaker in speakers:
        caminho_ficheiro = os.path.join(caminho_dir, nomeFichtsne_speaker[speaker])
        if os.path.exists(caminho_ficheiro):
            with open(caminho_ficheiro, 'rb') as file:
                tsne_result_speaker[speaker] = pickle.load(file)

# 3. Desenhar e Guardar Gráficos
print("A gerar gráficos PNG...")
for speaker in speakers:
    if speaker not in tsne_result_speaker:
        continue

    plt.figure(figsize=(8, 4))
    
    # Desenhar
    sns.scatterplot(
        x=tsne_result_speaker[speaker][:, 0], 
        y=tsne_result_speaker[speaker][:, 1], 
        hue=y_predicted_speaker[speaker], 
        palette="viridis", 
        legend="full"
    )
    plt.title("t-SNE para " + speaker)

    plot_path_png = os.path.join(caminho_dir, f'tsne_{speaker}.png')
    plt.savefig(plot_path_png, format='png', bbox_inches='tight')
    print(f"   Gráfico salvo: {plot_path_png}")

    plt.close() # Fecha para libertar memória1

# FUNÇÃO PCA
def apply_pca(vectors, n_components=2):
    # Se houver menos linhas do que componentes, ajusta ou retorna zeros
    n_samples = vectors.shape[0]
    
    if n_samples < 1:
        return np.zeros((0, n_components))
    
    # Se tivermos ex: 1 linha, não dá para calcular 2 componentes
    n_comp_eff = min(n_components, n_samples)
    
    pca_model = PCA(n_components=n_comp_eff)
    pca_result = pca_model.fit_transform(vectors)
    
    # Se o resultado tiver menos colunas que o esperado (ex: 1 componente), 
    # adicionamos zeros para o gráfico não falhar
    if pca_result.shape[1] < n_components:
        padding = np.zeros((n_samples, n_components - pca_result.shape[1]))
        pca_result = np.hstack((pca_result, padding))
        
    return pca_result

print("\n>>> A calcular PCA...")

nomeFichpca_speaker={}
pca_result_speaker={}

# 1. Definir nomes dos ficheiros
for speaker in speakers:
    nomeFichpca_speaker[speaker] = f'{MODEL_ML}_{algorithm}_{filename.split(".")[0]}_{metric_to_optimize}_{str(id_max)}_pca_{speaker}.pkl'

# 2. Verificar e Calcular
caminho_dir = os.path.join('./Resultados', diretoria)
files_exist_pca = all(os.path.exists(os.path.join(caminho_dir, file)) for file in nomeFichpca_speaker.values())

if not files_exist_pca:
    for speaker in speakers:
        # Verifica se o speaker existe e se tem dados. Isto impede que o código crashe quando chega à "Acao"
        if speaker in vectors_speaker and len(vectors_speaker[speaker]) > 0:
            try:
                pca_result_speaker[speaker] = apply_pca(vectors_speaker[speaker])
                
                with open(os.path.join(caminho_dir, nomeFichpca_speaker[speaker]), 'wb') as file:
                    pickle.dump(pca_result_speaker[speaker], file)
            except Exception as e:
                print(f"Erro ao calcular PCA para {speaker}: {e}")
        else:
            print(f"  Aviso: Sem vetores para {speaker}, a saltar PCA.") 
else:
    print("A carregar PCA já existente...")
    for speaker in speakers:
        caminho = os.path.join(caminho_dir, nomeFichpca_speaker[speaker])
        if os.path.exists(caminho):
            with open(caminho, 'rb') as file:
                pca_result_speaker[speaker] = pickle.load(file)

# 3. Desenhar Gráficos
print("A gerar gráficos PNG (PCA)...")
for speaker in speakers:
    if speaker not in pca_result_speaker:
        continue

    plt.figure(figsize=(8, 4))
    sns.scatterplot(
        x=pca_result_speaker[speaker][:, 0], 
        y=pca_result_speaker[speaker][:, 1], 
        hue=y_predicted_speaker[speaker], 
        palette="viridis", 
        legend="full"
    )
    plt.title("PCA para " + speaker)
    
    plot_path_png = os.path.join(caminho_dir, f'pca_{speaker}.png')
    plt.savefig(plot_path_png, format='png', bbox_inches='tight')
    print(f"   Gráfico salvo: {plot_path_png}")
    
    plt.close()


# CARREGAMENTO E PREPARAÇÃO DOS DADOS DE TESTE
print("\n>>> A CARREGAR DADOS DE TESTE...")

#if filename_test.endswith(".csv"):
#    dados_test = pd.read_csv(filename_test, sep=',')
#else:
#    dados_test = pd.read_excel(filename_test)

print("\n>>> A CARREGAR DADOS DE TESTE (CÓPIA DIRETA DO TREINO)...")
dados_test = dados.copy()


# Garantir colunas essenciais
if "Tipo" not in dados_test.columns: dados_test["Tipo"] = "fala"
if "Ação" not in dados_test.columns: dados_test["Ação"] = np.nan

normalized_df_test = normalize_dataset(dados_test, regex=True, removeGreetings=False, speaker='both')
normalized_df_test['Speaker'] = normalized_df_test['Speaker'].astype(str).str.strip()

if 'MAPA_GENERICO' in globals():
    normalized_df_test['Speaker'] = normalized_df_test['Speaker'].replace(MAPA_GENERICO)
elif 'MAPA_SPEAKERS' in globals():
    normalized_df_test['Speaker'] = normalized_df_test['Speaker'].replace(MAPA_SPEAKERS)

# Substituímos 'top_speakers.index' pela lista 'speakers' que já temos do treino
# Se aparecer alguém novo no teste, vira 'Others'.
normalized_df_test['Speaker'] = normalized_df_test['Speaker'].apply(
    lambda x: x if x in speakers else 'Others'
)

y_predicted_test_speaker = {}
df_test_speaker = {}
utterances_test_speaker = {}
normalized_df_test_speaker = {}
names_speaker = {}
names_aux = 0

if 'Algo_speaker' not in locals(): Algo_speaker = {}
if 'labels_speaker' not in locals(): labels_speaker = {}

#  CARREGAR MODELOS (APENAS SE NÃO FOR KEYWORDS)
if algorithm != "keywords":
    print(">>> [IA] A carregar modelos de Clustering e Linguagem...")
    
    # Carregar SentenceTransformer
    if 'model' not in locals():
        print("   A carregar SentenceTransformer...")
        try:
            model = SentenceTransformer(MODEL_ML)
        except: pass

    # Carregar KMeans/DBSCAN do disco
    dir_res = os.path.abspath('./Resultados/' + diretoria)
    if os.path.exists(dir_res):
        files = os.listdir(dir_res)
        for speaker in speakers:
            if speaker in Algo_speaker: continue # Já existe em memória
            for f in files:
                # Procura: tem o nome do speaker, é .pkl e é kmeans
                if f.endswith(".pkl") and speaker.lower() in f.lower() and "kmeans" in f.lower() and "_utt_" not in f:
                    try:
                        obj = joblib.load(os.path.join(dir_res, f))
                        if hasattr(obj, 'predict'):
                            Algo_speaker[speaker] = obj
                            print(f"Modelo carregado para {speaker}: {f}")
                            break
                    except: pass
else:
    print("[Keywords] Modelos de IA não serão carregados.")

for speaker in speakers:
    print(f"A processar speaker: {speaker}...")
    
    # Filtrar dados
    normalized_df_test_speaker[speaker] = normalized_df_test[normalized_df_test['Speaker'] == speaker]
    
    # Preparar DataFrame
    cols = ["dialogue_id", "turn_id", 'Speaker', 'utterance']
    possiveis = ["trueLabel", "speech_act_gerado", "Ação", "clusters_acoes"]
    for c in possiveis:
        if c in normalized_df_test_speaker[speaker].columns: cols.append(c)
    
    df_test_speaker[speaker] = normalized_df_test_speaker[speaker][cols].copy()
    tamanho_df = len(df_test_speaker[speaker])
    y_predicted_test_speaker[speaker] = np.full(tamanho_df, -1) # Inicializa com -1

    # CASO 1: KEYWORDS (Lê Excel)
    if algorithm == "keywords":
        # Define a coluna a ler (Ação ou Speech Act)
        coluna_alvo = 'Ação' if speaker == 'Acao' and 'Ação' in df_test_speaker[speaker].columns else 'speech_act_gerado'
        
        if coluna_alvo in df_test_speaker[speaker].columns and speaker in labels_speaker:
            try:
                # Mapa inverso: {'Cooperação': 0, ...}
                mapa_inv = {str(v).strip(): k for k, v in labels_speaker[speaker].items()}
                # Limpa dados do Excel
                dados_limpos = df_test_speaker[speaker][coluna_alvo].astype(str).str.strip()
                # Mapeia
                y_predicted_test_speaker[speaker] = dados_limpos.map(mapa_inv).fillna(-1).astype(int)
                print(f"[Keywords] Mapeado com sucesso via '{coluna_alvo}'.")
            except Exception as e:
                print(f"[Keywords] Erro: {e}")
        else:
            print(f"     [Keywords] Falta coluna '{coluna_alvo}' ou labels.")

    # CASO 2: KMEANS
    elif algorithm != "none":
        if speaker in Algo_speaker:
            utts = df_test_speaker[speaker]["utterance"].tolist()
            # Só corre se o modelo existir
            if utts and 'model' in locals():
                try:
                    vecs = model.encode(utts)
                    if hasattr(Algo_speaker[speaker], 'predict'):
                        y_predicted_test_speaker[speaker] = Algo_speaker[speaker].predict(vecs)
                        print(f"[IA] Predição feita ({len(utts)} falas).")
                except Exception as e: 
                    print(f"[IA] Erro na predição: {e}")
        else:
            print(f"     [IA] Sem modelo carregado para '{speaker}'.")

    # Guardar resultados
    df_test_speaker[speaker][f'clusters_{speaker}'] = y_predicted_test_speaker[speaker]

    # Preparar Nomes para o Grafo
    if speaker in n_clusters_speaker:
        try:
            if algorithm == "keywords":
                 names_speaker[speaker] = list(labels_speaker[speaker].values())
            else:
                 fim = n_clusters_speaker[speaker] + names_aux
                 if fim <= len(names):
                     names_speaker[speaker] = names[names_aux : fim]
                     names_aux += n_clusters_speaker[speaker]
        except: pass

print("A gerar df_final_test...")

# 1. Juntar todos os DataFrames parciais (Presidente, PS, etc.)
dfs_validos = [df_test_speaker[s] for s in speakers if s in df_test_speaker and not df_test_speaker[s].empty]

if dfs_validos:
    # Cria o DataFrame unificado
    df_final_test = pd.concat(dfs_validos, ignore_index=True)
    
    # 2. Identificar colunas de clusters (ex: clusters_Presidente)
    cols_existentes = [c for c in df_final_test.columns if c.startswith("clusters_")]
    
    if cols_existentes:
        # Soma as predições de todos os oradores numa coluna única
        df_final_test['clusters_final'] = df_final_test[cols_existentes].fillna(0).sum(axis=1).astype(int)
        print(f"Coluna 'clusters_final' criada somando: {cols_existentes}")
    else:
        # Se não houver colunas (ex: modo Keywords que falhou), cria a zeros
        print("  Aviso: Nenhuma coluna de clusters encontrada. A criar coluna a zeros.")
        df_final_test['clusters_final'] = 0
else:
    print("Nenhum dado foi processado no teste. A criar DF vazio.")
    df_final_test = pd.DataFrame(columns=['Speaker', 'clusters_final', 'dialogue_id', 'turn_id', 'utterance'])

# Ordenação Cronológica (Importante para o grafo não ficar baralhado)
if not df_final_test.empty:
    df_final_test = df_final_test.sort_values(by=['dialogue_id', 'turn_id']).reset_index(drop=True)

print("Chegou aqui sem erros!")


def extrair_label_puro(nome_no):
    # Remove contagens tipo do final para bater certo com o Excel
    match = re.search(r'^(.+?)(?:\s*\(\d+\))?$', str(nome_no))
    if match:
        return match.group(1).strip()
    return str(nome_no)

total_falas_nos_clusters = 0
total_falas_no_df = len(df_final)

if 'clusters_speaker' in globals():
    for speaker, clusters in clusters_speaker.items():
        if isinstance(clusters, dict):
            for c_id, lista_frases in clusters.items():
                total_falas_nos_clusters += len(lista_frases)

print(f"1. Total de linhas no Excel de Treino (df_final): {total_falas_no_df}")
print(f"2. Total de frases armazenadas nos Clusters: {total_falas_nos_clusters}")

if abs(total_falas_no_df - total_falas_nos_clusters) < 50:
    print("O grafo contém os dados de treino!")
else:
    print("Há uma discrepância grande. Verifica se as 'Ações' estão a ser contadas.")

# 1. DEFINIÇÃO DAS FUNÇÕES DE CÁLCULO (FF1)
import Levenshtein
import networkx as nx

def calcular_nf_iterativo(result, test_paths, avg_dialogue_length):
    """Calcula a Normalized Flow (NF) comparando caminhos de teste com o grafo."""
    lista_nos = list(result.nodes)
    if not lista_nos: return avg_dialogue_length

    real_source = None
    real_target = None

    # Procura nós que contenham SOD e EOD
    for node in lista_nos:
        node_str = str(node).upper()
        if "SOD" in node_str: real_source = node
        if "EOD" in node_str: real_target = node

    if real_source is None: real_source = lista_nos[0]
    if real_target is None: real_target = lista_nos[-1]
    
    total_dist = 0
    valid_paths = 0
    
    for p in test_paths:
        try:
            # Tenta encontrar o caminho mais curto no grafo treinado
            train_path_raw = nx.shortest_path(result, source=real_source, target=real_target)
            train_path_clean = [str(n).split('(')[0].strip() for n in train_path_raw]
            
            # Calcula a distância de Levenshtein entre o caminho real e o do grafo
            dist = Levenshtein.distance(p, train_path_clean)
            total_dist += dist
            valid_paths += 1
        except:
            # Se não houver caminho no grafo, penaliza com o tamanho do caminho
            total_dist += len(p)
            valid_paths += 1

    if valid_paths == 0: return avg_dialogue_length
    return total_dist / valid_paths

def flow_f1(result, test_paths, df_referencia, diretoria):
    """Calcula o Flow F1 Score baseando-se na cobertura de nós e fluxo."""
    total_utterances = len(df_referencia)
    num_dialogos = df_referencia['dialogue_id'].nunique() if 'dialogue_id' in df_referencia.columns else 1
    avg_dialogue_length = total_utterances / num_dialogos if num_dialogos > 0 else 1
    
    # NC (Node Coverage)
    nc = len(result.nodes) / total_utterances if total_utterances > 0 else 0
    if nc > 1: nc = 1.0 # Cap a 100%
    
    # NF (Normalized Flow)
    raw_nf = calcular_nf_iterativo(result, test_paths, avg_dialogue_length)
    nf = raw_nf / avg_dialogue_length if avg_dialogue_length > 0 else 1
    if nf > 1: nf = 1.0
    
    # FF1 Harmony Score
    denominador = (1 - nc) + (1 - nf)
    if denominador == 0:
        ff1 = 0
    else:
        ff1 = (2 * (1 - nc) * (1 - nf)) / denominador
        
    print(f"📊 Métricas Intermédias: NC={nc:.2f}, NF={nf:.2f}")
    return ff1

def converter_dialogos_para_caminhos(dados, y_predicted_speaker, labels_speaker, speakers):
    test_paths = []
    
    # Se labels_speaker estiver vazio, tenta recuperar
    if not labels_speaker:
        print("  Aviso: labels_speaker vazio. A tentar usar mapeamento genérico.")
    
    for dialogue_id in dados['dialogue_id'].unique():
        caminho_atual = ["SOD"]
        df_dialogo = dados[dados['dialogue_id'] == dialogue_id].sort_values(by='turn_id')
        
        for _, row in df_dialogo.iterrows():
            speaker = row['Speaker']
            
            # Tenta obter o nome do cluster
            cluster_name = None
            
            # Opção 1: Pela coluna clusters_final (já calculada)
            if 'clusters_final' in row:
                idx = row['clusters_final']
                # Tenta mapear ID -> Nome se possível
                if speaker in labels_speaker and idx in labels_speaker[speaker]:
                    cluster_name = labels_speaker[speaker][idx]
                else:
                    # Se não houver mapa, usa o ID
                    cluster_name = f"Cluster {idx}"
            
            if cluster_name:
                # Formato do nó: "SPEAKER -> ClusterName"
                caminho_atual.append(f"{speaker} -> {cluster_name}")
        
        caminho_atual.append("EOD")
        test_paths.append(caminho_atual)
        
    return test_paths

print("\n>>> A CALCULAR FLOW F1 SCORE...")

ff1 = 0.0

# 1. Recuperar nomes dos ficheiros
f_train = globals().get('filename', '')
f_test = globals().get('filename_test', '')

# Os ficheiros têm de ser diferentes!
if f_train == f_test:
    print(f"  AVISO: O ficheiro de Treino e Teste são iguais ('{f_train}').")

elif 'result' in locals() and 'df_final_test' in locals() and not df_final_test.empty:
    try:
        # Garante variáveis auxiliares
        if 'y_predicted_test_speaker' not in locals(): y_predicted_test_speaker = {}
        labels_map = globals().get('labels_speaker', {})
        
        print(f"   Treino: {f_train} | Teste: {f_test}")
        print("   A converter diálogos de teste em caminhos...")
        
        test_paths = converter_dialogos_para_caminhos(df_final_test, y_predicted_test_speaker, labels_map, speakers)
        
        if test_paths:
            print(f"   {len(test_paths)} caminhos de teste gerados.")
            ff1 = flow_f1(result, test_paths, df_final_test, diretoria)
            
            # Guardar na variável global
            globals()['ff1'] = ff1
            print(f"FF1 Calculado: {ff1:.4f}")
        else:
            print("  Aviso: Não foi possível gerar caminhos de teste.")

    except Exception as e:
        print(f"Erro no cálculo FF1: {e}")
        import traceback
        traceback.print_exc()
else:
    print("A saltar o cálculo de FF1: Falta o grafo ou dados de teste.")

def save_graph_and_variables(graph, variable_name, diretoria):
    base_path = os.path.join('./Resultados', diretoria)
    os.makedirs(base_path, exist_ok=True)

    dot_filename = os.path.join(base_path, f"{variable_name}_graph.dot")
    pkl_filename = os.path.join(base_path, f"{variable_name}_vars.pkl")

    # 1. Mapeamento de Labels
    label_to_cluster_speaker = {}
    if 'labels_speaker' in globals():
        for speaker in speakers:
            if speaker in labels_speaker:
                label_to_cluster_speaker[speaker] = {str(v).lower().strip(): k for k, v in labels_speaker[speaker].items()}

    # 2. Criar dicionário de IDs
    cluster_id_speaker = {}
    for node in graph.nodes():
        # Lógica simples para extrair nome
        label_puro = str(node).split("->")[-1].split("(")[0].strip().lower()
        for speaker in speakers:
            if speaker in label_to_cluster_speaker and label_puro in label_to_cluster_speaker[speaker]:
                cluster_id_speaker[node] = label_to_cluster_speaker[speaker][label_puro]

    # Se 'df_final' existir nas variáveis globais, usa-o. Caso contrário, tenta o de teste.
    df_para_interface = globals().get('df_final') 
    
    if df_para_interface is None or df_para_interface.empty:
        print("  Aviso: df_final (Treino) vazio ou inexistente. A usar Teste.")
        df_para_interface = globals().get('df_final_test')

    # Converter para dicionário (para o pickle não dar erro de versão pandas)
    df_dict = df_para_interface.to_dict('records') if df_para_interface is not None else []

    variables = {
        "speakers": globals().get('speakers', []),
        "speaker_cluster_counts": globals().get('speaker_cluster_counts', {}),
        "fsc_speaker": globals().get('fsc_speaker', {}),
        "avg_EOD": globals().get('avg_EOD', 0.5),
        "avg_SOD": globals().get('avg_SOD', 0.5),
        "diretoria": diretoria,
        "density": nx.density(graph) if graph else 0,
        "clusters_speaker": globals().get('clusters_speaker', {}),
        "cluster_id_speaker": cluster_id_speaker,
        "df_final": df_dict
    }

    # Guardar Pickle
    with open(pkl_filename, "wb") as f:
        pickle.dump(variables, f)

    # 4. Guardar DOT
    nx_pydot.write_dot(graph, dot_filename)
    
    # Injetar referência no DOT para o JS saber qual ficheiro ler
    with open(dot_filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    with open(dot_filename, "w", encoding="utf-8") as f:
        f.write(f'// variables_file={os.path.basename(pkl_filename)}\n')
        f.writelines(lines)

    print(f"Ficheiro de interface gerado com {len(df_dict)} registos (Falas + Ações).")
    return dot_filename

if 'fsc_speaker' not in locals(): fsc_speaker = {s: 0.0 for s in speakers}
if 'avg_SOD' not in locals(): avg_SOD = 0.5
if 'avg_EOD' not in locals(): avg_EOD = 0.5

try:
    print(f">>> A exportar para: {diretoria}")
    save_graph_and_variables(result, "DAs_train_emo_none_NONE_1", diretoria)
    print("\n Abre a interface!.")
except Exception as e:
    print(f"Erro ao guardar interface: {e}")

if 'fsc_speaker' not in locals(): fsc_speaker = {s: 0.0 for s in speakers}
if 'avg_SOD' not in locals(): avg_SOD = 0.5
if 'avg_EOD' not in locals(): avg_EOD = 0.5
if 'density' not in locals(): density = 0.0
if 'speaker_cluster_counts' not in locals(): speaker_cluster_counts = {}
if 'clusters_speaker' not in locals(): clusters_speaker = {}

try:
    print(f">>> A exportar para: {diretoria}")
    save_graph_and_variables(result, "DAs_train_emo_none_NONE_1", diretoria)
    print("\n O ficheiro da interface foi gerado.")
except Exception as e:
    print(f"Erro ao guardar interface: {e}")


for i, t in enumerate(threshold):
    print(f"\n--- A processar Threshold {t} ---")

    dir_path = os.path.abspath('./Resultados/' + diretoria)
    dot_filename = None

    if os.path.exists(dir_path):
        # Procura qualquer ficheiro que acabe em .dot na pasta
        files = os.listdir(dir_path)
        dot_files = [f for f in files if f.endswith(".dot")]
        
        if dot_files:
            dot_filename = os.path.join(dir_path, dot_files[0])
            print(f"Ficheiro DOT encontrado: {dot_files[0]}")
        else:
            print("Nenhum ficheiro .dot encontrado na pasta.")
    else:
        print(f"Pasta não encontrada: {dir_path}")

    nomeFichParSystem = f'{Path(filename).stem}_{metric_to_optimize}_{algorithm}_{str(id_max)}_param_{t}.txt'

    v_measure_test_speaker={}
    v_measure_train_speaker={}

    for speaker in speakers:
        v_measure_test_speaker[speaker] = 0
        v_measure_train_speaker[speaker] = 0
    v_measure_test = 0
    v_measure_train = 0
    accuracy = 0

    # 3. CARREGAR O GRAFO E GERAR IMAGEM
    flow = None
    
    if dot_filename and os.path.exists(dot_filename):
        try:
            # Carrega o grafo para memória
            (flow,) = pydot.graph_from_dot_file(dot_filename)
            
            # Guarda logo a imagem PNG para garantir que fica feita
            output_png = os.path.join(dir_path, 'flow_network.png')
            try:
                flow.write_png(output_png)
                print(f"   Imagem do grafo salva: {output_png}")
            except:
                pass # Se falhar o PNG, não faz mal, continuamos para a accuracy

        except Exception as e:
            print(f"Erro ao ler o ficheiro DOT: {e}")
            continue # Se não conseguimos ler o grafo, saltamos este loop
    else:
        print("A saltar o cálculo da accuracy.")
        continue

    # CALCULAR ACCURACY
    try:
        accuracy = calcular_accuracy_transicoes(df_final_test, flow, names_speaker)
        print("ACCURACY", accuracy)
    except Exception as e:
        print(f"Erro ao calcular accuracy: {e}")
        accuracy = 0

    df_final_test_clear = df_final_test.copy()

    #def modify_value(row, speakers):
    #    for speaker in speakers:
    #        if row['Speaker'] == speaker:
    #            return speaker[0].lower() + str(row['clusters_final'])
    #    return row['clusters_final']

    #def modify_true_value(row, speakers):
    #    for speaker in speakers:(flow,) = pydot.graph_from_dot_file(dot_filename)

    accuracy = calcular_accuracy_transicoes(df_final_test, flow, names_speaker) 
    print("ACCURACY", accuracy)

    df_final_test_clear = df_final_test.copy()


    def modify_value(row, speakers):
        for speaker in speakers:
            if row['Speaker'] == speaker:
                return speaker[0].lower() + str(row['clusters_final'])
        return row['clusters_final']

    def modify_true_value(row, speakers):
        for speaker in speakers:
            if row['Speaker'] == speaker:
                return speaker[0].lower() + str(row['trueLabel'])
        return row['trueLabel']

    def modify_train_value(row, speakers):
        for speaker in speakers:
            if row['Speaker'] == speaker:
                return speaker[0].lower() + str(row['n_clusters_final'])
        return row['n_clusters_final']

    df_final_test_clear['clusters_final'] = df_final_test_clear.apply(modify_value, axis=1, speakers=speakers)
    df_final['n_clusters_final'] = df_final.apply(lambda row: modify_train_value(row, speakers), axis=1)

    if acts:
        df_final_test_clear['trueLabel'] = df_final_test_clear.apply(lambda row: modify_true_value(row, speakers), axis=1)
        df_final['trueLabel'] = df_final.apply(lambda row: modify_true_value(row, speakers), axis=1)

        for speaker in speakers:
            v_measure_test_speaker[speaker] = metrics.v_measure_score(df_test_speaker[speaker]['trueLabel'], df_test_speaker[speaker]['clusters_'+ speaker], beta = beta) 
            
            v_measure_train_speaker[speaker] = metrics.v_measure_score(normalized_df_speaker[speaker]['trueLabel'], normalized_df_speaker[speaker]['clusters_speaker_'+ speaker], beta = beta)
        
        # Both
        v_measure_test = metrics.v_measure_score(df_final_test_clear['trueLabel'], df_final_test_clear['clusters_final'], beta = beta)
        v_measure_train = metrics.v_measure_score(df_final['trueLabel'], df_final['n_clusters_final'], beta = beta)

    silhouette_score = (sum(silhouette_speaker.values())) / len(speakers)
    #silhouette_score_norm = (silhouette_score + 1) / 2
    #Silhacc = (2 * accuracy * silhouette_score_norm) / (accuracy + silhouette_score_norm)

    # GUARDAR FICHEIRO DE PARÂMETROS
    file = open('./Resultados/' + diretoria +'/'+ nomeFichParSystem, 'w')
    
    for speaker in speakers:
        # Só tenta calcular se o speaker estiver no dicionário
        if speaker in y_predicted_speaker and len(y_predicted_speaker[speaker]) > 0:
            try:
                # Filtra os -1 (ruído) e conta os clusters únicos
                n_clusters_validos = len(np.unique(y_predicted_speaker[speaker][y_predicted_speaker[speaker] != -1]))
                print("\nClusters "+ speaker +" :", n_clusters_validos, file=file)
            except:
                print(f"\nClusters {speaker} : Erro cálculo", file=file)
        
        # Caso especial para a 'Acao' se não estiver no y_predicted_speaker
        elif speaker == 'Acao' and 'clusters_acoes' in df_final.columns:
             n_clusters_acao = df_final['clusters_acoes'].nunique()
             print(f"\nClusters {speaker} : {n_clusters_acao}", file=file)
        
        else:
            print(f"\nClusters {speaker} : 0 (Sem dados)", file=file)

        # Escrever V-Measure se existir
        if speaker in v_measure_test_speaker:
            print("V-measure Teste "+ speaker +" :", v_measure_test_speaker[speaker], file=file)
        if speaker in v_measure_train_speaker:
            print("V-measure Treino "+ speaker +" :", v_measure_train_speaker[speaker], file=file)

    print("\nAccuracy: ", accuracy, file=file)
    print("\nV-measure Teste: ", v_measure_test, file=file)
    print("\nV-measure Treino: ", v_measure_train, file=file)
    #print("Silhacc: ", Silhacc, file=file)

    file.close()

# Métricas novas
def verificar_caminhos_sod_eod(df_final_test, names_speaker):
    caminhos_sod_eod = []  # Lista para armazenar os caminhos de SOD ao EOD
    # Iterar sobre todos os diálogos
    for dialogue_id in set(df_final_test["dialogue_id"]):
        dialogue_utterances = df_final_test[df_final_test["dialogue_id"] == dialogue_id]
        dialogue_utterances = dialogue_utterances.sort_values(by='turn_id').reset_index(drop=True)

        caminho_atual = []  # Caminho atual do diálogo
        # Verificar se o diálogo não está vazio
        if len(dialogue_utterances) == 0:
            continue

        # Começo do diálogo (SOD)
        sod_label = None

        for speaker in speakers:
            if dialogue_utterances['Speaker'][0] == speaker:
                #print (df_final_test)
                sod_label = names_speaker[speaker][int(dialogue_utterances["clusters_" + speaker][0])]
                #print("sod_label",sod_label)
                break

        # Adicionar a primeira transição (SOD) ao caminho
        caminho_atual.append(f"SOD -> {sod_label}")

        # Percorrer as utterances para capturar as transições do SOD até o EOD
        for i in range(1, len(dialogue_utterances)):  # Começar a partir do segundo turno
            for speaker in speakers:
                if dialogue_utterances['Speaker'][i] == speaker:
                    label_atual = names_speaker[speaker][int(dialogue_utterances["clusters_" + speaker][i])]
                    #print("label_atual",label_atual)
                    break

            # Adicionar as transições anteriores ao caminho, mas sem os números entre parênteses
            caminho_atual.append(f"{label_atual}")

        # Adicionar o EOD explicitamente como o último item, com "sys" sempre como último
        eod_label = None
        for speaker in speakers:
            if dialogue_utterances['Speaker'].iloc[-1] == speaker:
                eod_label = names_speaker[speaker][int(dialogue_utterances["clusters_" + speaker].iloc[-1])]
                #print("eod_label", eod_label)
                break

        # Adicionar o EOD ao caminho
        caminho_atual.append(f"{eod_label} -> EOD")

        # Adicionar o caminho completo do diálogo (SOD ao EOD) à lista
        caminhos_sod_eod.append(caminho_atual)

    return caminhos_sod_eod

# Carregar os dados
if filename[-4:] == ".csv":
    dados_teste = pd.read_csv(filename_test, sep=',')
    dados_treino = pd.read_csv(filename, sep=',')
else:
    dados_teste = pd.read_excel(filename_test)
    dados_treino = pd.read_excel(filename)

def calcular_nf_iterativo(result, test_paths, avg_dialogue_length):
    """Garante que a métrica retorna sempre um número e encontra SOD/EOD."""
    lista_nos = list(result.nodes)
    if not lista_nos: return avg_dialogue_length

    real_source = None
    real_target = None

    # Procura nós que contenham SOD e EOD
    for node in lista_nos:
        node_str = str(node).upper()
        if "SOD" in node_str: real_source = node
        if "EOD" in node_str: real_target = node

    # Se falhar a procura, força o uso dos extremos do grafo para não crashar
    if real_source is None: real_source = lista_nos[0]
    if real_target is None: real_target = lista_nos[-1]
    
    print(f"NF Calculation: Start={real_source} | End={real_target}")

    total_dist = 0
    for p in test_paths:
        try:
            train_path_raw = nx.shortest_path(result, source=real_source, target=real_target)
            train_path_clean = [str(n).split('(')[0].strip() for n in train_path_raw]
            total_dist += Levenshtein.distance(p, train_path_clean)
        except:
            total_dist += len(p)

    return total_dist / len(test_paths) if len(test_paths) > 0 else avg_dialogue_length

def flow_f1(result, test_paths, df_referencia, diretoria):
    total_utterances = len(df_referencia)
    num_dialogos = df_referencia['dialogue_id'].nunique() if 'dialogue_id' in df_referencia.columns else 1
    avg_dialogue_length = total_utterances / num_dialogos if num_dialogos > 0 else 1
    
    nc = len(result.nodes) / total_utterances if total_utterances > 0 else 0
    raw_nf = calcular_nf_iterativo(result, test_paths, avg_dialogue_length)
    
    # Força raw_nf a ser número para evitar o TypeError
    if raw_nf is None: raw_nf = avg_dialogue_length
    
    nf = raw_nf / avg_dialogue_length if avg_dialogue_length > 0 else 1
    ff1 = (2 * (1 - nc) * (1 - nf)) / ((1 - nc) + (1 - nf)) if ((1 - nc) + (1 - nf)) > 0 else 0
        
    print(f"Métricas: NC={nc:.4f}, NF={nf:.4f}, FF1={ff1:.4f}")
    return ff1


def converter_dialogos_para_caminhos(dados, y_predicted_speaker, labels_speaker, speakers):
    test_paths = []
    
    # 1. Agrupar por diálogo
    for dialogue_id in dados['dialogue_id'].unique():
        caminho_atual = ["SOD"]
        
        # Filtra apenas as linhas deste diálogo
        df_dialogo = dados[dados['dialogue_id'] == dialogue_id].sort_values(by='turn_id')
        
        # 2. Contador local para cada speaker neste diálogo
        for _, row in df_dialogo.iterrows():
            speaker = row['Speaker']
            
            # Só processa se o speaker estiver nos analisados e tiver predições
            if speaker in speakers and speaker in y_predicted_speaker:
                try:
                    # Tenta obter o cluster para esta linha específica
                    # Se falhar por índice, usamos o valor da coluna 'clusters_final' se existir
                    if 'clusters_final' in row:
                        idx_real = row['clusters_final']
                    else:
                        # Fallback: se não conseguirmos mapear a linha ao array, ignoramos esta fala
                        continue
                    
                    # 3. Traduzir o número para o nome (ex: 0 -> 'Cooperação')
                    if speaker in labels_speaker and idx_real in labels_speaker[speaker]:
                        label_texto = labels_speaker[speaker][idx_real]
                        caminho_atual.append(f"{speaker} -> {label_texto}")
                except:
                    continue # Se der qualquer erro, salta esta fala

        caminho_atual.append("EOD")
        test_paths.append(caminho_atual)

    return test_paths

# # Guardar no dot para interface
def normalizar_texto(texto):
    if not isinstance(texto, str):
        texto = str(texto)
    return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

def extrair_label_puro(nome_no):
    match = re.search(r'->\s*(.+?)(?:\s*\(\d+\))?$', nome_no)
    if match:
        return match.group(1).strip().lower()
    return nome_no.lower()

def save(graph: nx.MultiDiGraph, variable_name: str) -> str:
    #current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 
    full_filename = './Resultados/' + diretoria + '/' + f"{variable_name}_graph.dot"

    # Use write_dot diretamente
    nx.drawing.nx_pydot.write_dot(graph, full_filename)
    print("full_filename", full_filename)

    return full_filename

def save_graph_and_variables(graph, variable_name, diretoria):
    base_path = os.path.join('./Resultados', diretoria)
    os.makedirs(base_path, exist_ok=True)

    dot_filename = os.path.join(base_path, f"{variable_name}_graph.dot")
    pkl_filename = os.path.join(base_path, f"{variable_name}_vars.pkl")

    # 1. Mapeamento de Labels para IDs
    label_to_cluster_speaker = {}
    for speaker in speakers:
        if speaker in labels_speaker:
            label_to_cluster_speaker[speaker] = {normalizar_texto(v): k for k, v in labels_speaker[speaker].items()}

    # 2. Criar dicionário de IDs de cluster por nó
    cluster_id_speaker = {}
    for node in graph.nodes():
        label_puro = extrair_label_puro(node)
        for speaker in speakers:
            if speaker in label_to_cluster_speaker and label_puro in label_to_cluster_speaker[speaker]:
                cluster_id_speaker[node] = label_to_cluster_speaker[speaker][label_puro]

    # Montar o dicionário de variáveis para o Pickle
    variables = {
        "speakers": globals().get('speakers', []),
        "speaker_cluster_counts": globals().get('speaker_cluster_counts', {}),
        "fsc_speaker": globals().get('fsc_speaker', {}),
        "avg_EOD": globals().get('avg_EOD', 0.5),
        "avg_SOD": globals().get('avg_SOD', 0.5),
        "diretoria": diretoria,
        "density": nx.density(graph) if graph else 0,
        "clusters_speaker": globals().get('clusters_speaker', {}),
        "cluster_id_speaker": cluster_id_speaker,
        "df_final": globals().get('df_final', pd.DataFrame()).to_dict('records') if 'df_final' in globals() else None
    }

    # Guardar Pickle
    with open(pkl_filename, "wb") as f:
        pickle.dump(variables, f)

    # Guardar DOT com cabeçalhos Lato para a interface
    pydot_graph = nx_pydot.to_pydot(graph)
    pydot_graph.write_raw(dot_filename)
    
    # Gerar PDF
    pdf_filename = dot_filename.replace(".dot", ".pdf")
    try:
        pydot_graph.write_pdf(pdf_filename, prog=r'C:\Program Files\Graphviz\bin\dot.exe')
        print(f"PDF gerado com sucesso em: {pdf_filename}")
    except Exception as e:
        print(f"Aviso: Não foi possível gerar o PDF. Erro: {e}")

    with open(dot_filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if not inserted and "digraph" in line:
            new_lines.append('graph [fontname="Lato"];\nnode [fontname="Lato"];\nedge [fontname="Lato"];\n')
            new_lines.append(f'// variables_file={os.path.basename(pkl_filename)}\n')
            inserted = True

    # E ADICIONAR errors="ignore"
    with open(dot_filename, "w", encoding="utf-8", errors="ignore") as f:
        f.writelines(new_lines)

    return dot_filename

# 1. Garantir que as variáveis de sistema existem para não crashar o save
if 'fsc_speaker' not in locals(): fsc_speaker = {}
if 'avg_SOD' not in locals(): avg_SOD = 0.5
if 'avg_EOD' not in locals(): avg_EOD = 0.5
if 'density' not in locals(): density = 0.0

print(">>> A gerar o grafo final...")
try:
    print(">>> A exportar dados para a Interface...")
    save_graph_and_variables(result, "DAs_train_emo_none_NONE_1", diretoria)
    print("Sucesso! Ficheiros gerados em: ./Resultados/" + diretoria)
except Exception as e:
    print(f"Erro ao exportar para interface: {e}")

# # Interface

def sanitize(obj):
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, np.generic):  # numpy.float64, etc.
        return obj.item()
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    else:
        return obj

import glob

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_dot():
    file = request.files['dot']
    filename = file.filename
    
    # Guarda o .dot temporariamente na pasta uploads
    os.makedirs('uploads', exist_ok=True)
    dot_path = os.path.join('uploads', filename)
    file.save(dot_path)

    # 1. Extrair o nome do ficheiro .pkl (porque está num comentário // variables_file=...)
    pkl_filename = None
    with open(dot_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith("// variables_file="):
                pkl_filename = line.strip().split("=", 1)[1]
                break
                
    if not pkl_filename:
        return jsonify({"error": "A referência variables_file não foi encontrada no .dot"}), 400

    # 2. Procurar o ficheiro .pkl dentro da pasta Resultados
    import glob
    found_pkls = glob.glob(f'./Resultados/*/{pkl_filename}')
    
    if not found_pkls:
        print(f"Erro Flask: Não encontrei o ficheiro {pkl_filename}")
        return jsonify({"error": "Ficheiro .pkl não encontrado"}), 404
        
    pkl_path = found_pkls[0]

    # 3. Carregar os dados originais
    with open(pkl_path, 'rb') as f:
        vars_loaded = pickle.load(f)

    with open(dot_path, 'r', encoding='utf-8') as f:
        dot_code = f.read()

    # Limpeza para evitar erros de JSON
    if 'df_final' in vars_loaded and hasattr(vars_loaded['df_final'], 'tolist'):
        vars_loaded['df_final'] = vars_loaded['df_final'].tolist()
        
    vars_loaded = sanitize(vars_loaded)

    # Reconstruir mapa de cores
    colors_map = {}
    for spk in vars_loaded.get('speakers', []):
        colors_map[spk] = atribuir_cor(spk)

    # 4. Devolver TUDO atualizado ao JavaScript
    return jsonify({
        'dot_code': dot_code,
        'speakers': vars_loaded.get('speakers', []), 
        'speaker_cluster_counts': vars_loaded.get('speaker_cluster_counts', {}),
        'fsc_speaker': vars_loaded.get('fsc_speaker', {}), 
        'avg_EOD': float(vars_loaded.get('avg_EOD', 0.5)),  
        'avg_SOD': float(vars_loaded.get('avg_SOD', 0.5)),
        'df_final': vars_loaded.get('df_final', []),
        'density': float(vars_loaded.get('density', 0)),
        'clusters_speaker': vars_loaded.get('clusters_speaker', {}),  
        'diretoria': vars_loaded.get('diretoria', ''), 
        'cluster_id_speaker': vars_loaded.get('cluster_id_speaker', {}),
        'colors': colors_map 
    })

# %%
import os
#os._exit(00)

# %% [markdown]
# ## Frontend

# %%
def extract_variables_file_from_dot(dot_filepath: str) -> str:
    with open(dot_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            print(line)
            if line.strip().startswith("// variables_file="):
                return line.strip().split("=", 1)[1]
    raise ValueError("variável 'variables_file' não encontrada no .dot")

def load_data_from_dot(dot_filepath: str):
    global speakers, speaker_cluster_counts, fsc_speaker
    global avg_EOD, avg_SOD, df_final, density
    global clusters_speaker, diretoria, cluster_id_speaker
    global dot_code, result

    # 1. Ler o grafo do ficheiro .dot
    result = nx_pydot.read_dot(dot_filepath)
    pkl_filename = extract_variables_file_from_dot(dot_filepath)
    
    if pkl_filename is None:
        raise ValueError("O ficheiro .dot não contém a referência ao ficheiro .pkl.")

    # 3. Construir caminho completo do .pkl com base na mesma pasta do .dot
    dot_dir = os.path.dirname(dot_filepath)
    pkl_filepath = os.path.join(dot_dir, pkl_filename)

    # 4. Carregar variáveis do .pkl
    with open(pkl_filepath, "rb") as f:
        vars_loaded = pickle.load(f)

    # 5. Extrair variáveis
    speakers = vars_loaded["speakers"]
    speaker_cluster_counts = vars_loaded["speaker_cluster_counts"]
    fsc_speaker = vars_loaded["fsc_speaker"]
    avg_EOD = vars_loaded["avg_EOD"]
    avg_SOD = vars_loaded["avg_SOD"]
    df_final = vars_loaded["df_final"]
    density = vars_loaded["density"]
    clusters_speaker = vars_loaded["clusters_speaker"]
    diretoria = vars_loaded["diretoria"]
    cluster_id_speaker = vars_loaded["cluster_id_speaker"]

    # 6. Ler código do ficheiro .dot como string
    with open(dot_filepath, encoding='utf-8', errors='ignore') as f:
        dot_code = f.read()

    return {
        "speakers": speakers,
        "speaker_cluster_counts": speaker_cluster_counts,
        "fsc_speaker": fsc_speaker,
        "avg_EOD": avg_EOD,
        "avg_SOD": avg_SOD,
        "df_final": df_final,
        "density": density,
        "clusters_speaker": clusters_speaker,
        "diretoria": diretoria,
        "cluster_id_speaker": cluster_id_speaker,
        "dot_code": dot_code,
        "graph_result": result,
    }

import webbrowser
import json 

cores_personalizadas = {}

def atribuir_cor(speaker):
    # Converter para maiúsculas para facilitar a comparação
    s_upper = str(speaker).upper().strip()

    # User / Cliente -> Azul Claro (#33CCFF)
    if s_upper in ["USER", "CLIENTE", "USR", "CONSUMIDOR"]:
        return "#33CCFF"

    # System / Agente -> Azul Escuro (#00008B)
    if s_upper in ["SYSTEM", "AGENTE", "BOT", "ASSISTENTE", "SYS", "OPERADOR"]:
        return "#00008B"

    # Nós de Início e Fim -> Dourado
    if s_upper in ["SOD", "EOD"]:
        return "#FFD700"

    # --- 2. CORES POLÍTICAS ---
    # Só são usadas se o nome do speaker contiver a sigla do partido
    PARTY_COLORS = {
        "BE": "#B22222", "PCP": "#FF0000", "PEV": "#008000", 
        "L": "#9ACD32", "PS": "#FF69B4", "PAN": "#20B2AA", 
        "JPP": "#32CD32", "GOV": "#000000", "IL": "#00BFFF", 
        "PSD": "#FFA500", "CDS-PP": "#1E90FF", "CH": "#000080", 
        "PRESIDENTE": "#708090"
    }
    
    for key, color in PARTY_COLORS.items():
        if key in s_upper:
            return color

    # --- 3. CORES "FRIENDS" ---
    FRIENDS_COLORS = {
        "RACHEL": "#FF6B6B", "JOEY": "#FF9966", "MONICA": "#66CDAA",
        "ROSS": "#6A5ACD", "CHANDLER": "#B266FF", "PHOEBE": "#FF69B4"
    }
    if s_upper in FRIENDS_COLORS:
        return FRIENDS_COLORS[s_upper]

    # Se já gerámos uma cor para este speaker antes, devolve a mesma
    if speaker in cores_personalizadas:
        return cores_personalizadas[speaker]

    # Gera uma cor nova baseada no NOME. 
    random.seed(speaker) 
    cor_aleatoria = "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
    
    cores_personalizadas[speaker] = cor_aleatoria
    return cor_aleatoria

speakers = []
speaker_cluster_counts = []
fsc_speaker = None
avg_EOD = None
avg_SOD = None
df_final = []
density = []
clusters_speaker = []
diretoria = ""
cluster_id_speaker = []
dot_code = ""
result = None

import glob

# 1. Procura todas as pastas dentro de ./Resultados que começam pelo nome do ficheiro
# ou simplesmente a pasta mais recente criada em Resultados
pastas_resultados = glob.glob(os.path.join("./Resultados", "*"))
if pastas_resultados:
    # Ordena as pastas por data de modificação (a mais recente primeiro)
    subpasta_recente = max(pastas_resultados, key=os.path.getmtime)
    
    # 2. Define o caminho para o ficheiro .dot dentro dessa pasta
    caminho_real = os.path.join(subpasta_recente, "DAs_train_emo_none_NONE_1_graph.dot")
    
    if os.path.exists(caminho_real):
        print(f"Ficheiro detetado automaticamente em: {caminho_real}")
        # Carrega os dados usando o caminho dinâmico
        data = load_data_from_dot(caminho_real)
    else:
        print(f"Erro: O ficheiro .dot não existe em {subpasta_recente}")
else:
    print("Erro: Nenhuma pasta encontrada em ./Resultados")

# INICIAR FLASK 

n_edges = len(result.edges)
total_nodes = len(result.nodes)

n_action_nodes = 0
n_dialogue_nodes = 0

# Percorre os nós do grafo para contar tipos
for node, attrs in result.nodes(data=True):
    if attrs.get('shape') == 'box' or "Ação" in str(node) or "ACTION" in str(node):
        n_action_nodes += 1
    else:
        n_dialogue_nodes += 1

# Garante a soma correta
if n_action_nodes + n_dialogue_nodes != total_nodes:
    n_dialogue_nodes = total_nodes - n_action_nodes

print("\n>>> A RECALCULAR MÉTRICAS PARA A BARRA LATERAL...")

utterances_in_dataset = 0
n_dialogues_in_dataset = 0
utterances_per_dialogue_mean = 0
utterances_per_dialogue_std = 0

# 1. Obter os dados brutos
raw_data = locals().get('df_final')

if isinstance(raw_data, list):
    # Se for lista, converte de volta para DataFrame
    dados_alvo = pd.DataFrame(raw_data)
elif isinstance(raw_data, pd.DataFrame):
    dados_alvo = raw_data
else:
    dados_alvo = pd.DataFrame()

# 3. Calcular Métricas
actions_in_dataset = 0

if not dados_alvo.empty:
    # Contar Falas e Ações
    if 'Tipo' in dados_alvo.columns:
        # Converter para string/lower para garantir que apanha "Ação", "ação", "acao"
        tipo_normalizado = dados_alvo['Tipo'].astype(str).str.lower().str.strip()
        
        utterances_in_dataset = len(dados_alvo[tipo_normalizado == 'fala'])
        actions_in_dataset = len(dados_alvo[tipo_normalizado == 'ação'])
    else:
        utterances_in_dataset = len(dados_alvo)
        actions_in_dataset = 0

    # Contar Diálogos
    if 'dialogue_id' in dados_alvo.columns:
        n_dialogues_in_dataset = dados_alvo['dialogue_id'].nunique()
        
        stats = dados_alvo.groupby('dialogue_id').size()
        utterances_per_dialogue_mean = stats.mean()
        utterances_per_dialogue_std = stats.std()
        if pd.isna(utterances_per_dialogue_std): utterances_per_dialogue_std = 0.0
    else:
        n_dialogues_in_dataset = 1
        utterances_per_dialogue_mean = utterances_in_dataset

    print(f"DADOS: {utterances_in_dataset} Falas | {actions_in_dataset} Ações | {n_dialogues_in_dataset} Diálogos")
else:
    print("Não há dados válidos em df_final.")

# --- Calcular Média de Sentimento (FSC) ---
fsc_mean = 0.0
if 'fsc_speaker' in globals() and isinstance(fsc_speaker, dict):
    vals = [v for v in fsc_speaker.values() if isinstance(v, (int, float)) and v > 0]
    if vals:
        fsc_mean = np.mean(vals)

sod_val = globals().get('avg_SOD')
eod_val = globals().get('avg_EOD')

# Se for None, transforma em 0 para não quebrar o round()
if sod_val is None: sod_val = 0.0
if eod_val is None: eod_val = 0.0

tem_sentimento = True
fsc_atual = fsc_mean

# Se o FSC for zero e os valores de SOD/EOD forem neutros ou nulos, escondemos
if fsc_atual == 0.0 and (sod_val in [0.0, 0.5]) and (eod_val in [0.0, 0.5]):
    tem_sentimento = False

# Definição dos Textos dos Tooltips
tips = {
    "utt_data": "Total number of utterances in the dataset.",
    "act_data": "Total number of non-verbal events (actions) in the dataset.",
    "dial_data": "Total number of dialogues in the dataset.",
    "utt_per_dial": "Average number of utterances per dialogue.",
    "n_dial_states": "Total number of discovered dialogue states (Ellipses).",
    "n_act_states": "Total number of discovered action states (Boxes).",
    "sent_coh": "Average standard deviation of sentiment across all states.",
    "n_trans": "Total number of transitions between states.",
    "sent_ini": "Average sentiment of states connected to SOD.",
    "sent_fin": "Average sentiment of states connected to EOD.",
    "sent_var": "Sentiment variation between SOD and EOD.",
    "density": "Graph density, an indicator of complexity.",
    "coverage": "Proportion of covered transitions in unseen dialogues.",
    "ff1": "Harmonic mean of normalized complexity and FuDGE score."
}

# --- Sentimento ---
if tem_sentimento:
    sod_visual = round(float(sod_val), 2)
    eod_visual = round(float(eod_val), 2)
    sent_variation = eod_visual - sod_visual
    
    str_fsc = f'<span class="metric-help" title="{tips["sent_coh"]}">Sentiment Cohesion</span>: {fsc_atual:.2f}<br><br>'
    str_sent_flow = (
        f'<span class="metric-help" title="{tips["sent_ini"]}">Initial Sentiment</span>: {sod_visual:.2f}<br>'
        f'<span class="metric-help" title="{tips["sent_fin"]}">Final Sentiment</span>: {eod_visual:.2f}<br>'
        f'<span class="metric-help" title="{tips["sent_var"]}">Sentiment Variation</span>: {sent_variation:+.2f}<br>'
    )
else:
    str_fsc = "<br>"
    str_sent_flow = ""

# --- Transition Coverage e FF1 ---
val_acc = globals().get('accuracy')
if val_acc is None: val_acc = globals().get('acc')

str_coverage = ""
if isinstance(val_acc, (int, float)) and val_acc > 0:
    str_coverage = f'<span class="metric-help" title="{tips["coverage"]}">Transition Coverage</span>: {val_acc:.2f}<br>'

val_f1 = globals().get('ff1')
str_f1 = ""
if isinstance(val_f1, (int, float)) and val_f1 > 0:
    str_f1 = f'<span class="metric-help" title="{tips["ff1"]}">Flow F1-Score</span>: {val_f1:.2f}<br>'

# --- Ações ---
str_actions_dataset = ""
if actions_in_dataset > 0:
    str_actions_dataset = f'<span class="metric-help" title="{tips["act_data"]}">Actions in Dataset</span>: {actions_in_dataset}<br>'

str_actions_graph = ""
if n_action_nodes > 0:
    str_actions_graph = f'<span class="metric-help" title="{tips["n_act_states"]}">Number of Action States</span>: <b id="count-action-states">{n_action_nodes}</b><br>'

metricas_html = f"""
<div id="metrics-container" style="padding-left: 10px; font-size: 14px; line-height: 1.6;">
    
    <h3 style="margin-bottom: 5px;">Dialogue Metrics</h3>
    <span class="metric-help" title="{tips["utt_data"]}">Utterances in Dataset</span>: {utterances_in_dataset}<br>
    {str_actions_dataset}
    <span class="metric-help" title="{tips["dial_data"]}">Dialogues in Dataset</span>: {n_dialogues_in_dataset}<br>
    <span class="metric-help" title="{tips["utt_per_dial"]}">Utterances per Dialogue</span>: {utterances_per_dialogue_mean:.1f}&nbsp;&plusmn;&nbsp;{utterances_per_dialogue_std:.1f}<br><br>

    <h3 style="margin-bottom: 5px;">Clustering Metrics</h3>
    <span class="metric-help" title="{tips["n_dial_states"]}">Number of Dialogue States</span>: <b id="count-states">{n_dialogue_nodes}</b><br>
    {str_actions_graph}
    {str_fsc}

    <h3 style="margin-bottom: 5px;">Flow Metrics</h3>
    <span class="metric-help" title="{tips["n_trans"]}">Number of Transitions</span>: <b id="count-edges">{n_edges}</b><br>
    {str_sent_flow}
    <span class="metric-help" title="{tips["density"]}">Flow Density</span>: {density:.2f}<br>
    
    {str_coverage}
    {str_f1}
</div>
"""

#clusters_speaker = None
thresholds = np.arange(0.0, 1.01, 0.1)
threshold_values = [round(t, 2) for t in thresholds]
threshold_data = {
    f"{t:.2f}": {
        "shapes": "",
        "annotations": "" 
    }
    for i, t in enumerate(thresholds)
}

import json
import numpy as np

# Função que converte qualquer número "estranho" do Pandas/Numpy para um número normal
def limpa_numpy(obj):
    if isinstance(obj, dict):
        # O JSON precisa que as chaves dos dicionários sejam strings (texto)
        return {str(k): limpa_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [limpa_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return limpa_numpy(obj.tolist())
    else:
        return obj

labels_speaker_limpo = limpa_numpy(globals().get('labels_speaker', {}))
clusters_speaker_limpo = limpa_numpy(globals().get('clusters_speaker', {}))

labels_speaker_json = json.dumps(labels_speaker_limpo)
clusters_speaker_json = json.dumps(clusters_speaker_limpo)

threshold_json = json.dumps(threshold_data)
dot_code_js = json.dumps(dot_code)
# ==============================================================================

checkboxes_html = """
<div id="speakers-list" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px;">
    <div>
        <input type="radio" id="todos" name="speaker" value="__all__" checked>
        <label for="todos" style="color: white;">Todos</label>
    </div>
"""

# 2. Loop para adicionar os speakers iniciais
for speaker in speakers:
    speaker_id = speaker.replace(" ", "_").lower()
    cor = atribuir_cor(speaker)
    
    checkboxes_html += f"""
    <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <input type="radio" id="{speaker_id}" name="speaker" value="{speaker}" style="margin-right: 8px;">
        <label for="{speaker_id}" style="color: white; cursor: pointer;">{speaker}</label>
    </div>
    """

checkboxes_html += "</div>"

legenda_items_str = ""
sorted_speakers_legend = sorted(list(set(speakers)))

for spk in sorted_speakers_legend:
    s_upper = spk.upper().strip()
    if s_upper in ["SOD", "EOD"] or "ACAO" in s_upper or "ACTION" in s_upper:
        continue

    cor_hex = atribuir_cor(spk)
    
    legenda_items_str += f"""
    <div style="display: flex; align-items: center; margin-bottom: 6px;">
        <div style="
            width: 16px; 
            height: 16px; 
            background-color: {cor_hex}; 
            margin-right: 10px; 
            border-radius: 4px; 
            border: 1px solid rgba(255,255,255,0.3);
            flex-shrink: 0;
        "></div>
        <span style="color: white; font-size: 13px; word-break: break-word;">{spk}</span>
    </div>
    """

# 2. HTML Final
legenda_html = f"""
<div class="box" style="margin-top: 20px;">
    <div class="collapsible collapsed">
        <h2 class="collapsible-header" style="justify-content: space-between; font-size: 16px; margin-top: 5px; margin-bottom: 5px;">
            Caption
            <span class="arrow"></span>
        </h2>
        <div class="collapsible-content">
            
            <div style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1);">
                <h4 style="margin: 10px 0 8px 0; color: #ccc; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Node Type</h4>
                
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <div style="
                        width: 14px; 
                        height: 14px; 
                        background-color: transparent; 
                        margin-right: 10px; 
                        border-radius: 50%; 
                        border: 2px solid #ccc; 
                        flex-shrink: 0;
                    "></div>
                    <span style="color: white; font-size: 13px;">Dialogue State</span>
                </div>

                <div style="display: flex; align-items: center; margin-bottom: 8px; margin-left: 24px;"> <span style="color: #666; margin-right: 6px; font-size: 10px;">↳</span> 
                    
                    <div style="
                        width: 12px; 
                        height: 12px; 
                        background-color: #FFD700; 
                        margin-right: 8px; 
                        border-radius: 50%; 
                        border: 1px solid rgba(255,255,255,0.5);
                        flex-shrink: 0;
                    "></div>
                    <span style="color: #ddd; font-size: 12px;">Special Nodes (Start/End)</span>
                </div>

                <div style="display: flex; align-items: center; margin-bottom: 6px; margin-top: 10px;">
                    <div style="
                        width: 14px; 
                        height: 14px; 
                        background-color: transparent; 
                        margin-right: 10px; 
                        border-radius: 2px; 
                        border: 2px solid #ccc; 
                        flex-shrink: 0;
                    "></div>
                    <span style="color: white; font-size: 13px;">Action State</span>
                </div>
            </div>

            <h4 style="margin: 0 0 8px 0; color: #ccc; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Speakers</h4>
            <div id="legend-container" style="max-height: 250px; overflow-y: auto; padding-right: 5px;">
                {legenda_items_str}
            </div>
            
        </div>
    </div>
</div>
"""

nome_imagem_heatmap = f"Heatmap_Transitions_{filename.split('.')[0]}.png"

html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8" />
    <link href="https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap" rel="stylesheet">

    <script>
        var Module = {{
            TOTAL_MEMORY: 268435456
        }};
    </script>

    <script src="https://unpkg.com/viz.js@2.1.2/viz.js"></script>
    <script src="https://unpkg.com/viz.js@2.1.2/full.render.js"></script>
    <script src="https://cdn.plot.ly/plotly-wlatest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    <style>
        body {{
            margin: 0;
            font-family: 'Lato', sans-serif;
        }}
        .metric-help {{
            cursor: help;           /* Cursor vira um ponto de interrogação */
            text-decoration: none;  /* Remove qualquer sublinhado */
            border-bottom: none;    /* Remove o pontilhado */
            transition: all 0.2s ease;
            opacity: 0.9;
        }}

        /* ESTILOS PARA O MODAL DO HEATMAP */
        /* ESTILOS PARA O MODAL DO HEATMAP */
        .modal {{
            display: none; 
            position: fixed; 
            z-index: 3000;
            padding-top: 50px; 
            left: 0; top: 0;
            width: 100%; height: 100%; 
            background-color: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(5px);
        }}
        .modal-content {{
            margin: auto;
            display: block;
            max-width: 80%;
            max-height: 85vh;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            animation: zoomIn 0.3s ease;
        }}
        .close-modal {{
            position: absolute;
            top: 20px; right: 40px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.2s;
        }}
        .close-modal:hover {{ color: #87CEEB; }}
        @keyframes zoomIn {{ from {{transform: scale(0.9); opacity: 0;}} to {{transform: scale(1); opacity: 1;}} }}
        
        .metric-help:hover {{
            color: #87CEEB;         /* Fica azul claro */
            text-shadow: 0px 0px 1px rgba(135, 206, 235, 0.5); /* Pequeno brilho */
            opacity: 1;
        }}
        .custom-tooltip {{
            position: absolute;
            background-color: rgba(30, 41, 59, 0.95);
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
            max-width: 350px;
            
            pointer-events: auto; /* Permite clicar e fazer scroll! */
            
            z-index: 10000;
            display: none;
            box-shadow: 0 8px 20px rgba(0,0,0,0.4); 
            border: 1px solid rgba(255,255,255,0.1);
            font-family: 'Lato', sans-serif;
            line-height: 1.5;
            
            /* Limites e Scroll */
            max-height: 60vh;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: rgba(255,255,255,0.3) transparent;
        }}
        #sidebar {{
            height: 100%;
            width: 0; 
            position: fixed;
            z-index: 999;
            top: 0;
            left: 0;
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            overflow-x: hidden;
            transition: 0.3s;
            padding-top: 60px;
            color: #e2e8f0; 
            box-shadow: 4px 0 15px rgba(0,0,0,0.3);
            font-family: 'Lato', sans-serif;
        }}
        #selection-rectangle {{
            position: absolute;
            border: 2px dashed #007bff;
            background-color: rgba(0, 123, 255, 0.2);
            pointer-events: none;
            display: none;
            z-index: 9999;
            }}
        #openSidebar {{
            position: fixed;
            top: 20px;
            left: 0;
            z-index: 1000;
            background-color: #002366;
            color: white;
            padding: 10px 15px;
            cursor: pointer;
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }}
        #sidebarContent {{
            padding: 15px;
            overflow-y: auto;
            max-height: 90%;
        }}
        .closebtn {{
            position: absolute;
            top: 10px;
            right: 20px;
            font-size: 30px;
            cursor: pointer;
        }}
        #grafico {{
            margin-left: 0;
            transition: margin-left 0.3s;
            padding: 10px;
            overflow: hidden;
            
        }}
        h1#main-title {{
            position : sticky;
            width: 100%;
            margin-left: -15px;
            background: white;
            z-index: 1;
            color: #1e293b;
            text-align: center;
            font-weight: 700;
            margin-top: -10px;
            margin-bottom: 20px;
            padding-top: 30px;
            padding-bottom: 15px;
            padding-right: 35px;
            font-size: 3rem;
            font-family: 'Lato', sans-serif;
        }}
        h2, h3 {{
            font-weight: bold;
            color: #fff;
        }}
        input[type=range] {{
            width: 90%;
        }}
        .box {{
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 3px 3px 8px rgba(0,0,0,0.8);
            font-size: 14px;
            line-height: 1.6;
            /*background-color: #fff;*/
        }}
        svg {{
            width: 100%;
            height: 80vh;
        }}
        .selected {{
            stroke: #87CEEB !important;
            stroke-width: 3px !important;
        }}
        .hidden {{
            display: none !important;
        }}
        .faded {{
            opacity: 0.2 !important;
        }}
        /* Slider customizado */
        input[type=range] {{
            -webkit-appearance: none;
            width: 90%;
            height: 9px;
            background: #ddd;
            border-radius: 10px;
            outline: none;
            cursor: pointer;
        }}
        input[type=range]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            background: #87CEEB;
            cursor: pointer;
            border-radius: 10px;
            border: 2px solid #999;
            margin-top: -3px;
            transition: background 0.3s ease;
            border-color: #0055aa;
        }}
        input[type=range]:hover::-webkit-slider-thumb {{
            background: ##87CEEB;
            border-color: #0055aa;
        }}
        input[type=range]::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            background: #87CEEB;
            border-radius: 10px;
            border: 2px solid #999;
            cursor: pointer;
        }}
        /* Radio buttons */
        input[type="radio"] {{
            accent-color: #87CEEB;
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        g.node title {{
            pointer-events: none;
        }}
        #graph-controls button {{
            background-color: rgb(0 123 255 / 11%);;
            border: none;
            color: white;
            padding: 0px;
            margin-right: 5px;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }}
        #graph-controls button:hover {{
            background-color: #0055aa ;
        }}
        #graph-wrapper {{
            position: relative;
        }}
        box1 {{
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 3px 3px 8px rgba(0,0,0,0.8);
            font-size: 14px;
            line-height: 1.6;
            /*background-color: #fff;*/
            display: flex;
            justify-content: center;
            gap: 5px;
            margin-bottom: 40%;
        }}
        .box1 button {{
            width: 32.5px;
            height: 32.5px;
            background-color: #87CEEB
            color: white; 
            font-size: 20px;
            border: none;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.1s;
        }}
        .box1 button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }}
        .box1 button:active {{
            transform: scale(0.95);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        #searchInput:focus {{
            border-color: #87CEEB;
            outline: none;
        }}
        .collapsible-header {{
            cursor: pointer;
            display: inline-flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 8px;
            user-select: none;
            margin-bottom: 10px;
            gap: 100px;
            margin-top: 1px;
            margin-bottom: 1px;
        }}
        .collapsible-header1 {{
            cursor: pointer;
            display: inline-flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 8px;
            user-select: none;
            margin-bottom: 10px;
            gap: 100px;
            margin-top: 1px;
            margin-bottom: 1px;
        }}
        .collapsible-content {{
            display: block;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}

        .collapsible.collapsed .collapsible-content {{
            display: none;
        }}

        .collapsible .arrow {{
            width: 12px;
            height: 12px;
            display: inline-block;
            border-left: 2px solid white;
            border-bottom: 2px solid white;
            transform: rotate(45deg);
            transition: transform 0.3s ease;
            margin-left: 8px;
        }}
        .collapsible.collapsed .arrow {{
            transform: rotate(-45deg); 
        }}
        .collapsible-header {{
            display: inline-flex;
            align-items: center;
            gap: 100px; 
        }}
        .collapsible-header1 {{
            display: inline-flex;
            align-items: center;
            gap: 117px; 
        }}
        .collapsible-header .arrow {{
            margin-left: 0; 
            font-size: 0.8em; 
        }}
        .floating-controls {{
            position: fixed;
            bottom: 25px;
            right: 25px;
            z-index: 1000;
            
            display: flex;
            flex-direction: row;
            align-items: center;
            gap: 8px;
            
            background: rgba(15, 23, 42, 0.85); 
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            
            padding: 8px 14px;
            border-radius: 50px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.15);
        }}

        .floating-controls button {{
            all: unset;
            width: 36px;  
            height: 36px;
            border-radius: 50%;
            background: transparent;
            color: #94a3b8;
            font-size: 15px;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
        }}

        .floating-controls button:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: #87CEEB;
            transform: translateY(-3px) scale(1.05);
        }}

        .floating-controls button.active {{
            background: rgba(135, 206, 235, 0.15);
            color: #87CEEB;
            box-shadow: inset 0 0 0 1.5px rgba(135, 206, 235, 0.5);
        }}
</style>
</head>

<body>
<div id="openSidebar" onclick="openSidebar()">☰</div>
<div id="sidebar">
    <span class="closebtn" onclick="closeSidebar()">×</span>
    
    <div id="sidebarContent">
        <div class="box">
            <h2 style="margin-top:-1px">Upload File</h2>
            <input type="file" id="dotFileInput" accept=".dot" style="display: none;">
            <label 
                for="dotFileInput" 
                id="fileLabel"
                style="
                    display: flex;
                    flex-direction: row;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    padding: 10px 16px;
                    background-color: rgb(0 123 255 / 11%);
                    border-radius: 6px;
                    cursor: pointer;
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                    transition: background-color 0.3s ease;
                    height: 40px;
                    width: 100%;
                    box-sizing: border-box;
                    box-shadow: 3px 3px 8px rgba(0,0,0,0.8);
                    overflow: hidden;
                "
                onmouseover="this.style.backgroundColor='#0055aa'"
                onmouseout="this.style.backgroundColor='#002366'"
            >
                <svg xmlns="http://www.w3.org/2000/svg" fill="white" viewBox="0 0 24 24" width="16" height="16" style="flex-shrink: 0; width: 15%;" >
                    <path d="M5 20h14v-2H5v2zm7-14l5 5h-3v4h-4v-4H7l5-5z"/>
                </svg>
                <span id="fileLabelText" 
                    style="
                        display: inline-block;
                        max-width: calc(100% - 30px);
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    "
                >Choose file</span>
            </label>
        </div>
        <div class="box">
        <div class="collapsible collapsed">
            <h2 class="collapsible-header">Parameters <span class="arrow"></span></h2>
            <div class="collapsible-content">
                <h3>Search Nodes</h3>
                <input type="text" id="searchInput" placeholder="Enter a word..." style="width: 90%; padding: 5px; border: 2px solid #87CEEB; border-radius: 10px;">

                <h3>Speakers</h3>
                {checkboxes_html}

                {legenda_html}

                <h3>Threshold</h3>
                <input type="range" id="thresholdSlider" min="0" max="100" value="0" step="1" oninput="updateThreshold(this.value)">
                <div style="color: white; margin-top: 5px;">Value: <span id="thresholdLabel">0.00</span></div>
            </div>
        </div>
        </div>
        <div class="box">
            <div class="collapsible collapsed">
            <h2 class="collapsible-header1">Metrics <span class="arrow"></span></h2>
            <div class="collapsible-content">
                {metricas_html}
    </div>
</div>
        </div>
    </div>
</div>

<div id="grafico">
    <h1 id="main-title">FlowDisco</h1>
    <p id="file-subtitle" style="text-align: center; color: #002366; font-family: 'Lato', sans-serif; margin-top: -23px; margin-bottom: 10px; margin-left: -10px;">Dialogue Flow Discovery for: {filename}</p>
    <div id="graph-wrapper" style="overflow: auto; width: 100%;">
        <div id="graph-container"></div>
    </div>
    <div class="floating-controls">
        <button id="btn-reset" title="Home"><i class="fas fa-home"></i></button>
        <button id="btn-zoom-in" title="Zoom In"><i class="fas fa-plus"></i></button>
        <button id="btn-zoom-out" title="Zoom Out"><i class="fas fa-minus"></i></button>
        <button id="btn-pan" title="Move"><i class="fas fa-hand-paper"></i></button>
        <button id="btn-zoom-box" title="Zoom Box"><i class="fas fa-object-group"></i></button>
        <button id="btn-screenshot" title="Print"><i class="fas fa-camera"></i></button>
        <button id="btn-heatmap" title="View Heatmap Matrix"><i class="fas fa-th"></i></button>
    </div>
</div>

<div id="heatmapModal" class="modal">
    <span class="close-modal" id="closeHeatmap">&times;</span>
    <img class="modal-content" id="imgHeatmap" src="{nome_imagem_heatmap}" alt="Transition Heatmap">
</div>

<div id="tooltip" class="custom-tooltip"></div>


<script>
    // Variáveis Globais
    const dotCode = {dot_code_js};
    let dictLabels = {labels_speaker_json};
    let dictFalas = {clusters_speaker_json};
    const container = document.getElementById("graph-container");
    const slider = document.getElementById("thresholdSlider");
    const sliderValue = document.getElementById("thresholdLabel");
    const viz = new Viz({{
        Module: {{
            TOTAL_MEMORY: 268435456
        }}
    }});
    
    let currentGraphElement = null;
    let currentSpeaker = "__all__";
    let fixedTooltip = false;
    
    function updateUploadedSentiment(dotString) {{
        const sodRegex = /SOD\s*->.*?sentiment="([-+]?[0-9]*\.?[0-9]+)"/g;
        const eodRegex = /->\s*(?:"?EOD"?|EOD\s*\(.*?\)).*?sentiment="([-+]?[0-9]*\.?[0-9]+)"/g;

        let sodValues = [];
        let eodValues = [];
        let match;

        while ((match = sodRegex.exec(dotString)) !== null) {{
            const val = parseFloat(match[1]);
            if (!isNaN(val)) sodValues.push(val);
        }}

        while ((match = eodRegex.exec(dotString)) !== null) {{
            const val = parseFloat(match[1]);
            if (!isNaN(val)) eodValues.push(val);
        }}

        if (sodValues.length > 0 && eodValues.length > 0) {{
            const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
            const avgSod = avg(sodValues);
            const avgEod = avg(eodValues);
            const diff = avgEod - avgSod;
            const diffSign = diff > 0 ? "+" : "";
            
            const sentimentHtml = `
                Initial Sentiment: ${{avgSod.toFixed(2)}}<br>
                Final Sentiment: ${{avgEod.toFixed(2)}}<br>
                Sentiment Variation: ${{diffSign}}${{diff.toFixed(2)}}
            `;

            const slot = document.getElementById("sentiment-slot");
            if (slot) {{
                slot.innerHTML = sentimentHtml;
            }}
        }}
    }}

    const highlightStyle = document.createElement('style');
    highlightStyle.innerHTML = `
        .graph-dimmed {{ 
            opacity: 0.1 !important; 
            transition: opacity 0.3s ease; 
        }}
        .graph-highlight {{ 
            opacity: 1 !important; 
            stroke-width: 3px !important; 
            transition: opacity 0.3s ease; 
        }}
        .graph-highlight text {{ 
            font-weight: bold !important; 
            opacity: 1 !important; 
        }}
    `;
    document.head.appendChild(highlightStyle);
    // Controlo de Câmara (Zoom/Pan)
    const btnReset = document.getElementById("btn-reset");
    const btnZoomIn = document.getElementById("btn-zoom-in");
    const btnZoomOut = document.getElementById("btn-zoom-out");
    const btnScreenshot = document.getElementById("btn-screenshot");
    const btnPan = document.getElementById("btn-pan");
    const btnZoomBox = document.getElementById("btn-zoom-box");
    const graphWrapper = document.getElementById("graph-wrapper");
    
    let isPanning = false;
    let isDragging = false;
    let startX = 0, startY = 0;
    let translateX = 0, translateY = 0;
    let zoomLevel = 1;
    let isZoomBoxActive = false;
    let selectionRect = null;

    function updateLegendFromGraph(svgElement) {{
        const legContainer = document.getElementById("legend-container");
        const captionContent = document.querySelector("#legend-container").parentElement;
        if (!captionContent) return;
        
        const speakerColors = {{}}; 

        svgElement.querySelectorAll("g.node").forEach(node => {{
            const fullText = node.getAttribute("data-full-name");
            if (!fullText) return;

            let speakerName = fullText;
            if (fullText.includes("->")) {{
                speakerName = fullText.split("->")[0];
            }} else if (fullText.includes(" - ")) {{
                speakerName = fullText.split(" - ")[0];
            }}
            speakerName = speakerName.split("(")[0].trim();

            const upperName = speakerName.toUpperCase();
            if (upperName.includes("ACAO") || upperName.includes("ACTION")) return;
            if (upperName === "SOD" || upperName === "EOD") return;
            
            if (speakerColors[speakerName]) return;

            let color = "#ccc"; 
            const shape = node.querySelector("ellipse, polygon, path, circle");
            if (shape) {{
                color = shape.getAttribute("stroke");
                if (!color || color === "black" || color === "none") {{
                     color = shape.getAttribute("fill");
                }}
                if (!color && shape.getAttribute("style")) {{
                     const match = shape.getAttribute("style").match(/stroke:\s*([^;]+)/);
                     if (match) color = match[1];
                }}
            }}
            
            if (color && color !== "none") {{
                speakerColors[speakerName] = color;
            }}
        }});

        const nodeTypesHtml = `
            <div style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1);">
                <h4 style="margin: 10px 0 8px 0; color: #ccc; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Node Type</h4>
                
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <div style="width: 14px; height: 14px; background-color: transparent; margin-right: 10px; border-radius: 50%; border: 2px solid #ccc; flex-shrink: 0;"></div>
                    <span style="color: white; font-size: 13px;">Dialogue State</span>
                </div>
                
                <div style="display: flex; align-items: center; margin-bottom: 8px; margin-left: 24px;">
                    <span style="color: #666; margin-right: 6px; font-size: 10px;">↳</span> 
                    <div style="width: 12px; height: 12px; background-color: #FFD700; margin-right: 8px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.5); flex-shrink: 0;"></div>
                    <span style="color: #ddd; font-size: 12px;">Special Nodes (Start/End)</span>
                </div>

                <div style="display: flex; align-items: center; margin-bottom: 6px; margin-top: 10px;">
                    <div style="width: 14px; height: 14px; background-color: transparent; margin-right: 10px; border-radius: 2px; border: 2px solid #ccc; flex-shrink: 0;"></div>
                    <span style="color: white; font-size: 13px;">Action State</span>
                </div>
            </div>
            <h4 style="margin: 0 0 8px 0; color: #ccc; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Speakers</h4>
            <div id="legend-container" style="max-height: 250px; overflow-y: auto; padding-right: 5px;"></div>
        `;

        captionContent.innerHTML = nodeTypesHtml;

        // Preencher Speakers
        const containerSpeakers = captionContent.querySelector("#legend-container");
        const sortedSpeakers = Object.keys(speakerColors).sort();
        
        let speakersHtml = "";
        sortedSpeakers.forEach(spk => {{
            const cor = speakerColors[spk];
            speakersHtml += `
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <div style="width: 16px; height: 16px; background-color: ${{cor}}; margin-right: 10px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.3); flex-shrink: 0;"></div>
                <span style="color: white; font-size: 13px; word-break: break-word;">${{spk}}</span>
            </div>`;
        }});
        
        containerSpeakers.innerHTML = speakersHtml;
    }}

    function updateSpeakersFromGraph(svgElement) {{
        const listContainer = document.getElementById("speakers-list");
        if (!listContainer) return;

        const foundSpeakers = new Set();
        
        svgElement.querySelectorAll("g.node").forEach(node => {{
            const text = node.getAttribute("data-full-name");
            
            if (text) {{
                let cleanName = text;

                // 1. Se tiver seta "->", corta pela seta
                if (text.includes("->")) {{
                    cleanName = text.split("->")[0];
                }}
                // 2. Se tiver traço COM ESPAÇOS (" - "), corta (ex: "PSD - Aplausos")
                // O "CDS-PP" não tem espaços, por isso fica inteiro!
                else if (text.includes(" - ")) {{
                    cleanName = text.split(" - ")[0];
                }}

                cleanName = cleanName.split("(")[0].trim();

                // Adiciona à lista se não for SOD nem EOD
                if (cleanName !== "SOD" && cleanName !== "EOD") {{
                    foundSpeakers.add(cleanName);
                }}
            }}
        }});

        // Limpar a lista antiga
        listContainer.innerHTML = "";

        // Adicionar botão "All"
        const labelAll = document.createElement("label");
        labelAll.innerHTML = '<input type="radio" name="speaker" value="__all__" checked> All';
        labelAll.style.display = "block";
        labelAll.querySelector("input").addEventListener("change", function() {{
            currentSpeaker = "__all__";
            const sld = document.getElementById("thresholdSlider");
            applyFilters(sld ? parseFloat(sld.value)/100 : 0, currentSpeaker);
        }});
        listContainer.appendChild(labelAll);

        // Adicionar oradores encontrados (ordenados)
        Array.from(foundSpeakers).sort().forEach(spk => {{
            const label = document.createElement("label");
            label.innerHTML = '<input type="radio" name="speaker" value="' + spk + '"> ' + spk;
            label.style.display = "block";
            
            label.querySelector("input").addEventListener("change", function() {{
                currentSpeaker = spk; 
                const sld = document.getElementById("thresholdSlider");
                applyFilters(sld ? parseFloat(sld.value)/100 : 0, currentSpeaker);
            }});
            listContainer.appendChild(label);
        }});
    }}

    // HIGHLIGHT DE VIZINHOS
    function enableNeighborHighlight(svgElement) {{
        if (!document.getElementById("highlight-style")) {{
            const highlightStyle = document.createElement('style');
            highlightStyle.id = "highlight-style";
            highlightStyle.innerHTML = `
                .graph-dimmed {{ opacity: 0.1 !important; transition: opacity 0.3s ease; }}
                .graph-highlight {{ opacity: 1 !important; stroke-width: 3px !important; transition: opacity 0.3s ease; }}
                .graph-highlight text {{ font-weight: bold !important; opacity: 1 !important; }}
            `;
            document.head.appendChild(highlightStyle);
        }}

        const nodes = svgElement.querySelectorAll("g.node");
        const edges = svgElement.querySelectorAll("g.edge");

        nodes.forEach(node => {{
            node.addEventListener("mouseenter", () => {{
                if (typeof fixedTooltip !== 'undefined' && fixedTooltip) return;
                const hoveredName = node.getAttribute("data-full-name");
                if (!hoveredName) return;

                nodes.forEach(n => n.classList.add("graph-dimmed"));
                edges.forEach(e => e.classList.add("graph-dimmed"));

                node.classList.remove("graph-dimmed");
                node.classList.add("graph-highlight");

                const neighborNames = new Set();
                
                edges.forEach(edge => {{
                    const source = edge.getAttribute("data-source-full");
                    const target = edge.getAttribute("data-target-full");

                    if (source === hoveredName || target === hoveredName) {{
                        edge.classList.remove("graph-dimmed");
                        edge.classList.add("graph-highlight");
                        if (source === hoveredName) neighborNames.add(target);
                        if (target === hoveredName) neighborNames.add(source);
                    }}
                }});

                nodes.forEach(n => {{
                    const name = n.getAttribute("data-full-name");
                    if (neighborNames.has(name)) {{
                        n.classList.remove("graph-dimmed");
                        n.classList.add("graph-highlight");
                    }}
                }});
            }});

            node.addEventListener("mouseleave", () => {{
                if (typeof fixedTooltip !== 'undefined' && fixedTooltip) return;
                nodes.forEach(n => n.classList.remove("graph-dimmed", "graph-highlight"));
                edges.forEach(e => e.classList.remove("graph-dimmed", "graph-highlight"));
            }});
        }});
    }}


    // =========================================================
    // 2. ARRASTAR NÓS E SETAS
    // =========================================================
    function makeNodesDraggable(svgElement) {{
        let selectedNode = null;
        let dragOffsetX, dragOffsetY;
        const nodeTransforms = new Map();

        function getNodeCenter(node) {{
            const bbox = node.getBBox();
            const transform = nodeTransforms.get(node) || {{ x: 0, y: 0 }};
            return {{
                x: bbox.x + bbox.width / 2 + transform.x,
                y: bbox.y + bbox.height / 2 + transform.y
            }};
        }}

        function updateConnectedEdges(node) {{
            const nodeName = node.getAttribute("data-full-name");
            if (!nodeName) return;

            svgElement.querySelectorAll("g.edge").forEach(edge => {{
                const sourceName = edge.getAttribute("data-source-full");
                const targetName = edge.getAttribute("data-target-full");

                if (sourceName === nodeName || targetName === nodeName) {{
                    const sourceNode = Array.from(svgElement.querySelectorAll("g.node"))
                        .find(n => n.getAttribute("data-full-name") === sourceName);
                    const targetNode = Array.from(svgElement.querySelectorAll("g.node"))
                        .find(n => n.getAttribute("data-full-name") === targetName);

                    if (sourceNode && targetNode) {{
                        const sourcePos = getNodeCenter(sourceNode);
                        const targetPos = getNodeCenter(targetNode);
                        const path = edge.querySelector("path");
                        const text = edge.querySelector("text");
                        
                        const polygon = edge.querySelector("polygon");
                        if (polygon) polygon.style.display = "none";

                        if (sourceName === targetName) {{
                            const yOffset = 25; 
                            const xSpread = 10;  
                            const loopHeight = 90; 
                            const loopWidth = 50; 

                            if (path) {{
                                path.setAttribute("d", 
                                    // M: Começa ligeiramente à esquerda e acima do centro
                                    `M${{sourcePos.x - xSpread}},${{sourcePos.y - yOffset}} ` +
                                    // C: Curva Bezier (Ponto Controlo 1, Ponto Controlo 2, Destino)
                                    `C${{sourcePos.x - loopWidth}},${{sourcePos.y - loopHeight}} ` + // Vai alto e esquerda
                                    `${{sourcePos.x + loopWidth}},${{sourcePos.y - loopHeight}} ` +  // Vai alto e direita
                                    // Destino: Ligeiramente à direita e acima do centro
                                    `${{sourcePos.x + xSpread}},${{sourcePos.y - yOffset}}`
                                );
                            }}
                            
                            if (text) {{
                                text.removeAttribute("transform");
                                text.removeAttribute("dy");
                                
                                text.setAttribute("x", sourcePos.x);
                                text.setAttribute("y", sourcePos.y - loopHeight + 25);
                                
                                text.setAttribute("text-anchor", "middle");
                                text.setAttribute("dominant-baseline", "middle");
                            }}

                        }} else {{
                            if (path) {{
                                path.setAttribute("d", `M${{sourcePos.x}},${{sourcePos.y}} L${{targetPos.x}},${{targetPos.y}}`);
                            }}

                            if (text) {{
                                text.removeAttribute("transform");
                                text.removeAttribute("dy");

                                const midX = (sourcePos.x + targetPos.x) / 2;
                                const midY = (sourcePos.y + targetPos.y) / 2;
                                
                                text.setAttribute("x", midX);
                                text.setAttribute("y", midY - 10);
                                
                                text.setAttribute("text-anchor", "middle");
                                text.setAttribute("dominant-baseline", "middle");
                            }}
                        }}
                    }}
                }}
            }});
        }}

        // Inicializar transforms
        svgElement.querySelectorAll("g.node").forEach(node => {{
            if (!nodeTransforms.has(node)) {{
                nodeTransforms.set(node, {{ x: 0, y: 0 }});
            }}

            node.addEventListener("mousedown", e => {{
                if ((typeof isPanning !== 'undefined' && isPanning) || 
                    (typeof isZoomBoxActive !== 'undefined' && isZoomBoxActive)) return;
                
                selectedNode = node;
                dragOffsetX = e.clientX;
                dragOffsetY = e.clientY;
                e.preventDefault();
                e.stopPropagation();
            }});
        }});

        document.addEventListener("mousemove", e => {{
            if (!selectedNode) return;
            
            const zoom = typeof zoomLevel !== 'undefined' ? zoomLevel : 1;
            const dx = (e.clientX - dragOffsetX) / zoom;
            const dy = (e.clientY - dragOffsetY) / zoom;
            
            dragOffsetX = e.clientX;
            dragOffsetY = e.clientY;

            const currentTransform = nodeTransforms.get(selectedNode);
            currentTransform.x += dx;
            currentTransform.y += dy;

            selectedNode.setAttribute("transform", `translate(${{currentTransform.x}}, ${{currentTransform.y}})`);
            updateConnectedEdges(selectedNode);
        }});

        document.addEventListener("mouseup", () => {{
            selectedNode = null;
        }});
    }}

    // AJUSTAR ESPESSURA DAS SETAS (PROBABILIDADE)
    function updateEdgeThickness(svgElement) {{
        svgElement.querySelectorAll("g.edge").forEach(edge => {{
            const label = edge.querySelector("text");
            let prob = 0.1;
            if (label) {{
                const textVal = label.textContent.replace(",", ".");
                prob = parseFloat(textVal);
            }}
            if (isNaN(prob)) prob = 0.1;
            const thickness = 1 + (prob * 5); // Ex: 100% de probabilidade = 6px
            const path = edge.querySelector("path");
            if (path) path.style.strokeWidth = thickness + "px";
        }});
    }}

    function initTooltips(svgElement) {{
        const tooltip = document.getElementById("tooltip");
        let hideTimeout = null;

        const showTooltip = () => {{
            if (hideTimeout) clearTimeout(hideTimeout);
            tooltip.style.display = "block";
        }};

        const hideTooltip = () => {{
            hideTimeout = setTimeout(() => {{
                if (!fixedTooltip) tooltip.style.display = "none";
            }}, 300);
        }};

        tooltip.addEventListener("mouseenter", showTooltip);
        tooltip.addEventListener("mouseleave", hideTooltip);
        
        svgElement.querySelectorAll("g.node").forEach(node => {{
            node.style.cursor = "pointer";

            node.addEventListener("mouseenter", (e) => {{
                if (fixedTooltip) return;
                showTooltip();
                
                const fullTitle = node.getAttribute("data-full-name");
                if (!fullTitle) return;

                let speakerName = fullTitle;

                // Se tiver seta "->", corta pela seta (Prioridade Máxima)
                if (fullTitle.includes("->")) {{
                    speakerName = fullTitle.split("->")[0];
                }} 
                // Se tiver " - " (traço COM espaços), corta pelo traço
                // Isto protege o CDS-PP, porque o CDS-PP não tem espaços no meio!
                else if (fullTitle.includes(" - ")) {{
                    speakerName = fullTitle.split(" - ")[0];
                }}
                
                // Limpeza final (tira números entre parênteses e espaços em branco)
                speakerName = speakerName.split("(")[0].trim();
                
                // Cabeçalho básico (SOD/EOD)
                if (speakerName.toUpperCase() === "SOD"){{
                    tooltip.innerHTML = "<div style='font-size: 16px; font-weight: 700;'>Start of Dialogue</div>";
                }} else if (speakerName.toUpperCase() === "EOD"){{
                    tooltip.innerHTML = "<div style='font-size: 16px; font-weight: 700;'>End of Dialogue</div>";
                }} else {{
                    
                    let tooltipHTML = `
                        <div style="font-size: 16px; font-weight: 700; margin-bottom: 2px;">
                            ${{fullTitle}}
                        </div>
                        <hr style="border:0; border-top:1px solid rgba(255,255,255,0.3); margin: 8px 0;">`;

                    let clusterId = undefined;
                    let speakerDicionario = speakerName;

                    let labelLimpa = fullTitle;
                    if (fullTitle.includes("->")) {{
                        labelLimpa = fullTitle.split("->")[1].trim(); 
                        labelLimpa = labelLimpa.replace(/\s*\(\d+\)$/, "").trim(); 
                    }} 
                    // Se for do tipo "PSD - Aplausos", limpar também
                    else if (fullTitle.includes(" - ")) {{
                        labelLimpa = fullTitle.split(" - ")[1].trim();
                        labelLimpa = labelLimpa.replace(/\s*\(\d+\)$/, "").trim();
                    }}

                    // Procura ID no Dicionário Labels
                    // Procura ID no Dicionário Labels (Fuzzy Match Seguro)
                    if (dictLabels[speakerName]) {{
                        // Limpa a label do grafo: tira espaços, traços e põe minúsculas
                        let cleanLabelLimpa = labelLimpa.toLowerCase().replace(/[^a-z0-9]/g, "");
                        
                        for (let id in dictLabels[speakerName]) {{
                            // Limpa a label do dicionário da mesma forma
                            let cleanDictLabel = dictLabels[speakerName][id].toLowerCase().replace(/[^a-z0-9]/g, "");
                            
                            // Fazemos apenas o match exato das strings limpas! 
                            // Isto impede o "Compreendi!" de infetar os outros nós.
                            if (cleanLabelLimpa !== "" && cleanDictLabel === cleanLabelLimpa) {{
                                clusterId = id;
                                break;
                            }}
                        }}
                    }}

                    // Fallback para Ações
                    if (clusterId === undefined && dictLabels["Acao"]) {{
                        for (let id in dictLabels["Acao"]) {{
                            if (fullTitle.includes(dictLabels["Acao"][id].trim())) {{
                                clusterId = id;
                                speakerDicionario = "Acao";
                                break;
                            }}
                        }}
                    }}

                    // Procurar Frases
                    let listaFrases = [];
                    if (clusterId !== undefined && dictFalas[speakerDicionario] && dictFalas[speakerDicionario][clusterId]) {{
                        listaFrases = dictFalas[speakerDicionario][clusterId];
                    }}

                    if (listaFrases.length > 0) {{
                        let frasesUnicas = [...new Set(listaFrases)].sort(() => 0.5 - Math.random());
                        
                        tooltip.dataset.phrases = JSON.stringify(frasesUnicas);
                        tooltip.dataset.currentCount = 5;

                        tooltipHTML += `<ul id="lista-frases-ul" style="margin: 8px 0 0 0; padding-left: 0; list-style-type: none;">`;
                        
                        frasesUnicas.slice(0, 5).forEach(fala => {{
                            let falaCurta = fala.length > 130 ? fala.substring(0, 127) + "..." : fala;
                            tooltipHTML += `
                                <li style="margin-bottom: 6px; display: flex; align-items: flex-start;">
                                    <span style="margin-right: 8px; opacity: 0.6; font-size: 10px;">➤</span>
                                    <span style="font-style: italic; font-weight: 300; opacity: 0.9;">"${{falaCurta}}"</span>
                                </li>`;
                        }});
                        tooltipHTML += `</ul>`;
                        
                        if (frasesUnicas.length > 5) {{
                            tooltipHTML += `
                            <div id="btn-load-more" 
                                 style="text-align: right; font-size: 11px; opacity: 0.7; margin-top: 8px; 
                                        cursor: pointer; color: #87CEEB; font-weight: bold; transition: 0.2s;">
                                + ${{frasesUnicas.length - 5}} other utterances (show more)
                            </div>`;
                        }}

                    }} else {{
                        tooltipHTML += `<span style="font-size: 12px; font-style: italic; color: #aaa;">No utterance available in this cluster.</span>`;
                    }}

                    tooltip.innerHTML = tooltipHTML;

                    // Evento Click Botão
                    const btnLoad = document.getElementById("btn-load-more");
                    if (btnLoad) {{
                        btnLoad.addEventListener("click", (evt) => {{
                            evt.stopPropagation();
                            
                            let todasFrases = JSON.parse(tooltip.dataset.phrases);
                            let count = parseInt(tooltip.dataset.currentCount);
                            let ul = document.getElementById("lista-frases-ul");
                            
                            let proximaLeva = todasFrases.slice(count, count + 5);
                            
                            proximaLeva.forEach(fala => {{
                                let falaCurta = fala.length > 130 ? fala.substring(0, 127) + "..." : fala;
                                ul.insertAdjacentHTML('beforeend', `
                                    <li style="margin-bottom: 6px; display: flex; align-items: flex-start; animation: fadeIn 0.5s;">
                                        <span style="margin-right: 8px; opacity: 0.6; font-size: 10px;">➤</span>
                                        <span style="font-style: italic; font-weight: 300; opacity: 0.9;">"${{falaCurta}}"</span>
                                    </li>
                                `);
                            }});
                            
                            count += 5;
                            tooltip.dataset.currentCount = count;
                            
                            let restantes = todasFrases.length - count;
                            if (restantes > 0) {{
                                btnLoad.innerHTML = `+ ${{restantes}} other lines (show more)`;
                            }} else {{
                                btnLoad.style.display = "none";
                            }}
                        }});
                    }}
                }}

                // Cor da Tooltip
                const shape = node.querySelector("ellipse, polygon, circle, path");
                let fillColor = "#8B0000"; 
                if (shape) {{
                    const stroke = shape.getAttribute("stroke");
                    if (stroke && stroke !== "black") fillColor = stroke;
                    const styleAttr = shape.getAttribute("style");
                    if (styleAttr) {{
                        const match = styleAttr.match(/stroke:\s*(#[0-9a-fA-F]{{3,6}})/);
                        if (match) fillColor = match[1];
                    }}
                }}
                tooltip.style.backgroundColor = fillColor;
            }});

            node.addEventListener("mousemove", (e) => {{
                if (!fixedTooltip) {{
                    const tooltipRect = tooltip.getBoundingClientRect();
                    const tooltipWidth = tooltipRect.width;
                    const tooltipHeight = tooltipRect.height;
                    
                    const windowWidth = window.innerWidth;
                    const windowHeight = window.innerHeight;
                    const scrollY = window.scrollY || window.pageYOffset;
                    const scrollX = window.scrollX || window.pageXOffset;

                    let left = e.pageX + 15;
                    let top = e.pageY + 15;

                    if (e.clientY + tooltipHeight + 20 > windowHeight) {{
                        top = e.pageY - tooltipHeight - 15;
                    }}
                    if (top < scrollY + 10) {{
                        top = scrollY + 10;
                    }}
                    if (e.clientX + tooltipWidth + 20 > windowWidth) {{
                        left = e.pageX - tooltipWidth - 15;
                    }}
                    if (left < scrollX + 10) {{
                        left = scrollX + 10;
                    }}

                    tooltip.style.left = left + "px";
                    tooltip.style.top = top + "px";
                }}
            }});

            node.addEventListener("mouseleave", () => {{
                hideTooltip();
            }});
        }});
        
        document.addEventListener("click", (e) => {{
            if (!e.target.closest("g.node") && !e.target.closest("g.edge") && !e.target.closest("#tooltip")) {{
                fixedTooltip = false;
                tooltip.style.display = "none";
            }}
        }});
    }}

    function applyEdgeFilter(threshold) {{
        if (!currentGraphElement) return;
        const visibleNodes = new Set();

        currentGraphElement.querySelectorAll("g.edge").forEach(edge => {{
            const label = edge.querySelector("text");
            if (!label) return;
            const valueStr = label.textContent;
            const value = parseFloat(valueStr.replace(',', '.'));

            if (!isNaN(value) && value >= threshold) {{
                edge.classList.remove("hidden");
                visibleNodes.add(edge.getAttribute("data-source-full"));
                visibleNodes.add(edge.getAttribute("data-target-full"));
            }} else {{
                edge.classList.add("hidden");
            }}
        }});

        currentGraphElement.querySelectorAll("g.node").forEach(node => {{
            const nodeName = node.getAttribute("data-full-name");
            if (visibleNodes.has(nodeName) || nodeName.toLowerCase() === "sod" || nodeName.toLowerCase() === "eod") {{
                node.classList.remove("hidden");
            }} else {{
                node.classList.add("hidden");
            }}
        }});
    }}

    function applyFilters(threshold, speaker) {{
        if (!currentGraphElement) return;
        
        const speakerTarget = speaker ? speaker.trim() : "__all__";
        const allSpeakers = (speakerTarget === "__all__");
        const nodesWithActiveEdges = new Set();

        function getSpeakerName(fullName) {{
            if (!fullName) return "";
            let name = fullName;
            if (fullName.includes("->")) name = fullName.split("->")[0];
            else if (fullName.includes(" - ")) name = fullName.split(" - ")[0];
            return name.split("(")[0].trim();
        }}

        // Esconder/Mostrar Setas
        currentGraphElement.querySelectorAll("g.edge").forEach(edge => {{
            const label = edge.querySelector("text");
            const value = label ? parseFloat(label.textContent.replace(",", ".")) : NaN;
            
            const sourceFull = edge.getAttribute("data-source-full") || "";
            const targetFull = edge.getAttribute("data-target-full") || "";
            
            const edgeSourceSpeaker = getSpeakerName(sourceFull);
            const edgeTargetSpeaker = getSpeakerName(targetFull);
            
            const matchThreshold = !isNaN(value) && value >= threshold;
            let isActionable = false;

            if (allSpeakers) {{
                isActionable = matchThreshold;
            }} else {{
                const isFromSpeaker = (edgeSourceSpeaker === speakerTarget);
                const isFromSOD = (edgeSourceSpeaker.toLowerCase() === "sod" && edgeTargetSpeaker === speakerTarget);
                const isToEOD = (edgeSourceSpeaker === speakerTarget && edgeTargetSpeaker.toLowerCase() === "eod");
                if (matchThreshold && (isFromSpeaker || isFromSOD || isToEOD)) isActionable = true;
            }}

            if (isActionable) {{
                edge.style.opacity = "1";
                if (label) label.style.opacity = "1";
                nodesWithActiveEdges.add(sourceFull);
                nodesWithActiveEdges.add(targetFull);
            }} else {{
                edge.style.opacity = "0.05";
                if (label) label.style.opacity = "0";
            }}
        }});

        // Esconder/Mostrar Nós
        currentGraphElement.querySelectorAll("g.node").forEach(node => {{
            const nodeName = node.getAttribute("data-full-name") || "";
            const nodeSpeaker = getSpeakerName(nodeName);
            const isTargetSpeaker = !allSpeakers && (nodeSpeaker === speakerTarget);
            const isSpecial = (nodeName.toLowerCase().includes("sod") || nodeName.toLowerCase().includes("eod"));

            let shouldBeActive = false;
            if (allSpeakers) {{
                shouldBeActive = nodesWithActiveEdges.has(nodeName);
            }} else {{
                if (isTargetSpeaker || isSpecial) shouldBeActive = nodesWithActiveEdges.has(nodeName);
            }}

            if (shouldBeActive || (isSpecial && threshold === 0)) {{
                node.style.opacity = "1";
                const shape = node.querySelector("ellipse, polygon, path");
                if (isTargetSpeaker && !isSpecial) {{
                    if(shape) shape.style.strokeWidth = "3px";
                }} else {{
                    if(shape) shape.style.strokeWidth = "";
                }}
            }} else {{
                node.style.opacity = "0.1";
            }}
        }});
        
        atualizarMetricasPorVisibilidade();
    }}


    function atualizarMetricasPorVisibilidade() {{
        setTimeout(() => {{
            if (!currentGraphElement) return;
            
            const visibleNodes = Array.from(currentGraphElement.querySelectorAll("g.node"))
                .filter(n => parseFloat(window.getComputedStyle(n).opacity) > 0.5);
            
            let countDialogue = 0;
            let countActions = 0;

            visibleNodes.forEach(node => {{
                if (node.querySelector("polygon") && !node.querySelector("ellipse")) {{
                    countActions++;
                }} else {{
                    countDialogue++;
                }}
            }});

            const edgesVisiveis = Array.from(currentGraphElement.querySelectorAll("g.edge"))
                .filter(e => parseFloat(window.getComputedStyle(e).opacity) > 0.5).length;

            const countStatesElem = document.getElementById("count-states");
            if (countStatesElem) countStatesElem.innerText = countDialogue;

            const countActionsElem = document.getElementById("count-action-states");
            if (countActionsElem) countActionsElem.innerText = countActions;

            const countEdgesElem = document.getElementById("count-edges");
            if (countEdgesElem) countEdgesElem.innerText = edgesVisiveis;
        }}, 100);
    }}

    function updateThreshold(sliderValueRaw) {{
        const threshold = parseFloat(sliderValueRaw) / 100;
        const sliderLabel = document.getElementById("thresholdLabel");
        if (sliderLabel) sliderLabel.textContent = threshold.toFixed(2);
        applyFilters(threshold, currentSpeaker);
    }}

    const searchInput = document.getElementById("searchInput");
    if (searchInput) {{
        const newInput = searchInput.cloneNode(true);
        searchInput.parentNode.replaceChild(newInput, searchInput);
        
        newInput.addEventListener("input", (e) => {{
            const query = e.target.value.toLowerCase().trim();
            const sld = document.getElementById("thresholdSlider");
            const threshold = sld ? parseFloat(sld.value) / 100 : 0;
            
            if (query === "") {{
                currentGraphElement.querySelectorAll("g.node, g.edge").forEach(el => {{
                    el.classList.remove("faded", "hidden");
                    el.style.opacity = "";
                    const shape = el.querySelector("ellipse, polygon, path");
                    if(shape) shape.style.strokeWidth = ""; 
                }});
                applyFilters(threshold, currentSpeaker);
                return;
            }}

            const matchedNodeTitles = new Set();
            
            currentGraphElement.querySelectorAll("g.node").forEach(node => {{
                const fullText = node.textContent.toLowerCase(); 
                const nodeTitle = node.getAttribute("data-full-name");

                if (fullText.includes(query)) {{
                    matchedNodeTitles.add(nodeTitle);
                    node.classList.remove("faded", "hidden");
                    node.style.opacity = "1";
                    const shape = node.querySelector("ellipse, polygon, path");
                    if(shape) shape.style.strokeWidth = "5px"; 
                }} else {{
                    node.classList.add("faded");
                    node.style.opacity = "0.1";
                    const shape = node.querySelector("ellipse, polygon, path");
                    if(shape) shape.style.strokeWidth = ""; 
                }}
            }});

            currentGraphElement.querySelectorAll("g.edge").forEach(edge => {{
                const sourceFull = edge.getAttribute("data-source-full");
                const targetFull = edge.getAttribute("data-target-full");

                if (matchedNodeTitles.has(sourceFull) || matchedNodeTitles.has(targetFull)) {{
                    edge.classList.remove("faded", "hidden");
                    edge.style.opacity = "1"; 
                }} else {{
                    edge.classList.add("faded");
                    edge.style.opacity = "0.05";
                }}
            }});
            
            atualizarMetricasPorVisibilidade();
        }});
    }}

    function enviarFicheiro(file) {{
        const formData = new FormData();
        formData.append("dot", file);

        fetch("http://127.0.0.1:5051/upload", {{
            method: "POST",
            body: formData
        }})
        .then(res => res.json())
        .then(data => {{
            console.log("Ficheiro recebido. A atualizar métricas.");

            if (data.clusters_speaker) dictFalas = data.clusters_speaker;
            if (data.cluster_id_speaker) dictIds = data.cluster_id_speaker;

            if (data.dot_code) {{
                renderNewDot(data.dot_code);
            }}
            
            setTimeout(() => {{
                const svg = document.querySelector("#graph-container svg");
                let n_dialogue = 0;
                let n_actions = 0;
                let n_edges = 0;
                
                if (svg) {{
                    const allNodes = svg.querySelectorAll("g.node");
                    n_edges = svg.querySelectorAll("g.edge").length;

                    allNodes.forEach(node => {{
                        // polígono = ação, elipse = diálogo
                        if (node.querySelector("polygon") && !node.querySelector("ellipse")) {{
                            n_actions++;
                        }} else {{
                            n_dialogue++;
                        }}
                    }});
                }}

                let strActionsGraph = "";
                if (n_actions > 0) {{
                    strActionsGraph = `Number of Action States: <b id="count-action-states">${{n_actions}}</b><br>`;
                }}

                const metricsDiv = document.getElementById("metrics-container");
                if (metricsDiv) {{
                    metricsDiv.innerHTML = `
                    <div style="font-size: 14px; line-height: 1.6; margin-top: 5px;">
                        <h3 style="margin-bottom: 5px;">Clustering Metrics</h3>
                        
                        <span class="metric-help" title="{tips['n_dial_states']}">Number of Dialogue States</span>: <b id="count-states">${{n_dialogue}}</b><br>
                        
                        ${{strActionsGraph ? strActionsGraph.replace("Number of Action States", '<span class="metric-help" title="{tips['n_act_states']}">Number of Action States</span>') : ""}}
                        
                        <h3 style="margin-bottom: 5px; margin-top:10px;">Flow Metrics</h3>
                        <span class="metric-help" title="{tips['n_trans']}">Number of Transitions</span>: <b id="count-edges">${{n_edges}}</b>
                    </div>
                    
                    <div id="sentiment-slot" style="font-size: 14px; line-height: 1.6;"></div>

                    <p style="font-size: 10px; color: #888; margin-top: 15px; line-height: 1.2; border-top: 1px solid #444; padding-top: 5px;">
                        *Detailed metrics not available for uploaded .dot files.
                    </p>
                `;
                }}
            }}, 500); 
        }})
        .catch(err => {{
            console.error("Erro na comunicação com o Flask:", err);
        }});
    }}

    const dotFileInput = document.getElementById("dotFileInput");
    if (dotFileInput) {{
        dotFileInput.addEventListener("change", function(event) {{
            const file = event.target.files[0];
            if (!file) return;

            // 1. Atualiza nomes na UI
            const lblText = document.getElementById("fileLabelText");
            if (lblText) lblText.textContent = file.name;
            
            const subtitle = document.getElementById("file-subtitle");
            if (subtitle) subtitle.textContent = "Dialogue Flow Discovery for: " + file.name;

            const reader = new FileReader();
            reader.onload = function(e) {{
                const dotContent = e.target.result;
                renderNewDot(dotContent);
                enviarFicheiro(file);
                setTimeout(() => {{
                    updateUploadedSentiment(dotContent);
                }}, 700);
            }};
            reader.readAsText(file, 'utf-8');

            
        }});
    }}

    function renderNewDot(dotString) {{
        console.log("A desenhar o grafo e a aplicar interatividade...");
        const container = document.getElementById("graph-container");
        container.innerHTML = "";

        viz.renderSVGElement(dotString)
            .then(element => {{
                const nodeNames = [];
                element.querySelectorAll("g.node").forEach(n => {{
                    let t = n.querySelector("title");
                    if(t) {{
                        const fullTitle = t.textContent.trim();
                        n.setAttribute("data-full-name", fullTitle);
                        nodeNames.push(fullTitle);
                    }}
                }});
                
                element.querySelectorAll("g.edge").forEach(edge => {{
                    const title = edge.querySelector("title");
                    if (title) {{
                        const text = title.textContent.trim(); 
                        let sourceFull = ""; let targetFull = "";
                        
                        for (const name of nodeNames) {{
                            if (text.startsWith(name + "->")) sourceFull = name;
                            if (text.endsWith("->" + name)) targetFull = name;
                        }}
                        if (sourceFull && targetFull) {{
                            edge.setAttribute("data-source-full", sourceFull);
                            edge.setAttribute("data-target-full", targetFull);
                        }}
                    }}
                }});

                if (typeof makeNodesDraggable === "function") makeNodesDraggable(element); 
                if (typeof updateSpeakersFromGraph === "function") updateSpeakersFromGraph(element);
                if (typeof updateLegendFromGraph === "function") updateLegendFromGraph(element);
                
                updateEdgeThickness(element);
                enableNeighborHighlight(element);
                
                if (typeof initTooltips === "function") initTooltips(element);

                const titles = element.getElementsByTagName("title");
                while (titles.length > 0) titles[0].remove();

                container.appendChild(element); 
                currentGraphElement = element;
                
                // Força o filtro da barra se ele não estiver a zeros
                const sld = document.getElementById("thresholdSlider");
                const val = sld ? parseFloat(sld.value)/100 : 0;
                applyFilters(val, currentSpeaker);
            }})
            .catch(error => {{
                console.error("Erro ao renderizar:", error);
            }});
    }}

    function openSidebar_reset() {{
        document.getElementById("sidebar").style.width = "315px";
        document.getElementById("openSidebar").style.display = "none";
        document.getElementById("grafico").style.marginLeft = "315px";
        document.getElementById("sidebar").style.transition = "all 0s ease";
    }}
    
    function openSidebar() {{
        document.getElementById("sidebar").style.width = "315px";
        document.getElementById("openSidebar").style.display = "none";
        document.getElementById("grafico").style.marginLeft = "315px";
    }}

    function closeSidebar() {{
        document.getElementById("sidebar").style.width = "0";
        document.getElementById("openSidebar").style.display = "block";
        document.getElementById("grafico").style.marginLeft = "0";
    }}

    function desativaModos() {{
        isPanning = false;
        isZoomBoxActive = false;
        btnPan.classList.remove("active");
        btnZoomBox.classList.remove("active");
        btnPan.style.backgroundColor = "";
        btnZoomBox.style.backgroundColor = "";
        graphWrapper.style.cursor = "default";
        if (selectionRect) selectionRect.style.display = "none";
    }}

    function getRelativePositionInImage(e) {{
        const wrapperRect = graphWrapper.getBoundingClientRect();
        const mouseX = e.clientX - wrapperRect.left;
        const mouseY = e.clientY - wrapperRect.top;
        return {{ x: (mouseX - translateX) / zoomLevel, y: (mouseY - translateY) / zoomLevel }};
    }}

    btnPan.addEventListener("click", () => {{
        if (isPanning) desativaModos();
        else {{
            desativaModos();
            isPanning = true;
            btnPan.classList.add("active");
            btnPan.style.backgroundColor = "#0055aa";
            graphWrapper.style.cursor = "grab";
        }}
    }});

    btnZoomBox.addEventListener("click", () => {{
        if (isZoomBoxActive) desativaModos();
        else {{
            desativaModos();
            isZoomBoxActive = true;
            btnZoomBox.classList.add("active");
            btnZoomBox.style.backgroundColor = "#0055aa";
            graphWrapper.style.cursor = "crosshair";

            if (!selectionRect) {{
                selectionRect = document.createElement("div");
                selectionRect.id = "selection-rectangle";
                selectionRect.style.position = "absolute";
                selectionRect.style.border = "2px dashed #0055aa";
                selectionRect.style.backgroundColor = "rgba(0, 85, 170, 0.2)";
                selectionRect.style.pointerEvents = "none";
                graphWrapper.appendChild(selectionRect);
            }}
            selectionRect.style.display = "none";
        }}
    }});

    graphWrapper.addEventListener("mousedown", (e) => {{
        if (isZoomBoxActive) {{
            e.preventDefault();
            const pos = getRelativePositionInImage(e);
            startX = pos.x;
            startY = pos.y;
            selectionRect.style.left = `${{startX * zoomLevel + translateX}}px`;
            selectionRect.style.top = `${{startY * zoomLevel + translateY}}px`;
            selectionRect.style.width = "0px";
            selectionRect.style.height = "0px";
            selectionRect.style.display = "block";
        }} else if (isPanning) {{
            e.preventDefault();
            isDragging = true;
            startX = e.clientX - translateX;
            startY = e.clientY - translateY;
            graphWrapper.style.cursor = "grabbing";
        }}
    }});

    graphWrapper.addEventListener("mousemove", (e) => {{
        if (isZoomBoxActive && selectionRect.style.display !== "none") {{
            const pos = getRelativePositionInImage(e);
            const rectX = Math.min(pos.x, startX);
            const rectY = Math.min(pos.y, startY);
            const rectWidth = Math.abs(pos.x - startX);
            const rectHeight = Math.abs(pos.y - startY);
            selectionRect.style.left = `${{rectX * zoomLevel + translateX}}px`;
            selectionRect.style.top = `${{rectY * zoomLevel + translateY}}px`;
            selectionRect.style.width = `${{rectWidth * zoomLevel}}px`;
            selectionRect.style.height = `${{rectHeight * zoomLevel}}px`;
        }}
    }});

    document.addEventListener("mousemove", (e) => {{
        if (isPanning && isDragging) {{
            translateX = e.clientX - startX;
            translateY = e.clientY - startY;
            container.style.transform = `translate(${{translateX}}px, ${{translateY}}px) scale(${{zoomLevel}})`;
            container.style.transformOrigin = "0 0";
        }}
    }});

    graphWrapper.addEventListener("mouseup", (e) => {{
        if (isZoomBoxActive && selectionRect.style.display !== "none") {{
            selectionRect.style.display = "none";
            const pos = getRelativePositionInImage(e);
            const x1 = Math.min(startX, pos.x);
            const y1 = Math.min(startY, pos.y);
            const x2 = Math.max(startX, pos.x);
            const y2 = Math.max(startY, pos.y);
            const selectedWidth = x2 - x1;
            const selectedHeight = y2 - y1;
            if (selectedWidth >= 10 && selectedHeight >= 10) {{
                const wrapperRect = graphWrapper.getBoundingClientRect();
                const scaleX = wrapperRect.width / selectedWidth;
                const scaleY = wrapperRect.height / selectedHeight;
                zoomLevel = Math.min(scaleX, scaleY);
                translateX = -x1 * zoomLevel;
                translateY = -y1 * zoomLevel;
                container.style.transform = `translate(${{translateX}}px, ${{translateY}}px) scale(${{zoomLevel}})`;
                container.style.transformOrigin = "0 0";
            }}
            desativaModos();
        }}
    }});

    document.addEventListener("mouseup", () => {{
        if (isPanning) {{
            isDragging = false;
            graphWrapper.style.cursor = "grab";
        }}
    }});

    btnReset.addEventListener("click", () => {{
        zoomLevel = 1;
        translateX = 0;
        translateY = 0;
        if (graphWrapper) {{
            graphWrapper.style.transform = "scale(1)";
            graphWrapper.style.transformOrigin = "50% 50%";
        }}
        if (container) {{
            container.style.transform = "translate(0px, 0px) scale(1)";
            container.style.transformOrigin = "center center";
        }}
        const t = document.getElementById("tooltip");
        if (t) t.style.display = "none";
    }});  

    btnZoomIn.addEventListener("click", () => {{
        zoomLevel = Math.min(zoomLevel + 0.1, 3);
        graphWrapper.style.transform = `scale(${{zoomLevel}})`;
        graphWrapper.style.transformOrigin = "50% 50%";
    }});

    btnZoomOut.addEventListener("click", () => {{
        const isBoxZoomed = container && container.style.transform.includes("scale") && !container.style.transform.includes("scale(1)");
        if (isBoxZoomed && zoomLevel > 1) zoomLevel = 1;
        zoomLevel = Math.max(zoomLevel - 0.1, 0.1); 
        if (graphWrapper) {{
            graphWrapper.style.transform = `scale(${{zoomLevel}})`;
            graphWrapper.style.transformOrigin = "50% 50%";
        }}
    }});

    // ZOOM COM A RODA DO RATO (SCROLL)
    graphWrapper.addEventListener("wheel", (e) => {{
        e.preventDefault(); // Impede a página de fazer scroll para cima/baixo

        // Define a suavidade do zoom (0.05 é mais suave que os botões)
        const zoomSpeed = 0.05; 

        if (e.deltaY < 0) {{
            // Rodou para cima -> Zoom In (limite máximo de 3x)
            zoomLevel = Math.min(zoomLevel + zoomSpeed, 3);
        }} else {{
            // Rodou para baixo -> Zoom Out (limite mínimo de 0.1x)
            zoomLevel = Math.max(zoomLevel - zoomSpeed, 0.1);
        }}

        // Aplica a escala tal como os botões fazem
        if (graphWrapper) {{
            graphWrapper.style.transform = `scale(${{zoomLevel}})`;
            graphWrapper.style.transformOrigin = "50% 50%";
        }}
    }}, {{ passive: false }});

    btnScreenshot.addEventListener("click", () => {{
        const wrapper = document.getElementById("graph-wrapper");
        const rect = wrapper.getBoundingClientRect();
        
        // Capturar o nome do ficheiro do subtítulo
        const subtitle = document.getElementById("file-subtitle").textContent;
        // Extrai o que está depois de "for: " e remove espaços extras
        let fileName = subtitle.split("for:")[1] ? subtitle.split("for:")[1].trim() : "flow_disco";
        // Remove a extensão .dot original
        fileName = fileName.replace(".dot", ""); 

        html2canvas(document.body, {{
            backgroundColor: "#ffffff",
            scale: 2,
            useCORS: true,
            windowWidth: window.innerWidth,
            windowHeight: window.innerHeight
        }}).then(canvas => {{
            const scale = 2;
            const cropX = rect.left * scale;
            const cropY = rect.top * scale;
            const cropWidth = rect.width * scale;
            const cropHeight = rect.height * scale;
            
            const croppedCanvas = document.createElement("canvas");
            croppedCanvas.width = cropWidth;
            croppedCanvas.height = cropHeight;
            const ctx = croppedCanvas.getContext("2d");
            
            ctx.drawImage(canvas, cropX, cropY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);
            
            const link = document.createElement("a");
            // Usa o nome extraído dinamicamente
            link.download = fileName + ".png";
            link.href = croppedCanvas.toDataURL("image/png");
            link.click();
        }}).catch(error => {{
            console.error(error);
            alert("Erro ao capturar imagem.");
        }});
    }});

    document.querySelectorAll(".collapsible-header, .collapsible-header1").forEach(header => {{
        header.addEventListener("click", () => {{
            header.parentElement.classList.toggle("collapsed");
        }});
    }});

    document.addEventListener("DOMContentLoaded", () => {{
        if (typeof dotCode !== 'undefined') {{
            renderNewDot(dotCode);
        }}
    }});

    // MODAL DO HEATMAP
    const modalHeatmap = document.getElementById("heatmapModal");
    const btnHeatmap = document.getElementById("btn-heatmap");
    const closeHeatmap = document.getElementById("closeHeatmap");

    if (btnHeatmap && modalHeatmap) {{
        btnHeatmap.addEventListener("click", () => {{
            modalHeatmap.style.display = "block";
        }});
    }}

    if (closeHeatmap) {{
        closeHeatmap.addEventListener("click", () => {{
            modalHeatmap.style.display = "none";
        }});
    }}

    // Fechar se clicar fora da imagem
    window.addEventListener("click", (e) => {{
        if (e.target === modalHeatmap) {{
            modalHeatmap.style.display = "none";
        }}
    }});
    
    // Fechar com a tecla ESC
    document.addEventListener("keydown", (e) => {{
        if (e.key === "Escape" && modalHeatmap.style.display === "block") {{
            modalHeatmap.style.display = "none";
        }}
    }});
</script>

</body>
</html>
"""

# Criar o nome do ficheiro HTML automaticamente
nome_base = str(filename).split('.')[0]
nome_html = f"Interface_{nome_base}.html"

# Juntar à diretoria dos resultados
caminho_html = os.path.join('./Resultados/' + diretoria, nome_html)

# Garantir que a pasta existe e guardar
os.makedirs(os.path.dirname(caminho_html), exist_ok=True)

with open(caminho_html, 'w', encoding='utf-8') as f:
    f.write(html_content)

webbrowser.open('file://' + os.path.realpath(caminho_html))

n_cluster = 0
for speaker in speakers:
    if speaker in y_predicted_speaker:
        n_cluster += len(np.unique(y_predicted_speaker[speaker][y_predicted_speaker[speaker]!= -1]))

print(f"Total Clusters: {n_cluster}")
try:
    print(f"Total Nodes: {n_nodes}")
except NameError:
    print("n_nodes não definido neste contexto.")

# Fluxo com 14 níveis
from graphviz import Digraph
from collections import Counter
import pandas as pd
import os

CORES_ATOS = {
    'REPRESENTATION': '#A9CCE3', 
    'SUBJECTIVITY': '#A9DFBF',   
    'EVALUATION': '#F5B7B1',     
    'FIGURATION': '#D7BDE2',     
    'DYNAMICS': '#F5CBA7'        
}

def gerar_fluxo_hierarquico(df_atual, diretoria_out, nome_ficheiro, titulo_grafo=""):
    print("\n" + "="*50)
    print(f"A GERAR O FLUXO HIERÁRQUICO: {titulo_grafo}")
    print("="*50)
    
    if isinstance(df_atual, list):
        df_soneto = pd.DataFrame(df_atual)
    else:
        df_soneto = df_atual.copy()
    
    if df_soneto.empty:
        print("Sem dados para gerar o fluxo hierárquico.")
        return

    # Corta qualquer linha gerada por alucinação (acima da 14ª)
    df_soneto = df_soneto[df_soneto['turn_id'] <= 14]

    dot = Digraph(comment='Hierarchical Flow')
    dot.attr(rankdir='TB', ranksep='0.6', nodesep='0.5')
    
    if titulo_grafo:
        dot.attr(label=f'<<B>{titulo_grafo}</B>>', labelloc='t', fontsize='22', fontname='Arial')
    
    dot.node('SOP', 'SOP', shape='circle', style='filled', fillcolor='#FFCC00', color='#FFCC00', fontname='Arial', fontcolor='black')
    dot.node('EOP', 'EOP', shape='circle', style='filled', fillcolor='#FFCC00', color='#FFCC00', fontname='Arial', fontcolor='black')
    
    transicoes = []
    niveis = {i: set() for i in range(1, 15)}
    
    for p_id, grupo in df_soneto.groupby('dialogue_id'):
        grupo = grupo.sort_values('turn_id')
        
        if 'speech_act_gerado' in grupo.columns:
            atos = grupo['speech_act_gerado'].fillna("Indef").tolist()
        elif 'trueLabel' in grupo.columns:
            atos = grupo['trueLabel'].fillna("Indef").tolist()
        else:
            continue
            
        turn_ids = grupo['turn_id'].tolist()
        if len(atos) == 0: continue
            
        for i in range(len(atos)):
            linha_num = int(turn_ids[i])
            if 1 <= linha_num <= 14:
                niveis[linha_num].add(f"L{linha_num}_{atos[i]}")
            
        no_inicial = f"L{int(turn_ids[0])}_{atos[0]}"
        transicoes.append(('SOP', no_inicial))
        
        for i in range(len(atos) - 1):
            origem = f"L{int(turn_ids[i])}_{atos[i]}"
            destino = f"L{int(turn_ids[i+1])}_{atos[i+1]}"
            transicoes.append((origem, destino))
            
        no_final = f"L{int(turn_ids[-1])}_{atos[-1]}"
        transicoes.append((no_final, 'EOP'))

    for i in range(1, 15):
        dot.node(f'Label_L{i}', f'Line {i}', shape='plaintext', fontname='Arial', fontsize='12', fontcolor='black', fontstyle='bold')
        if i < 14:
            dot.edge(f'Label_L{i}', f'Label_L{i+1}', style='invis')

    for i in range(1, 15):
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node(f'Label_L{i}')
            for no in niveis[i]:
                ato_label = no.split("_", 1)[1] if "_" in no else no
                ato_limpo = ato_label.strip().upper()
                cor_fundo = CORES_ATOS.get(ato_limpo, '#E5E7E9') 
                s.node(no, ato_label, shape='ellipse', style='filled', fillcolor=cor_fundo, color='black', penwidth='1.0', fontname='Arial', fontcolor='black')

    contagem = Counter(transicoes)
    origens_totais = {}
    for (orig, dest), count in contagem.items():
        origens_totais[orig] = origens_totais.get(orig, 0) + count

    # CÁLCULO DA DENSIDADE
    num_nos = 2 + sum(len(nos) for nos in niveis.values())
    num_arestas = len(contagem)
    
    densidade = 0.0
    if num_nos > 1:
        densidade = num_arestas / (num_nos * (num_nos - 1))
        
    print(f"\nMÉTRICAS:")
    print(f"   -> Nós: {num_nos}")
    print(f"   -> Arestas: {num_arestas}")
    print(f"   -> Densidade Global: {densidade:.4f} ({(densidade * 100):.2f}%)")
    print("==================================================\n")

    for (orig, dest), count in contagem.items():
        prob = count / origens_totais[orig]
        
        espessura = str(1.0 + (prob * 5.0))
        intensidade = int(max(0, min(200, 200 - (prob * 200))))
        cor = f"#{intensidade:02x}{intensidade:02x}{intensidade:02x}"
        
        dot.edge(orig, dest, label=f"{prob:.2f}", penwidth=espessura, color=cor, fontcolor=cor, fontname='Arial', fontsize='10')

    caminho_final = os.path.join('./Resultados', diretoria_out, nome_ficheiro)
    try:
        dot.render(caminho_final, format='png', cleanup=True)
        print(f"Diagrama guardado em: {caminho_final}.png\n")
    except Exception as e:
        print(f"Erro ao gerar PNG. Ficheiro .dot guardado em: {caminho_final}.dot")
        dot.save(f"{caminho_final}.dot")

# CHAMADA DA FUNÇÃO
# try:
#     traducao_dominios = {'Desporto': 'Sports', 'Música': 'Music', 'Cidades': 'Cities'}
#     modelo_original = MODELOS[0] if 'MODELOS' in locals() and MODELOS else "Modelo"
#     nome_modelo_limpo = modelo_original.replace(":", "_")
#     dominio_pt = FILTRO_EXTRA_VALOR if 'FILTRO_EXTRA_VALOR' in locals() else "Domínio"
#     dominio_en = traducao_dominios.get(dominio_pt, dominio_pt) 
#     temp_val = TEMP if 'TEMP' in locals() else "X"
#     
#     meu_titulo = f"{dominio_en} ({modelo_original} | T={temp_val})"
#     nome_fich_hierarquico = f"Fluxo_Hierarquico_{nome_modelo_limpo}_{dominio_en}_T{temp_val}"
#     diretoria = "Imagens_Grafos" if 'diretoria' not in locals() else diretoria
#     gerar_fluxo_hierarquico(df_final, diretoria, nome_fich_hierarquico, titulo_grafo=meu_titulo)
# except Exception as e:
#     print(f"Erro na chamada da função: {e}")
# -----------------------------------

# 1. FUNÇÕES AUXILIARES
def validar_matematica_transicoes(df_atual, linha_origem=4, ato_origem='FIGURATION'):
    print(f"Teste para analisar a transição da Linha {linha_origem} ({ato_origem}) para a Linha {linha_origem + 1}\n")
    
    if isinstance(df_atual, list):
        df_base = pd.DataFrame(df_atual)
    else:
        df_base = df_atual.copy()

    # Garantir que usamos apenas as falas
    if 'Tipo' in df_base.columns:
        df_soneto = df_base[df_base['Tipo'].astype(str).str.lower() == 'fala'].copy()
    else:
        df_soneto = df_base.copy()

    coluna_ato = 'speech_act_gerado' if 'speech_act_gerado' in df_soneto.columns else 'trueLabel'
    
    # 1. Encontrar todos os poemas que na Linha de Origem tiveram o Ato de Origem
    poemas_na_origem = df_soneto[
        (df_soneto['turn_id'] == linha_origem) & 
        (df_soneto[coluna_ato] == ato_origem)
    ]
    
    ids_poemas = poemas_na_origem['dialogue_id'].tolist()
    total_origem = len(ids_poemas)
    
    if total_origem == 0:
        print(f"Não há nenhum poema que tenha '{ato_origem}' na Linha {linha_origem}.")
        return

    print(f"PASSO 1: O modelo gerou '{ato_origem}' na Linha {linha_origem} em exatamente {total_origem} poemas.")
    
    # 2. Olhar apenas para a Linha Seguinte, MAS APENAS para os IDs que encontrámos no Passo 1
    linha_seguinte = linha_origem + 1
    poemas_no_destino = df_soneto[
        (df_soneto['turn_id'] == linha_seguinte) & 
        (df_soneto['dialogue_id'].isin(ids_poemas))
    ]
    
    # 3. Contar para onde é que eles foram
    contagem_destinos = poemas_no_destino[coluna_ato].value_counts()
    
    for destino, quantidade in contagem_destinos.items():
        # 4. Calcular a probabilidade
        probabilidade = quantidade / total_origem
        print(f"-> Foram para '{destino}': {quantidade} poemas")
        print(f"   {quantidade} / {total_origem} = {probabilidade:.2f} ({probabilidade*100:.0f}%)")


# GERAÇÃO DO FLUXO HIERÁRQUICO (APENAS PARA POESIA)
if TIPO_DATASET == "POESIA":
    try:
        modelo_original = MODELOS[0] if 'MODELOS' in locals() and MODELOS else "Modelo"
        nome_modelo_limpo = modelo_original.replace(":", "_")
        temp_val = TEMP if 'TEMP' in locals() else "X"
        
        dominio_en = "Poetry" 
        titulo_visual = "Poetry Domain"
        
        meu_titulo = f"{titulo_visual} ({modelo_original} | T={temp_val})"
        nome_fich_hierarquico = f"Fluxo_Hierarquico_{nome_modelo_limpo}_{dominio_en}_T{temp_val}"
        
        diretoria_grafos = "Imagens_Grafos" if 'diretoria' not in locals() else diretoria
        
        # Chama a função de gerar as 14 linhas
        gerar_fluxo_hierarquico(df_final, diretoria_grafos, nome_fich_hierarquico, titulo_grafo=meu_titulo)
        
        # Faz a validação matemática
        validar_matematica_transicoes(df_final, linha_origem=1, ato_origem='DYNAMICS')
        
    except Exception as e:
        print(f"Erro na geração do fluxo de poesia: {e}")
else:
    print(f"\nFluxo Hierárquico e validação ignorados (dataset no modo '{TIPO_DATASET}').")


# LIMPEZA DE FICHEIROS TEMPORÁRIOS
directory = './Resultados/' + diretoria
files_to_keep = (".csv", ".pdf", ".xlsx", ".txt", ".png", ".json", ".html", ".dot", ".pkl")

if os.path.exists(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and not file.lower().endswith(files_to_keep):
            try:
                os.remove(file_path)
                print(f"Apagado ficheiro temporário: {file}")
            except Exception as e:
                print(f"Erro ao apagar {file}: {e}")


# 4. INÍCIO DO SERVIDOR FLASK
print("\nO servidor Flask está a iniciar na porta 5051...")
try:
    app.run(host='0.0.0.0', port=5051, debug=False, use_reloader=False)
except Exception as e:
    print(f"\nErro ao iniciar o Flask: {e}")
