import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from plataformateste.clustering import clustering_kmeans
from plataformateste.clusters_description import (
    describe_clusters_bigrams,
    describe_clusters_closest,
    describe_clusters_verbs,
)
from plataformateste.graphs import generate_markov_chain
from plataformateste.vector_representation import (
    preprocessing_tfidf,
    topic_features_to_remove,
    use_sentence_transformer,
    word2vec,
)

PROBLEM = "single"
NUMBER_OF_FEATURES = 100
CLUSTER_OPT = "equal"
# Models of SentenceTransformer choosed by results here: 'https://www.sbert.net/docs/pretrained_models.html'
# options: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'all-distilroberta-v1', 'all-MiniLM-L12-v2'
MODEL_NAME = "all-MiniLM-L6-v2"


def choose_number_of_clusters(normalized_df, opt=CLUSTER_OPT):
    column = normalized_df["trueLabel"].value_counts()

    total = column.sum()
    max_value = column.max()

    number_exact = normalized_df.trueLabel.nunique()
    total = total / max_value
    total = round(total, 0)
    total = int(total)

    if opt == "equal":
        n_clusters = number_exact
    elif opt == "total":
        n_clusters = total
    else:
        raise ValueError

    return n_clusters


def normalize_dataset(df_initial, regex=None, speaker=None):
    """Normalize turn_id by (all turn_id's/max turn_id) and some column names."""
    df = df_initial.copy()

    url_pattern = r"https?://\S+"
    url_placeholder = "xURLx"
    user_tags_pattern = "^@\S+"
    user_tags_placeholder = "xUSERNAMEx"

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

    if "turn_id" in df.columns:
        max_value = np.max(df["turn_id"])
        df["turn_id"] = df["turn_id"] / max_value
        df["turn_id"] = df["turn_id"].round(decimals=3)

    if "trueLabel" in df.columns:
        df["trueLabel"] = df["trueLabel"].replace(" ", "_", regex=True)

    df["utterance"] = df["utterance"].apply(lambda x: x.lower())

    if regex is True:
        df["utterance"] = df["utterance"].replace(
            to_replace=url_pattern, value=url_placeholder, regex=True
        )
        df["utterance"] = df["utterance"].replace(
            to_replace=user_tags_pattern, value=user_tags_placeholder, regex=True
        )

    # greetings_stopwords = ["hello", "hi", "bye", "goodbye", "hey"]

    if speaker == "0" or speaker == "USER":
        df = df[df["Speaker"].values == "USR"]
        df = df.reset_index(drop=True)
    elif speaker == "1" or speaker == "SERVICE":
        df = df[df["Speaker"].values == "SYS"]
        df = df.reset_index(drop=True)

    return df


def set_labels(normalized_df, problem=PROBLEM):
    """Change the labels according to the problem to be treated."""
    normalized_df["trueLabel"] = normalized_df["trueLabel"].fillna("none")
    normalized_df["trueLabel"] = normalized_df["trueLabel"].astype(str)
    print(normalized_df["trueLabel"].value_counts())
    set_of_labels = set()

    if problem == "multi":
        print("Multi-Label Problem")
        for i in range(len(normalized_df["trueLabel"])):
            for j in normalized_df["trueLabel"][i].split("\n"):
                set_of_labels.add(j.split(",")[0])
    elif problem == "single":
        print("Single-Label Problem")
        for i in range(len(normalized_df["trueLabel"])):
            for j in normalized_df["trueLabel"][i].split("\n"):
                set_of_labels.add(j.split(" ")[0])
    else:
        raise ValueError

    print(set_of_labels)
    return set_of_labels


def evaluation(y_predicted, normalized_df, set_of_labels, n_clusters):
    evaluation_df = pd.DataFrame()

    evaluation_df["trueLabel"] = normalized_df["trueLabel"].fillna("none")

    labels_pred = np.array(y_predicted)

    ground_truth = pd.DataFrame(evaluation_df, columns=["trueLabel"])

    map_of_labels = dict(zip(set_of_labels, np.arange(n_clusters)))

    labels_true = evaluation_df["trueLabel"].map(map_of_labels)

    counts = np.linspace(0, 0, num=n_clusters * n_clusters).reshape(
        (n_clusters, n_clusters)
    )  # columns clusters, lines labels

    for i in range(len(labels_pred)):
        for j in ground_truth["trueLabel"][i].split(";"):
            if j.split(" ")[0] != "noofferr":
                counts[map_of_labels[j.split(",")[0]]][int(labels_pred[i])] = (
                    counts[map_of_labels[j.split(",")[0]]][int(labels_pred[i])] + 1
                )
    print("\n")
    print(counts)
    for i in range(n_clusters):  # cluster i
        maior = 0
        indice = -1
        for j in range(n_clusters):  # label j
            if counts[j][i] >= maior:
                maior = counts[j][i]
                indice = j
        if indice != -1:
            print("cluster:", i, "label:", list(set_of_labels)[indice])
            evaluation_df["trueLabel"] = evaluation_df["trueLabel"].replace(
                [list(set_of_labels)[indice]], [i]
            )
        else:
            print("cluster:", i, "label:not defined")

    print(
        "\nExternal Evaluation:\nAccuracy: ",
        metrics.accuracy_score(labels_true, labels_pred),
    )
    print("V-Measure: ", metrics.v_measure_score(labels_true, labels_pred))


def transition_matrix(n_states, transitions):  # transitions is for y_predicted
    matrix = [[0] * n_states for _ in range(n_states)]

    for i, j in zip(transitions, transitions[1:]):
        matrix[i][j] += 1

    # now convert to probabilities:
    for row in matrix:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
    return matrix


def create_matrix(n_clusters, clus_labelled, y_predicted):  # return of matrix
    m = transition_matrix(n_clusters, y_predicted)
    names = [_ for _ in clus_labelled["labels"]]
    df_matrix = pd.DataFrame(m, index=names, columns=names)
    df_matrix = df_matrix.round(decimals=2)

    return df_matrix


def silhouette_method(data, min_k, max_k, incr):
    number_clusters = 0
    atual_silhouette = 0.000
    # Prepare the scaler
    scale = StandardScaler().fit(data)

    # Fit the scaler
    scaled_data = pd.DataFrame(scale.fit_transform(data))

    # For the silhouette method k needs to start from 2
    n_clusters_axis = range(min_k, max_k, incr)
    silhouettes = []

    # Fit the method
    for k in n_clusters_axis:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init="random")
        kmeans.fit(scaled_data)
        score = metrics.silhouette_score(scaled_data, kmeans.labels_)
        silhouettes.append(score)
        print(score)
        if score > atual_silhouette:
            number_clusters = k
            atual_silhouette = score

    # Plot the results
    plt.figure(figsize=(15, 5))
    plt.plot(n_clusters_axis, silhouettes, "bx-")
    plt.xlabel("Values of K")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Method")
    plt.grid(True)
    plt.show()

    return number_clusters


def run_test(
    df: pd.DataFrame,
    package: str,
    representation: str,
    labels_type: str,
    n_clusters: int = 0,
    model_cache_folder: str = None,
):
    nlp = spacy.load(package)
    stopwordseng = nltk.corpus.stopwords.words("english")

    # normalize turn_id, regex if true normalize URL's and Usernames started with '@', remove greeting words
    normalized_df = normalize_dataset(df, regex=True, speaker=False)

    # Functions to transform into vectors (WORD EMBEDDING)
    if representation == "tfidf":
        topic_features = topic_features_to_remove(
            nlp, normalized_df, NUMBER_OF_FEATURES, topic_feature=False
        )  # topic features
        vectors = preprocessing_tfidf(
            stopwordseng, normalized_df, topic_features, 0.8, 3
        )
    elif representation == "word2vec":
        vectors = word2vec(nlp, normalized_df)
    elif representation == "sentenceTransformer":
        model = SentenceTransformer(MODEL_NAME, cache_folder=model_cache_folder)
        vectors = use_sentence_transformer(normalized_df, model)
    else:
        raise ValueError

    if "trueLabel" in normalized_df.columns:
        # equal | total # nCluster = nDA / Int | total / max
        n_clusters = choose_number_of_clusters(normalized_df)
    elif not n_clusters:
        # unsupervised way (vectors, minK, maxK, increment) sequence of numbers from minK to maxK, but increment
        n_clusters = silhouette_method(vectors, 2, 14, 1)

    print("O número de Clusters a usar será " + str(n_clusters) + "!!")

    # K-Means
    y_predicted, centers = clustering_kmeans(vectors, n_clusters)

    # só caso exista labels
    if "trueLabel" in normalized_df.columns:
        # nº de Dialog Acts / Intents reais -> usado para o t-SNE e PCA
        # nDAsInt = normalizedDF['trueLabel'].nunique()
        set_of_labels = set_labels(normalized_df)
        evaluation(y_predicted, normalized_df, set_of_labels, n_clusters)

    # Describe Clusters -> Bigrams | Verbs | Closest Document
    if labels_type == "verbs":
        verbs = describe_clusters_verbs(nlp, n_clusters, normalized_df, y_predicted)
        print(verbs)

        # change first position according the way we wanna describe the clusters
        matrix = create_matrix(n_clusters, verbs, y_predicted)
    elif labels_type == "closestDocuments":
        closest_documents = describe_clusters_closest(
            n_clusters, normalized_df, y_predicted, vectors, centers
        )
        print(closest_documents)

        # change first position according the way we wanna describe the clusters
        matrix = create_matrix(n_clusters, closest_documents, y_predicted)
    elif labels_type == "bigrams":
        bigrams = describe_clusters_bigrams(n_clusters, normalized_df, y_predicted)
        print(bigrams)

        # change first position according the way we wanna describe the clusters
        matrix = create_matrix(n_clusters, bigrams, y_predicted)
    else:
        raise ValueError

    return generate_markov_chain(n_clusters, matrix)
