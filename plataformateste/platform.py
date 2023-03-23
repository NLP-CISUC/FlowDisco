import re
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from spacy_download import load_spacy

from plataformateste.clustering import clustering_kmeans
from plataformateste.clusters_description import (
    describe_clusters_bigrams,
    describe_clusters_closest,
    describe_clusters_verbs,
)
from plataformateste.graphs import (
    generate_markov_chain,
    generate_markov_chain_separately,
)
from plataformateste.vector_representation import (
    preprocessing_tfidf,
    topic_features_to_remove,
    use_sentence_transformer,
    word2vec,
)

PROBLEM = "multi"
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


def normalize_dataset(df_initial, regex=None, removeGreetings=None, speaker=None):
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

    if "speaker" in df.columns:
        df.rename(columns={"speaker": "Speaker"}, inplace=True)

    if "trueLabel" in df.columns:
        df["trueLabel"] = df["trueLabel"].replace(" ", "_", regex=True)

    if "trueLabel" in df.columns:
        df["utterance"] = df["utterance"].apply(lambda x: x.lower())

    if "trueLabel" in df.columns:
        df.trueLabel = df.trueLabel.fillna("none")

    if regex is True:
        df["utterance"] = df["utterance"].replace(
            to_replace=url_pattern, value=url_placeholder, regex=True
        )
        df["utterance"] = df["utterance"].replace(
            to_replace=user_tags_pattern, value=user_tags_placeholder, regex=True
        )

    greetings_stopwords = ["hello", "hi", "bye", "goodbye", "hey"]

    if speaker == "both":
        df = df

    if speaker == "0" or speaker == "USER":
        df = df[df["Speaker"].values == "USR"]
        # df = df[df['speaker'].values == 'USER'] #multiwoz
        df = df.reset_index(drop=True)

    if speaker == "1" or speaker == "SERVICE":
        df = df[df["Speaker"].values == "SYS"]
        df = df[df["speaker"].values == "SYSTEM"]  # multiwoz
        df = df.reset_index(drop=True)

    df["Speaker"] = df["Speaker"].str.strip()

    # create dialogue_id based on turn_id
    if "dialogue_id" not in df.columns and "turn_id" in df.columns:
        dialog = 0
        result = []
        i_anterior = -1
        for i in df["turn_id"]:
            if i_anterior == -1 or i > i_anterior:
                i_anterior = i
            else:
                dialog = dialog + 1
                i_anterior = -1
            result.append(dialog)
        df["dialogue_id"] = result

    if "turn_id" not in df.columns and "dialogue_id" in df.columns:
        df["turn_id"] = df.groupby("dialogue_id").cumcount()

    # create variable which is a incremental sequence by number of utterances
    df["sequence"] = [i for i in range(len(df))]

    return df


def set_labels(normalized_df, problem=PROBLEM):
    if "trueLabel" in normalized_df.columns:
        """Change the labels according to the problem to be treated."""
        normalized_df["trueLabel"] = normalized_df["trueLabel"].fillna("none")
        normalized_df["trueLabel"] = normalized_df["trueLabel"].astype(str)
        print(normalized_df["trueLabel"].value_counts())

        set_of_labels = set()
    elif "trueLabel" not in normalized_df.columns:
        print("O dataset escolhido não tem labels originais.")

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

    numberLabels = len(set_of_labels)

    print(set_of_labels)
    return set_of_labels, numberLabels


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def preProcessingText(normalizedDF):
    dataframe = normalizedDF
    # dataframe['utterance']= dataframe['utterance'].apply(lambda x:remove_punctuation(x))
    dataframe["utterance"] = dataframe["utterance"].apply(lambda x: x.lower())
    re.sub(r"https?://[\n\S]+\b", "<URL>", dataframe["utterance"])
    # dataframe['utterance']= dataframe['utterance'].replace(to_replace = r\"\\d+ \\d+|012\\d{8}|017\\d{8}|017-\\d{3}-\\d{3}\", value = '#PhoneNumber', regex = True")
    # dataframe['utterance']= dataframe['utterance'].replace(to_replace = r\"C.B \\d+, \\d+ [A-Za-z].[A-Za-z]\", value = 'postalcode', regex = True")

    print(dataframe)
    return dataframe


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def inversepurity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    #  Inverse Purity, all you need to do is replace "axis=0" by "axis=1".
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)


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
        "\nEvaluation:\nAccuracy: ",
        metrics.accuracy_score(labels_true, labels_pred),
    )
    print("V-Measure: ", metrics.v_measure_score(labels_true, labels_pred))
    print("Purity: ", purity_score(labels_true, labels_pred))
    print("Inverse Purity: ", inversepurity_score(labels_true, labels_pred))
    purityf1 = (
        purity_score(labels_true, labels_pred)
        + inversepurity_score(labels_true, labels_pred)
    ) / 2
    print("Purity F1: ", purityf1)


def EvaluationMultiLabel(normalizedDF, number_clusters, problem, df_final):
    # define all labels in a single list to get all classes names, if it's multilabel or single label
    setOfLabels = set()
    if problem == "multi":
        # print("Multi-Label Problem")
        for i in range(len(normalizedDF["trueLabel"])):
            for j in normalizedDF["trueLabel"][i].split(","):
                setOfLabels.add(j.split(" ")[0])
                for j in normalizedDF["trueLabel"][i].split("\n"):
                    setOfLabels.add(j.split(",")[0])
        setOfLabels.remove("")
    if problem == "single":
        # print("Single-Label Problem")
        for i in range(len(normalizedDF["trueLabel"])):
            for j in normalizedDF["trueLabel"][i].split("\n"):
                setOfLabels.add(j.split(" ")[0])
    classes = list(setOfLabels)

    # define the true Label in a list
    y_expected = []
    for i in range(len(normalizedDF["trueLabel"])):
        sublist = []
        for j in normalizedDF["trueLabel"][i].split(","):
            if len(j) > 0:
                sublist.append(j)
        y_expected.append(sublist)

    # multilabel Binarizer
    multilabel_binarizer = MultiLabelBinarizer(classes=classes)
    multilabel_binarizer.fit([[classes]])
    y_expected_binarizer = multilabel_binarizer.transform(y_expected)

    # create dataframe with all labels binarized
    df_labels_binarizer = pd.DataFrame(y_expected_binarizer, columns=[classes])

    # create matrix of occurrencies lines -> clusters; rows -> labels
    matriz_contagens = np.zeros((number_clusters, number_clusters))
    for index, row in df_labels_binarizer.iterrows():
        for i in range(len(row.values)):
            if row.values[i] == 1:
                matriz_contagens[df_final["cluster"][index], i] += 1

    # create dataframe with counts (clusters, labels)
    df_countLabels = pd.DataFrame(matriz_contagens, columns=[classes])

    df_countLabels = df_countLabels.reset_index(drop=True)
    df_countLabels = pd.DataFrame(df_countLabels.to_records())
    df_countLabels = df_countLabels.drop(columns=["index"])
    df_countLabels.columns = df_countLabels.columns.str.replace(
        "[('')]", "", regex=True
    )

    # look for the max values in each row
    mxs = df_countLabels.eq(df_countLabels.max(axis=1), axis=0)
    # join the column names of the max values of each row into a single string
    df_countLabels["MaxLabel"] = mxs.dot(mxs.columns + "").str.rstrip("")

    def match(row):
        pred_labels = row["predictedLabel"].rstrip(",").split(",")
        true_labels = row["trueLabel"].rstrip(",").split(",")
        for label in pred_labels:
            if label in true_labels:
                return True
        return False

    def copy_value_or_null(row):
        if row["PartialMatch"]:
            return row["predictedLabel"]
        else:
            return "null"

    if problem == "single":
        df_final["predictedLabel"] = df_final["predictedLabel"].replace(
            ",", "", regex=True
        )
        df_final["ExactlyMatch"] = df_final.predictedLabel.eq(df_final.trueLabel)
        true_count = (df_final["ExactlyMatch"]).value_counts()[True]
        false_count = (df_final["ExactlyMatch"]).value_counts()[False]
        total_count = true_count + false_count
        accuracy_score = true_count / total_count
        print("Evaluation:\nExact Accuracy: ", accuracy_score)
        print(
            "Exact V-Measure: ",
            metrics.v_measure_score(df_final["trueLabel"], df_final["predictedLabel"]),
        )

    if problem == "multi":
        df_final["predictedLabel"] = df_final["cluster"].map(df_countLabels["MaxLabel"])
        df_final["ExactMatch"] = df_final.predictedLabel.eq(df_final.trueLabel)
        true_count = (df_final["ExactMatch"]).value_counts()[True]
        false_count = (df_final["ExactMatch"]).value_counts()[False]
        total_count = true_count + false_count
        accuracy_score = true_count / total_count
        print("Evaluation:\nExact Accuracy: ", accuracy_score)
        print(
            "Exact V-Measure: ",
            metrics.v_measure_score(df_final["trueLabel"], df_final["predictedLabel"]),
        )

        df_final["PartialMatch"] = df_final.apply(match, axis=1)
        true_count_partial = (df_final["PartialMatch"]).value_counts()[True]
        false_count_partial = (df_final["PartialMatch"]).value_counts()[False]
        total_count_partial = true_count_partial + false_count_partial
        accuracy_score_partial = true_count_partial / total_count_partial
        print("Partial Accuracy: ", accuracy_score_partial)
        # df_final['PartialLabel'] = df_final['PartialMatch'].values
        df_final["PartialLabel"] = df_final.apply(copy_value_or_null, axis=1)
        print(
            "Partial V-Measure: ",
            metrics.v_measure_score(df_final["trueLabel"], df_final["PartialLabel"]),
        )


def match(row):
    pred_labels = row["predictedLabel"].rstrip(",").split(",")
    true_labels = row["trueLabel"].rstrip(",").split(",")
    for label in pred_labels:
        if label in true_labels:
            return True
    return False


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
    nlp = load_spacy(package)
    stopwordseng = nltk.corpus.stopwords.words("english")

    data = "/app/data/MultiWoz22_DAs.csv"
    corpus = data
    user = "both_separately"  # or both_separetely
    if corpus == data:
        problem = "multi"
    else:
        problem = "single"

    if user == "both" or user == "BOTH":
        # normalize turn_id, regex if true normalize URL's and Usernames started with '@', remove greeting words
        normalized_df = normalize_dataset(
            df, regex=True, removeGreetings=False, speaker="both"
        )
        print(normalized_df)
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
        normalized_df = normalized_df.assign(cluster=y_predicted)

        # só caso exista labels
        if "trueLabel" in normalized_df.columns:
            # nº de Dialog Acts / Intents reais -> usado para o t-SNE e PCA
            # nDAsInt = normalizedDF['trueLabel'].nunique()
            set_of_labels, numberLabels = set_labels(normalized_df, problem)
            evaluation(y_predicted, normalized_df, set_of_labels, n_clusters)
        elif "trueLabel" not in normalized_df.columns:
            print("O dataset escolhido não tem labels originais.")

        # Describe Clusters -> Bigrams | Verbs | Closest Document
        if labels_type == "verbs":
            verbs = describe_clusters_verbs(nlp, n_clusters, normalized_df, y_predicted)
            print(verbs)
            labels_type = verbs["labels"].to_dict()

        elif labels_type == "closestDocuments":
            closest_documents = describe_clusters_closest(
                n_clusters, normalized_df, y_predicted, vectors, centers
            )
            print(closest_documents)
            labels_type = closest_documents["labels"].to_dict()

        elif labels_type == "bigrams":
            bigrams = describe_clusters_bigrams(n_clusters, normalized_df, y_predicted)
            print(bigrams)
            labels_type = bigrams["labels"].to_dict()

        # criar as labels para os clusters, quando se faz system e utilizador juntos
        # get list of values
        listValuesBoth = list(labels_type.values())
        listValuesClusters = listValuesBoth
        # print(listValuesClusters)
        names = []
        names = listValuesClusters
        word1 = "SOD"
        word2 = "EOD"
        ttl = word1, word2
        names.extend(ttl)

        # matriz criada para os dois juntos (system + user)
        df_final = normalized_df
        numberLabelsMatrix = n_clusters + 2  # +2 (EOD & SOD)
        occurrence_matrix = np.zeros(
            (numberLabelsMatrix, numberLabelsMatrix)
        )  # nclusters(system+user) + 2 (para EOD e SOD)
        # CRIAR MATRIZ DE OCORRÊNCIAS
        print("Columns normalized", normalized_df.columns)
        print("Columns df_final", df_final.columns)
        normalized_df.sort_values(by=["sequence"], inplace=True)
        df_final.sort_values(by=["sequence"], inplace=True)
  
        for initial_cluster_id in df_final.groupby("dialogue_id").first()[
            "cluster"
        ]:
            occurrence_matrix[n_clusters, initial_cluster_id] = (
                occurrence_matrix[n_clusters, initial_cluster_id] + 1
            )

        for final_cluster_id in df_final.groupby("dialogue_id").last()["cluster"]:
            occurrence_matrix[final_cluster_id, n_clusters + 1] = (
                occurrence_matrix[final_cluster_id, n_clusters + 1] + 1
            )

        # Find the sum of each row
        row_sums = occurrence_matrix.sum(axis=1)

        # Divide each element in the matrix by the corresponding row sum
        transition_matrix = np.divide(occurrence_matrix, row_sums[:, np.newaxis])

        matrix = pd.DataFrame(transition_matrix, index=names, columns=names)
        matrix = matrix.round(decimals=2)
        matrix = matrix.fillna(0.00)

        return generate_markov_chain(n_clusters, matrix)  # both

    elif user == "both_separately" or user == "BOTH_SEPARATELY":
        # normalize turn_id, regex if true normalize URL's and Usernames started with '@', remove greeting words
        normalized_df = normalize_dataset(
            df, regex=True, removeGreetings=False, speaker="both"
        )
        normalized_df_user = normalized_df[normalized_df["Speaker"] == "USER"]
        normalized_df_system = normalized_df[normalized_df["Speaker"] == "SYSTEM"]

        # Functions to transform into vectors (WORD EMBEDDING)
        if representation == "tfidf":
            topic_features = topic_features_to_remove(
                nlp, normalized_df, NUMBER_OF_FEATURES, topic_feature=False
            )  # topic features
            vectors_user = preprocessing_tfidf(
                stopwordseng, normalized_df_user, topic_features, 0.8, 3
            )

            vectors_system = preprocessing_tfidf(
                stopwordseng, normalized_df_system, topic_features, 0.8, 3
            )
        elif representation == "word2vec":
            vectors_user = word2vec(nlp, normalized_df_user)
            vectors_system = word2vec(nlp, normalized_df_system)
        elif representation == "sentenceTransformer":
            model = SentenceTransformer(MODEL_NAME, cache_folder=model_cache_folder)
            vectors_user = use_sentence_transformer(normalized_df_user, model)
            vectors_system = use_sentence_transformer(normalized_df_system, model)
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
        y_predicted_user, centers_user = clustering_kmeans(vectors_user, n_clusters)
        y_predicted_system, centers_system = clustering_kmeans(
            vectors_system, n_clusters
        )
        normalized_df_user = normalized_df_user.assign(cluster=y_predicted_user)
        normalized_df_system = normalized_df_system.assign(
            cluster=y_predicted_system + n_clusters
        )

        df_final = pd.concat([normalized_df_user, normalized_df_system])

        # só caso exista labels
        if "trueLabel" in df_final.columns:
            # nº de Dialog Acts / Intents reais -> usado para o t-SNE e PCA
            # nDAsInt = normalizedDF['trueLabel'].nunique()
            set_of_labels, numberLabels = set_labels(df_final, problem)
            evaluation(y_predicted_user, normalized_df_user, set_of_labels, n_clusters)
            evaluation(
                y_predicted_system, normalized_df_system, set_of_labels, n_clusters
            )
        elif "trueLabel" not in df_final.columns:
            print("O dataset escolhido não tem labels originais.")

        # Describe Clusters -> Bigrams | Verbs | Closest Document
        if labels_type == "verbs":
            verbs_user = describe_clusters_verbs(
                nlp, n_clusters, normalized_df_user, y_predicted_user
            )
            verbs_system = describe_clusters_verbs(
                nlp, n_clusters, normalized_df_system, y_predicted_system
            )
            labels_type_user = verbs_user["labels"].to_dict()
            labels_type_system = verbs_system["labels"].to_dict()

        elif labels_type == "closestDocuments":
            closest_documents_user = describe_clusters_closest(
                nlp, n_clusters, normalized_df_user, y_predicted_user
            )
            closest_documents_system = describe_clusters_closest(
                nlp, n_clusters, normalized_df_system, y_predicted_system
            )
            labels_type_user = closest_documents_user["labels"].to_dict()
            labels_type_system = closest_documents_system["labels"].to_dict()

        elif labels_type == "bigrams":
            bigrams_user = describe_clusters_bigrams(
                n_clusters, normalized_df_user, y_predicted_user
            )
            bigrams_system = describe_clusters_bigrams(
                n_clusters, normalized_df_system, y_predicted_system
            )
            labels_type_user = bigrams_user["labels"].to_dict()
            labels_type_system = bigrams_system["labels"].to_dict()

        # criar as labels para os clusters, quando se faz system e utilizador juntos
        # get list of values
        listValuesUsers = list(labels_type_user.values())
        listValuesSystems = list(labels_type_system.values())

        # add prefix user and sys
        prefix_user = "user -> "
        listValuesUsers = list(
            map(lambda element: prefix_user + "" + element, listValuesUsers)
        )
        prefix_system = "sys -> "
        listValuesSystems = list(
            map(lambda element: prefix_system + "" + element, listValuesSystems)
        )
        listValuesClusters = listValuesUsers + listValuesSystems

        names = []
        names = listValuesClusters
        word1 = "SOD"
        word2 = "EOD"
        ttl = word1, word2
        names.extend(ttl)

        # matriz criada para os dois juntos (system + user)
        nClustersBoth = n_clusters + n_clusters
        numberLabelsMatrix = nClustersBoth + 2  # +2 (EOD & SOD)
        occurrence_matrix = np.zeros(
            (numberLabelsMatrix, numberLabelsMatrix)
        )  # nclusters(system+user) + 2 (para EOD e SOD)
        # CRIAR MATRIZ DE OCORRÊNCIAS
        print("DF_FINAL", df_final.columns)
        df_final.sort_values(by=["sequence"], inplace=True)

        for initial_cluster_id in df_final.groupby("dialogue_id").first()[
            "cluster"
        ]:
            occurrence_matrix[n_clusters, initial_cluster_id] = (
                occurrence_matrix[n_clusters, initial_cluster_id] + 1
            )

        for final_cluster_id in df_final.groupby("dialogue_id").last()["cluster"]:
            occurrence_matrix[final_cluster_id, n_clusters + 1] = (
                occurrence_matrix[final_cluster_id, n_clusters + 1] + 1
            )

        # Find the sum of each row
        row_sums = occurrence_matrix.sum(axis=1)

        # Divide each element in the matrix by the corresponding row sum
        transition_matrix = np.divide(occurrence_matrix, row_sums[:, np.newaxis])

        matrix = pd.DataFrame(transition_matrix, index=names, columns=names)
        matrix = matrix.round(decimals=2)
        matrix = matrix.fillna(0.00)

        return generate_markov_chain_separately(n_clusters, matrix)  # both_separetely

    return None
