import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer


def describe_clusters_bigrams(n_clusters, normalized_df, y_predicted):
    df_teste = pd.DataFrame()  # utterance and y_predicted
    df_teste["predicted"] = y_predicted
    df_teste["corpus"] = normalized_df["utterance"]

    list_clusters = []  # clusters
    list_bigrams = []  # bigrams

    for p in range(n_clusters):
        predicted_df = df_teste[df_teste["predicted"] == p]  # mudar numeros
        corpus = predicted_df["corpus"]

        df_corpus = pd.DataFrame(corpus)
        df_corpus.columns = ["docs_in_cluster"]

        c_vec = CountVectorizer(stop_words=None, ngram_range=(2, 2))

        # matrix of ngrams
        ngrams = c_vec.fit_transform(df_corpus["docs_in_cluster"].astype("U"))

        # count frequency of ngrams
        count_values = ngrams.toarray().sum(axis=0)

        # list of ngrams
        vocab = c_vec.vocabulary_

        df_ngram = pd.DataFrame(
            sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)
        ).rename(columns={0: "frequency", 1: "labels"})
        list_clusters.append(f"Cluster {p}")
        list_bigrams.append(df_ngram["labels"].iat[0])

    df1 = pd.DataFrame(list_clusters, columns=["clusters"])
    df2 = pd.DataFrame(list_bigrams, columns=["labels"])
    df2["labels"] = df2["labels"].str.cat(
        df2.groupby("labels").cumcount().astype(str).str.replace("0", ""), sep=" "
    )  # handle repeated labelled

    df_labels = pd.concat([df1, df2], axis=1)
    print("Cluster Describe Bigrams:")

    return df_labels


def describe_clusters_verbs(nlp, n_clusters, normalized_df, y_predicted):
    df_teste = pd.DataFrame()  # utterance and y_predicted
    df_teste["predicted"] = y_predicted
    df_teste["corpus"] = normalized_df["utterance"]
    df_teste["corpus"] = df_teste["corpus"].astype(str)  # ?

    list_clusters = []  # clusters
    list_verbs = []  # verbs
    list_sintagmas = []

    for utt in list(df_teste["corpus"]):
        doc = nlp(utt)
        sintagma = " "
        for token in doc:
            if token.text[0] >= "A" and token.text[0] <= "z":
                if token.dep_ == "ROOT":
                    sintagma = sintagma + " " + token.lemma_
                    for child in token.children:
                        if (
                            child.text[0] >= "A"
                            and child.text[0] <= "z"
                            and child.dep_ != "nsubj"
                        ):
                            sintagma = sintagma + " " + child.lemma_
        list_sintagmas.append(sintagma)

    df_teste["corpus"] = list_sintagmas

    # remove white space, with this verbs technic happen to much
    df_teste["corpus"].replace(" ", np.nan, inplace=True)
    df_teste.dropna()

    for p in range(n_clusters):
        predicted_df = df_teste[df_teste["predicted"] == p]  # mudar numeros

        corpus = predicted_df["corpus"]

        df_corpus = pd.DataFrame(corpus)
        df_corpus.columns = ["docs_in_cluster"]

        # count frequency of verbs
        count_values = df_corpus.value_counts().index.tolist()[0]

        list_clusters.append(f"Cluster {p}")
        list_verbs.append(count_values[0])

    df1 = pd.DataFrame(list_clusters, columns=["clusters"])
    df2 = pd.DataFrame(list_verbs, columns=["labels"])
    df2["labels"] = df2["labels"].str.cat(
        df2.groupby("labels").cumcount().astype(str).str.replace("0", ""), sep=" "
    )  # handle repeated labelled

    df_labels = pd.concat([df1, df2], axis=1)

    print("Cluster Describe Verbs:")

    return df_labels


def describe_clusters_closest(n_clusters, normalized_df, y_predicted, vectors, centers):
    docs = normalized_df["utterance"]
    # order_centroids = centers.argsort()[:, ::-1]
    closest, _ = metrics.pairwise_distances_argmin_min(
        centers, vectors, metric="cosine"
    )
    # mydict = {i: np.where(y_predicted == i)[0] for i in range(len(set(y_predicted)))}

    df_teste = pd.DataFrame()  # utterance and y_predicted
    df_teste["predicted"] = y_predicted
    df_teste["corpus"] = normalized_df["utterance"]

    list_clusters = []  # clusters
    list_closest_doc = []  # closest document

    for p in range(n_clusters):
        predicted_df = df_teste[df_teste["predicted"] == p]  # mudar numeros
        corpus = predicted_df["corpus"]

        df_corpus = pd.DataFrame(corpus)
        df_corpus.columns = ["docs_in_cluster"]

        # count frequency of verbs
        # count_values = df_corpus.value_counts().index.tolist()[0]

        list_clusters.append(f"Cluster {p}")
        list_closest_doc.append(docs[closest[p]])

    df1 = pd.DataFrame(list_clusters, columns=["clusters"])
    df2 = pd.DataFrame(list_closest_doc, columns=["labels"])
    df2["labels"] = df2["labels"].str.cat(
        df2.groupby("labels").cumcount().astype(str).str.replace("0", ""), sep=" "
    )  # handle repeated labelled

    df_labels = pd.concat([df1, df2], axis=1)

    print("Cluster Describe Closest Document:")
    return df_labels
