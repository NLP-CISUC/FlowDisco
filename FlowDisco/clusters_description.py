import numpy as np
import pandas as pd
from sklearn import metrics
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

def describe_clusters_kBERT(n_clusters, normalized_df, y_predicted, stopwords, n_grams):
    docs = normalized_df["utterance"]
    doc = docs.to_string()
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, n_grams), stop_words=stopwords, highlight=False, top_n=n_clusters)

    keywords_list= list(dict(keywords).keys())

    df_teste = pd.DataFrame()  # utterance and y_predicted
    df_teste["predicted"] = y_predicted
    df_teste["corpus"] = normalized_df["utterance"]

    list_clusters = []  # clusters
    list_kBERT = []  # kBERT

    for p in range(n_clusters):
        predicted_df = df_teste[df_teste["predicted"] == p]  # mudar numeros
        corpus = predicted_df["corpus"]

        df_corpus = pd.DataFrame(corpus)
        df_corpus.columns = ["docs_in_cluster"]

        list_clusters.append(f"Cluster {p}")
        dataframe_output = pd.DataFrame(keywords, columns=['Output', 'temp'])
        list_kBERT.append(dataframe_output["Output"].iat[p])

    df1 = pd.DataFrame(list_clusters, columns=["clusters"])
    df2 = pd.DataFrame(list_kBERT, columns=["labels"])
    df2["labels"] = df2["labels"].str.cat(
        df2.groupby("labels").cumcount().astype(str).str.replace("0", ""), sep=" "
    )  # handle repeated labelled

    df_labels = pd.concat([df1, df2], axis=1)

    print("Cluster Describe KBERT")
    print(df_labels)

    return df_labels

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
        n_grams = c_vec.fit_transform(df_corpus["docs_in_cluster"].astype('U'))

        # count frequency of ngrams
        count_values = n_grams.toarray().sum(axis=0)

        # list of ngrams
        vocab = c_vec.vocabulary_

        df_ngram = pd.DataFrame(
            sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)).rename(columns={0: "frequency", 1: "labels"})
        list_clusters.append(f"Cluster {p}")
        list_bigrams.append(df_ngram["labels"].iat[0])

    df1 = pd.DataFrame(list_clusters, columns=["clusters"])
    df2 = pd.DataFrame(list_bigrams, columns=["labels"])
    df2["labels"] = df2["labels"].str.cat(
        df2.groupby("labels").cumcount().astype(str).str.replace("0", ""), sep=" "
    )  # handle repeated labelled

    df_labels = pd.concat([df1, df2], axis=1)
    print("Cluster Describe Bigrams")
    print(df_labels)

    return df_labels

def describe_clusters_verbs(nlp, n_clusters, normalized_df, y_predicted):
    df_teste = normalized_df.copy()
    df_teste["predicted"] = y_predicted

    df_teste["utterance"] = df_teste["utterance"].astype(str)
    list_clusters = []
    list_verbs = []
    list_sintagmas = []

    for utt in list(df_teste["utterance"]):
        doc = nlp(utt)
        sintagma = " "
        for token in doc:
            if token.text[0] >= 'A' and token.text[0] <= 'z' or token.text[0] == '@':
                if token.dep_ == 'ROOT':
                    sintagma = sintagma + " " + token.lemma_
                    for child in token.children:
                        if (
                            (child.text[0] >= 'A'
                            and child.text[0] <= 'z' or child.text[0] == '@')
                            and child.dep_ != "nsubj" 
                        ):
                            sintagma = sintagma + " " + child.lemma_
    
        list_sintagmas.append(sintagma)

    df_teste["utterance"] = list_sintagmas

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

        list_clusters.append(f"Cluster {p}")
        list_verbs.append(count_values)

    df1 = pd.DataFrame(list_clusters, columns=["clusters"])
    df2 = pd.DataFrame(list_verbs, columns=["labels"])
    df2["labels"] = df2["labels"].str.cat(
        df2.groupby("labels").cumcount().astype(str).str.replace("0", ""), sep=" "
    )

    df_labels = pd.concat([df1, df2], axis=1)

    print("Cluster Describe Verbs")
    print(df_labels)
    return df_labels
	
def describe_clusters_closest(normalized_df, y_predicted, vectors, centers, n_clusters):
    docs = normalized_df["utterance"]
    order_centroids = centers.argsort()[:, ::-1]
    closest, _ = metrics.pairwise_distances_argmin_min(
        centers, vectors, metric="cosine"
    )
    mydict = {i: np.where(y_predicted == i)[0] for i in range(len(set(y_predicted)))}

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
        #count_values = df_corpus.value_counts().index.tolist()[0]

        list_clusters.append(f"Cluster {p}")
        list_closest_doc.append(docs[closest[p]])

    df1 = pd.DataFrame(list_clusters, columns=["clusters"])
    df2 = pd.DataFrame(list_closest_doc, columns=["labels"])
    df2["labels"] = df2["labels"].str.cat(
        df2.groupby("labels").cumcount().astype(str).str.replace("0", ""), sep=" "
    )  # handle repeated labelled

    df_labels = pd.concat([df1, df2], axis=1)
    
    print("Cluster Describe Closest Document")
    print(df_labels)
    return df_labels

#def describe_clusters_closest(normalized_df, y_predicted, vectors, centers, n_clusters):

# find the nearest instance of centroid for each cluster
# nearest_instances, _ = pairwise_distances_argmin_min(vectors_user, centers_user)

# # find the nearest instance to each centroid for each cluster
# nearest_instances = []
# for i in range(nClustersSystem):
#     cluster_indices = np.where(df_user['cluster'] == i)[0]
#     distances = pairwise_distances_argmin_min(vectors_system[cluster_indices], [centers_system[i]])
#     nearest_instance_index = cluster_indices[distances[0][0]]
#     nearest_instances.append(nearest_instance_index)

# # print the nearest instance to each centroid, along with the original utterance
# for i, instance_index in enumerate(nearest_instances):
#     cluster_label = i
#     df_cluster = df_user[df_user["cluster"] == cluster_label]
#     num_rows = len(df_cluster)
#     print("Cluster {}: num rows = {}, instance_index = {}".format(cluster_label, num_rows, instance_index))
#     if instance_index >= num_rows:
#         print("Error: instance_index out of bounds")
#     else:
#         utterance = "".join(df_cluster.iloc[instance_index]["utterance"])
#         print("Nearest instance to centroid of cluster {}:".format(cluster_label))
#         print("Utterance: {}".format(utterance))

# print("Nearest instance to each centroid for each cluster:", nearest_instances)
