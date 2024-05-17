import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import spacy
import re
import string
from sentence_transformers import SentenceTransformer
import statistics
import pickle
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
#Added in 2023/05/09
from keybert import KeyBERT
#---------------------------
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from plataformateste.clustering import clustering_kmeans
from plataformateste.vector_representation import use_sentence_transformer
from plataformateste.clusters_description import describe_clusters_bigrams
from plataformateste.clusters_description import (
    describe_clusters_bigrams,
    describe_clusters_closest,
    describe_clusters_verbs,
    describe_clusters_kBERT
)
from plataformateste.graphs import generate_markov_chain, generate_markov_chain_separately
from plataformateste.vector_representation import (
    preprocessing_tfidf,
    topic_features_to_remove,
    use_sentence_transformer,
    word2vec,
)

PROBLEM = "multi"
NUMBER_OF_FEATURES = 100
CLUSTER_OPT = "equal"
nomeFichPickleUser = 'kmeans_user.pkl'
nomeFichPickleSystem = 'kmeans_system.pkl'

#Model for English
# Models of SentenceTransformer choosed by results here: 'https://www.sbert.net/docs/pretrained_models.html'
# options: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'all-distilroberta-v1', 'all-MiniLM-L12-v2'
MODEL_NAME = "all-MiniLM-L6-v2"

#Model for Portuguese
#MODEL_NAME = "rufimelo/bert-large-portuguese-cased-sts"

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
    user_tags_pattern = "^@\\S+"
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
        df['Speaker'] = df['Speaker'].replace('SERVICE', 'SYSTEM')
        df['Speaker'] = df['Speaker'].replace('SYS', 'SYSTEM')        

    df['Speaker'] = df['Speaker'].str.strip()

     #create dialogue_id based on turn_id
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

    if 'turn_id' not in df.columns and 'dialogue_id' in df.columns:
        df['turn_id'] = df.groupby('dialogue_id').cumcount()

    #create variable which is a incremental sequence by number of utterances
    df['sequence'] = [i for i in range(len(df))]

    return df

def set_labels(normalized_df, problem=PROBLEM):
    if "trueLabel" in normalized_df.columns:
        """Change the labels according to the problem to be treated."""
        normalized_df["trueLabel"] = normalized_df["trueLabel"].fillna("none")
        normalized_df["trueLabel"] = normalized_df["trueLabel"].astype(str)
        set_of_labels = set()
    elif "trueLabel" not in normalized_df.columns:
        print("O dataset escolhido não tem labels originais.") 

    if problem == "multi":
        for i in range(len(normalized_df["trueLabel"])):
            for j in normalized_df["trueLabel"][i].split("\n"):
                set_of_labels.add(j.split(",")[0])
    elif problem == "single":
        for i in range(len(normalized_df["trueLabel"])):
            for j in normalized_df["trueLabel"][i].split("\n"):
                set_of_labels.add(j.split(" ")[0])
    else:
        raise ValueError
    
    numberLabels = len(set_of_labels)

    print(set_of_labels)
    return set_of_labels, numberLabels

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def preProcessingText(normalizedDF):
  dataframe = normalizedDF
  dataframe['utterance']= dataframe['utterance'].apply(lambda x: x.lower())
  re.sub(r'https?://[\n\S]+\b', '<URL>', dataframe['utterance'])
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
    print('Purity: ', purity_score(labels_true, labels_pred))
    print('Inverse Purity: ', inversepurity_score(labels_true, labels_pred))
    purityf1 = (purity_score(labels_true, labels_pred)+inversepurity_score(labels_true, labels_pred))/2
    print('Purity F1: ', purityf1)

def EvaluationMultiLabel(normalizedDF, number_clusters, problem, df_final):
  # define all labels in a single list to get all classes names, if it's multilabel or single label
  setOfLabels = set()
  if problem == "multi":
      #print("Multi-Label Problem")
      for i in range(len(normalizedDF['trueLabel'])):
          for j in normalizedDF['trueLabel'][i].split(','):
            setOfLabels.add(j.split(" ")[0])
            for j in normalizedDF['trueLabel'][i].split('\n'):
              setOfLabels.add(j.split(",")[0])
      setOfLabels.remove("")
  if problem == "single":
    #print("Single-Label Problem")
    for i in range(len(normalizedDF['trueLabel'])):
      for j in normalizedDF['trueLabel'][i].split('\n'):
        setOfLabels.add(j.split(" ")[0])
  classes = list(setOfLabels)
  
  # define the true Label in a list
  y_expected = []
  for i in range(len(normalizedDF['trueLabel'])):
    sublist = []
    for j in normalizedDF['trueLabel'][i].split(','):
      if(len(j)>0):
        sublist.append(j)
    y_expected.append(sublist)

  # multilabel Binarizer
  multilabel_binarizer = MultiLabelBinarizer(classes=classes)
  multilabel_binarizer.fit([[classes]])
  y_expected_binarizer = multilabel_binarizer.transform(y_expected)

  # create dataframe with all labels binarized
  df_labels_binarizer = pd.DataFrame(y_expected_binarizer, columns = [classes])

  # create matrix of occurrencies lines -> clusters; rows -> labels
  matriz_contagens = np.zeros((number_clusters, number_clusters)) 
  for index, row in df_labels_binarizer.iterrows():
    for i in range(len(row.values)):
      if row.values[i] == 1:
        matriz_contagens[df_final['n_clusters_final'][index], i] += 1
      
  #create dataframe with counts (clusters, labels)
  df_countLabels = pd.DataFrame(matriz_contagens, columns = [classes])

  df_countLabels = df_countLabels.reset_index(drop=True)
  df_countLabels = pd.DataFrame(df_countLabels.to_records())
  df_countLabels = df_countLabels.drop(columns=["index"])
  df_countLabels.columns = df_countLabels.columns.str.replace("[('')]", "", regex=True)

  # look for the max values in each row
  mxs = df_countLabels.eq(df_countLabels.max(axis=1), axis=0)
  # join the column names of the max values of each row into a single string
  df_countLabels['MaxLabel'] = mxs.dot(mxs.columns + '').str.rstrip('')

  def match(row):
    pred_labels = row['predictedLabel'].rstrip(',').split(',')
    true_labels = row['trueLabel'].rstrip(',').split(',')
    for label in pred_labels:
        if label in true_labels:
            return True
    return False

  def copy_value_or_null(row):
    if row['PartialMatch']:
        return row['predictedLabel']
    else:
        return 'null'

  if problem == "single":
    df_final['predictedLabel'] = df_final['predictedLabel'] .replace(',','', regex=True)
    df_final['ExactlyMatch'] = df_final.predictedLabel.eq(df_final.trueLabel)
    true_count = (df_final['ExactlyMatch']).value_counts()[True]
    false_count = (df_final['ExactlyMatch']).value_counts()[False]
    total_count = true_count + false_count
    accuracy_score = true_count/total_count
    print("Evaluation:\nExact Accuracy: ", accuracy_score)
    print('Exact V-Measure: ', metrics.v_measure_score(df_final['trueLabel'], df_final['predictedLabel']))

  if problem == "multi":
    df_final['predictedLabel'] = df_final['cluster'].map(df_countLabels['MaxLabel'])
    df_final['ExactMatch'] = df_final.predictedLabel.eq(df_final.trueLabel)
    true_count = (df_final['ExactMatch']).value_counts()[True]
    false_count = (df_final['ExactMatch']).value_counts()[False]
    total_count = true_count + false_count
    accuracy_score = true_count/total_count
    print("Evaluation:\nExact Accuracy: ", accuracy_score)
    print('Exact V-Measure: ', metrics.v_measure_score(df_final['trueLabel'], df_final['predictedLabel']))

    df_final['PartialMatch'] = df_final.apply(match, axis=1)
    true_count_partial = (df_final['PartialMatch']).value_counts()[True]
    false_count_partial = (df_final['PartialMatch']).value_counts()[False]
    total_count_partial = true_count_partial + false_count_partial
    accuracy_score_partial = true_count_partial/total_count_partial
    print("Partial Accuracy: ", accuracy_score_partial)
    #df_final['PartialLabel'] = df_final['PartialMatch'].values
    df_final['PartialLabel'] = df_final.apply(copy_value_or_null, axis=1)
    print('Partial V-Measure: ', metrics.v_measure_score(df_final['trueLabel'], df_final['PartialLabel']))

def match(row):
    pred_labels = row['predictedLabel'].rstrip(',').split(',')
    true_labels = row['trueLabel'].rstrip(',').split(',')
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
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=2)
        kmeans.fit(scaled_data)
        score = metrics.silhouette_score(scaled_data, kmeans.labels_)
        silhouettes.append(score)
        print(score)
        if score > atual_silhouette:
            number_clusters = k
            atual_silhouette = score

    # Plot the results
    # plt.figure(figsize=(20, 5))
    # plt.plot(n_clusters_axis, silhouettes)
    # plt.xlabel("Values of K")
    # plt.ylabel("Silhouette score")
    # plt.title("Silhouette Method")
    # plt.savefig("teste.png", format="png")
    # plt.show()
    # plt.close()

    scatter_plot = sns.scatterplot(
        x=n_clusters_axis, y=silhouettes)
 
    # use get_figure function and store the plot i
    # n a variable (scatter_fig)
    scatter_fig = scatter_plot.get_figure()
    
    # use savefig function to save the plot and give
    # a desired name to the plot.
    scatter_fig.savefig('scatterplot.png')

    return number_clusters

def run_test(
    df: pd.DataFrame,
    package: str,
    representation: str,
    labels_type: str,
    n_clusters: int = 0,
    n_grams: int=0,
    model_cache_folder: str = None
):

    nlp = spacy.load(package)
    stopwords = nltk.corpus.stopwords.words("english")
    #stopwords = nltk.corpus.stopwords.words("portuguese")
    data = "/app/data/MultiWOZ_DAs.csv"
    corpus = data
    user = "both_separately" #both or both_separately
    if corpus == data:
        problem = "single"
    else:
        problem = "multi"

    #df = df.loc[df['Acc_Query'] == 'VODAFONE']
    #df = df[df['Acc_Query'].str.contains('RTP')]
    #df = df[df['Acc_Query'] == 'VODAFONE']
    if user == "both" or user == "BOTH":
        # normalize turn_id, regex if true normalize URL's and Usernames started with '@', remove greeting words
        normalized_df = normalize_dataset(df, regex=True, removeGreetings=False, speaker='both')
        
        model = SentenceTransformer(MODEL_NAME, cache_folder=model_cache_folder)

        # Functions to transform into vectors (WORD EMBEDDING)
        if representation == "tfidf":
            topic_features = topic_features_to_remove(
                nlp, normalized_df, NUMBER_OF_FEATURES, topic_feature=False
            )  # topic features
            vectors = preprocessing_tfidf(
                stopwords, normalized_df, topic_features, 0.8, 3
            )
        elif representation == "word2vec":
            vectors = word2vec(nlp, normalized_df)
        elif representation == "sentenceTransformer":
            vectors = use_sentence_transformer(normalized_df, model)
        else:
            raise ValueError


        if "trueLabel" in normalized_df.columns:
            # equal | total # nCluster = nDA / Int | total / max
            n_clusters = choose_number_of_clusters(normalized_df)
        elif not n_clusters:
            # unsupervised way (vectors, minK, maxK, increment) sequence of numbers from minK to maxK, but increment
            n_clusters = silhouette_method(vectors, 1, 1, 1)

        print("O número de Clusters a usar será " + str(n_clusters) + "!!")

       # K-Means
        y_predicted, centers = clustering_kmeans(vectors, n_clusters, nomeFichPickleSystem)
        normalized_df = normalized_df.assign(cluster = y_predicted)

        # só caso exista labels
        if "trueLabel" in normalized_df.columns:
            # nº de Dialog Acts / Intents reais -> usado para o t-SNE e PCA
            # nDAsInt = normalizedDF['trueLabel'].nunique()
            set_of_labels, numberLabels = set_labels(normalized_df, problem)
            evaluation(y_predicted, normalized_df, set_of_labels, n_clusters)

        # Describe Clusters -> Bigrams | Verbs | Closest Document
        if labels_type == "verbs":
            verbs = describe_clusters_verbs(nlp, n_clusters, normalized_df, y_predicted)
            print(verbs)
            labels_type = verbs['labels'].to_dict()

        if labels_type == "kBERT":
            kBERT = describe_clusters_kBERT(n_clusters, normalized_df, y_predicted, stopwords, n_grams)
            print(kBERT)
            labels_type = kBERT['labels'].to_dict()

        #elif labels_type == "closestDocuments":
        #    closest_documents = describe_clusters_closest(normalized_df, y_predicted, vectors, centers, n_clusters)
        #    labels_type = closest_documents['labels'].to_dict()

        elif labels_type == "bigrams":
            bigrams = describe_clusters_bigrams(n_clusters, normalized_df, y_predicted)
            print(bigrams)
            labels_type = bigrams['labels'].to_dict()
       
        #criar as labels para os clusters, quando se faz system e utilizador juntos
        # get list of values
        listValuesBoth = (list(labels_type.values()))
        listValuesClusters = listValuesBoth 
        names = []
        names = listValuesClusters
        word1 = 'SOD'
        word2 = 'EOD'
        ttl = word1 , word2
        names.extend(ttl)

        #matriz criada para os dois juntos (system + user)
        df_final = normalized_df
        numberLabelsMatrix = (n_clusters+2) # +2 (EOD & SOD)
        occurrence_matrix = np.zeros((numberLabelsMatrix, numberLabelsMatrix)) #nclusters(system+user) + 2 (para EOD e SOD)
        #CRIAR MATRIZ DE OCORRÊNCIAS

        normalized_df.sort_values(by=['sequence'], inplace=True) 
        df_final.sort_values(by=['sequence'], inplace=True) 

        for i in range(int(normalized_df['dialogue_id'].iat[-1])+1):
            turno_anterior = 0
            fim_dialogo = -1
            inicio_dialogo = df_final['cluster'].loc[(df_final.dialogue_id == i) & (df_final.turn_id == 0)]
            occurrence_matrix[n_clusters, inicio_dialogo] = occurrence_matrix[n_clusters, inicio_dialogo] + 1
            for turno in df_final['turn_id'].loc[(df_final.dialogue_id == i) & (df_final.turn_id != 0)]:
                dialogo = df_final['cluster'].loc[(df_final.dialogue_id == i) & (df_final.turn_id == turno)]
                dialogo_anterior = df_final['cluster'].loc[(df_final.dialogue_id == i) & (df_final.turn_id == turno_anterior)]
                occurrence_matrix[dialogo_anterior, dialogo] = occurrence_matrix[dialogo_anterior, dialogo] + 1
                turno_anterior = turno
                fim_dialogo = dialogo
            occurrence_matrix[fim_dialogo, n_clusters+1] = occurrence_matrix[fim_dialogo, n_clusters+1]  + 1
        
        # Find the sum of each row
        row_sums = occurrence_matrix.sum(axis=1)

        # Divide each element in the matrix by the corresponding row sum
        transition_matrix = occurrence_matrix / (row_sums[:, np.newaxis])
        matrix = pd.DataFrame(transition_matrix, index=names, columns=names)
        matrix = matrix.round(decimals = 2)
        matrix = matrix.fillna(0.00)
     
        return generate_markov_chain(n_clusters, matrix) #both 

    elif user == "both_separately" or user == "BOTH_SEPARATELY":
         # normalize turn_id, regex if true normalize URL's and Usernames started with '@', remove greeting words
        normalized_df = normalize_dataset(df, regex=True, removeGreetings=False, speaker='both')
        normalized_df_user = normalized_df[normalized_df['Speaker'] == 'USER']
        normalized_df_system = normalized_df[normalized_df['Speaker'] == 'SYSTEM']
    
        # Functions to transform into vectors (WORD EMBEDDING)
        model = SentenceTransformer(MODEL_NAME, cache_folder=model_cache_folder)

        if representation == "tfidf":
            topic_features = topic_features_to_remove(
                nlp, normalized_df, NUMBER_OF_FEATURES, topic_feature=False
            )  # topic features
            vectors_user = preprocessing_tfidf(
                stopwords, normalized_df_user, topic_features, 0.8, 3
            )

            vectors_system = preprocessing_tfidf(
                stopwords, normalized_df_system, topic_features, 0.8, 3
            )
        elif representation == "word2vec":
            vectors_user = word2vec(nlp, normalized_df_user)
            vectors_system = word2vec(nlp, normalized_df_system)
        elif representation == "sentenceTransformer":
            vectors_user = use_sentence_transformer(normalized_df_user, model)
            vectors_system = use_sentence_transformer(normalized_df_system, model)
        else:
            raise ValueError

        if "trueLabel" in normalized_df.columns:
            # equal | total # nCluster = nDA / Int | total / max
            n_clusters_user = choose_number_of_clusters(normalized_df_user)
            n_clusters_system = choose_number_of_clusters(normalized_df_system)
        elif not n_clusters:
            # unsupervised way (vectors, minK, maxK, increment) sequence of numbers from minK to maxK, but increment
            #Colocar o int(raiz quadrada de len(vectores user) / 2)
            n_clusters_user = silhouette_method(vectors_user, 2, 24, 1)
            n_clusters_system = silhouette_method(vectors_system, 2, 24, 1)

        if n_clusters_user == 0: 
            n_clusters_user = n_clusters_system
        if n_clusters_system == 0:
            n_clusters_system = n_clusters_user
        if n_clusters_user == 0 and n_clusters_system == 0:
            n_clusters_user == 5
            n_clusters_system == 5

        print("Número de Clusters para USER: " + str(n_clusters_user))
        print("Número de Clusters para SYSTEM: " + str(n_clusters_system))
       # K-Means
        y_predicted_user, centers_user = clustering_kmeans(vectors_user, n_clusters_user, nomeFichPickleUser)
        y_predicted_system, centers_system = clustering_kmeans(vectors_system, n_clusters_system, nomeFichPickleSystem)
        
        normalized_df_user = normalized_df_user.assign(cluster_user = y_predicted_user)
        normalized_df_system = normalized_df_system.assign(cluster_system = y_predicted_system)
        df_final = pd.concat([normalized_df_user, normalized_df_system])

        print("df_final", df_final)

        # só caso exista labels
        if "trueLabel" in df_final.columns:
            # nº de Dialog Acts / Intents reais -> usado para o t-SNE e PCA
            # nDAsInt = normalizedDF['trueLabel'].nunique()
            set_of_labels = set_labels(df_final, problem)
            #evaluation(y_predicted_user, normalized_df_user, set_of_labels, n_clusters_user)
            #evaluation(y_predicted_system, normalized_df_system, set_of_labels, n_clusters_system)
            #EvaluationMultiLabel(normalized_df_user, n_clusters_user, problem, df_final)
            #EvaluationMultiLabel(normalized_df_system, n_clusters_system, problem, df_final)
        elif "trueLabel" not in df_final.columns:
            print("O dataset escolhido não tem labels originais.")

        # Describe Clusters -> Bigrams | Verbs | Closest Document
        if labels_type == "verbs":
            verbs_user = describe_clusters_verbs(nlp, n_clusters_user, normalized_df_user, y_predicted_user)
            verbs_system = describe_clusters_verbs(nlp, n_clusters_system, normalized_df_system, y_predicted_system)
            labels_type_user = verbs_user['labels'].to_dict()
            labels_type_system = verbs_system['labels'].to_dict()
            print("labels_type_user", labels_type_user)
            print("labels_type_system", labels_type_system)
        #elif labels_type == "closestDocuments":
        #    closest_documents_user = describe_clusters_closest(nlp, n_clusters, normalized_df_user, y_predicted_user)
        #    closest_documents_system = describe_clusters_closest(nlp, n_clusters, normalized_df_system, y_predicted_system)
        #    labels_type_user = closest_documents_user['labels'].to_dict()
        #    labels_type_system = closest_documents_system['labels'].to_dict()

        elif labels_type == "bigrams":
            bigrams_user = describe_clusters_bigrams(n_clusters_user, normalized_df_user, y_predicted_user)
            bigrams_system = describe_clusters_bigrams(n_clusters_system, normalized_df_system, y_predicted_system)
            labels_type_user = bigrams_user['labels'].to_dict()
            labels_type_system = bigrams_system['labels'].to_dict()

        elif labels_type == "kBERT":
            kBERT_user = describe_clusters_kBERT(n_clusters_user, normalized_df_user, y_predicted_user, stopwords, n_grams)
            kBERT_system = describe_clusters_kBERT(n_clusters_system, normalized_df_system, y_predicted_system, stopwords, n_grams)
            labels_type_user = kBERT_user['labels'].to_dict()
            labels_type_system = kBERT_system['labels'].to_dict()
       
        #criar as labels para os clusters, quando se faz system e utilizador juntos
        # get list of values
        listValuesUsers = (list(labels_type_user.values()))
        listValuesSystems = (list(labels_type_system.values()))

         # add prefix user and sys
        prefix_user = "user -> "
        listValuesUsers = list(map(lambda element: prefix_user + "" + element, listValuesUsers))
        prefix_system = "sys -> "
        listValuesSystems = list(map(lambda element: prefix_system + "" + element, listValuesSystems))
        listValuesClusters = listValuesUsers + listValuesSystems

        names = []
        names = listValuesClusters
        word1 = 'SOD'
        word2 = 'EOD'
        ttl = word1 , word2
        names.extend(ttl)
      
        df_final['n_clusters_final']= df_final['cluster_user'].fillna(0) + df_final['cluster_system'].fillna(0)
        df_final.sort_values(by=['sequence'], inplace=True)
        df_final['n_clusters_final'] = df_final['n_clusters_final'].astype(int)
        #df_final.loc[df_final['Speaker'] == 'SYSTEM', 'n_clusters_final'] += n_clusters_user

        #matriz criada para os dois juntos (system + user)
        nClustersBoth = n_clusters_user + n_clusters_system
        numberLabelsMatrix = (nClustersBoth+2) # +2 (EOD & SOD)
        occurrence_matrix = np.zeros((numberLabelsMatrix, numberLabelsMatrix)) #nclusters(system+user) + 2 (para EOD e SOD)
        #CRIAR MATRIZ DE OCORRÊNCIAS
        for i in range(int(normalized_df['dialogue_id'].iat[-1])+1):
            turno_anterior = 0
            fim_dialogo = -1
            inicio_dialogo = df_final['n_clusters_final'].loc[(df_final.dialogue_id == i) & (df_final.turn_id == 0)]
            occurrence_matrix[nClustersBoth, inicio_dialogo] = occurrence_matrix[nClustersBoth, inicio_dialogo] + 1
            for turno in df_final['turn_id'].loc[(df_final.dialogue_id == i) & (df_final.turn_id != 0)]:
                dialogo = df_final['n_clusters_final'].loc[(df_final.dialogue_id == i) & (df_final.turn_id == turno)]
                dialogo_anterior = df_final['n_clusters_final'].loc[(df_final.dialogue_id == i) & (df_final.turn_id == turno_anterior)]
                occurrence_matrix[dialogo_anterior, dialogo] = occurrence_matrix[dialogo_anterior, dialogo] + 1
                turno_anterior = turno
                fim_dialogo = dialogo
            occurrence_matrix[fim_dialogo, nClustersBoth+1] = occurrence_matrix[fim_dialogo, nClustersBoth+1]  + 1

        #Aplicar o sentimento ao fluxo
        if "sentiment" in df_final.columns:
           df_final['avg_sentiment'] = ''
           line = 0
           for n in range(max(df_final['n_clusters_final']) + 1):
               cluster_df = df_final[df_final['n_clusters_final'] == n]
               avg_sent = statistics.mean(cluster_df['sentiment'])
               df_final['avg_sentiment'][line] = avg_sent
               line = line + 1

               df_final.loc[df_final.n_clusters_final == n, 'avg_sentiment'] = avg_sent

    # Add utterances if the Speaker is not interspersed
        # to_delete = []

        # for index, row in df_final.iterrows():
        #     if index == 0:
        #         continue      
            
        #     if row['Speaker'] == df_final.iloc[index -1]['Speaker']:
        #         anterior = df_final.iloc[index-1]['utterance']
        #         atual = row['utterance']

        #         nova = anterior + "\n" + atual

        #         to_delete.append(index - 1)

        #         df_final.loc[index, ['utterance']] = nova

        # df_final.drop(to_delete, inplace=True)
        # df_final.reset_index(inplace=True, drop=True)

        # t_id = 1

        # for index, row in df_final.iterrows():
        #     if index == 0:
        #         df_final.loc[index, ['turn_id']] = 0
        #         continue
            
        #     if row['dialogue_id'] == df_final.iloc[index -1]['dialogue_id']:
        #         df_final.loc[index, ['turn_id']] = t_id
        #         t_id +=1

        #     if row['dialogue_id'] != df_final.iloc[index -1]['dialogue_id']:
        #         df_final.loc[index, ['turn_id']] = 0
        #         t_id = 1

        # Find the sum of each row
        row_sums = occurrence_matrix.sum(axis=1)

        # Divide each element in the matrix by the corresponding row sum
        transition_matrix = np.divide(occurrence_matrix, row_sums[:, np.newaxis])
        matrix = pd.DataFrame(transition_matrix, index=names, columns=names)
        matrix = matrix.round(decimals = 2)
        matrix = matrix.fillna(0.00)

        #AVALIAÇÃO
        print(df_final.info())
        df_final.to_csv('file_name.csv')

        with open('kmeans_user.pkl', 'rb') as f:
            kmeans = pickle.load(f)
            print("kmeans User", kmeans)

        with open('kmeans_system.pkl', 'rb') as f:
            kmeans = pickle.load(f)
            print("kmeans System", kmeans)

        # dados_codificados = model.encode(df_final["utterance"]) #encoding sentence
        # clusters_ids = kmeans.predict(dados_codificados)
        # print("clusters_ids", clusters_ids)
        # # Obter os dialogue_ids e turn_ids
        # dialogue_ids = df_final["dialogue_id"].tolist()
        # turn_ids = df_final["turn_id"].tolist()

        # # Definir os clusters de interesse
        # clusters_interesse = sorted(set(df_final["n_clusters_final"]))
        # SOD_cluster = clusters_interesse[0]  # Start of Dialogue (SOD)
        # EOD_cluster = clusters_interesse[-1]  # End of Dialogue (EOD)

        # df_final["n_clusters_final"] = clusters_ids

        # def calcular_accuracy_transicoes(dialogue_ids, cluster_ids):
        #     num_dialogos = len(set(dialogue_ids))  # Número de diálogos distintos
        #     print("num_dialogos", num_dialogos)
        #     num_utterances = len(cluster_ids)  # Número total de utterances
        #     print("num_utterances", num_utterances)

        #     transicoes_previstas = 0
        #     transicoes_totais = 0

        #     for dialogue_id in set(dialogue_ids):
        #         dialogue_indices = df_final[df_final["dialogue_id"] == dialogue_id].index
        #         num_utterances_dialogo = len(dialogue_indices)

        #         for i in range(num_utterances_dialogo - 1):
        #             cluster_atual = cluster_ids[dialogue_indices[i]]
        #             cluster_proximo = cluster_ids[dialogue_indices[i + 1]]

        #             if i == 0 and cluster_atual == SOD_cluster and cluster_proximo != SOD_cluster:
        #                 # Transição do SOD para o primeiro cluster
        #                 transicoes_previstas += 1
        #                 print("Transição do SOD para o primeiro cluster", transicoes_previstas)
        #             elif cluster_atual != EOD_cluster and cluster_proximo != EOD_cluster:
        #                 # Transição entre clusters intermédios
        #                 transicoes_previstas += 1
        #                 print("Transição entre clusters intermédios", transicoes_previstas)

        #             transicoes_totais += 1

        #         # Verificar transição do último cluster para o EOD
        #         ultimo_utterance_index = dialogue_indices[-1]
        #         if ultimo_utterance_index < num_utterances - 1:
        #             if cluster_ids[ultimo_utterance_index] != cluster_ids[ultimo_utterance_index + 1]:
        #                 transicoes_previstas += 1
        #                 print("Verificar transição do último cluster para o EOD", transicoes_previstas)

        #                 transicoes_totais += 1

        #         # Verificar transição do SOD para o primeiro cluster em cada diálogo
        #         primeiro_utterance_index = dialogue_indices[0]
        #         if cluster_ids[primeiro_utterance_index] != SOD_cluster:
        #             transicoes_previstas += 1
        #             transicoes_totais += 1
        #             print("transição do SOD para o 1º cluster", transicoes_previstas)

        #     # Verificar transição do último cluster para o EOD em cada diálogo
        #     for dialogue_id in set(dialogue_ids):
        #         dialogue_indices = df_final[df_final["dialogue_id"] == dialogue_id].index
        #         ultimo_utterance_index = dialogue_indices[-1]
        #         if cluster_ids[ultimo_utterance_index] != EOD_cluster:
        #             transicoes_previstas += 1
        #             transicoes_totais += 1
        #             print("transição do último cluster para o EOD", transicoes_previstas)


        #     transicoes_totais += num_dialogos

        #     accuracy = transicoes_previstas / transicoes_totais
        #     return accuracy

        # # Calcular a accuracy das transições
        # accuracy = calcular_accuracy_transicoes(dialogue_ids, clusters_ids)

        # # Mostrar a accuracy
        # print("Precisão das transições:", accuracy)
        
        return generate_markov_chain_separately(n_clusters_user, n_clusters_system, matrix, df_final) #both_separetely
    
    return None

def run_test_evaluation(
    df: pd.DataFrame,
    package: str,
    representation: str,
    labels_type: str,
    n_clusters: int = 0,
    n_grams: int=0,
    model_cache_folder: str = None
):

    nlp = spacy.load(package)
    stopwords = nltk.corpus.stopwords.words("english")
    #stopwords = nltk.corpus.stopwords.words("portuguese")
    data = "/app/data/MultiWOZ_DAs.csv"
    corpus = data
    user = "both_separately" #both or both_separately
    if corpus == data:
        problem = "single"
    else:
        problem = "multi"

    if user == "both_separately" or user == "BOTH_SEPARATELY":
            
        import joblib
        import os
        
        file_path_system = os.path.abspath('models/kmeans_system.pkl')
        file_path_user = os.path.abspath('models/kmeans_user.pkl')

        # Carregar o modelo KMeans
        kmeansSystem = joblib.load(file_path_system)
        kmeansUser = joblib.load(file_path_user)
        # normalize turn_id, regex if true normalize URL's and Usernames started with '@', remove greeting words
        normalized_df = normalize_dataset(df, regex=True, removeGreetings=False, speaker='both')
        normalized_df_user = normalized_df[normalized_df['Speaker'] == 'USER']
        normalized_df_system = normalized_df[normalized_df['Speaker'] == 'SYSTEM']
    
        # Functions to transform into vectors (WORD EMBEDDING)
        model = SentenceTransformer(MODEL_NAME, cache_folder=model_cache_folder)

        if representation == "tfidf":
            topic_features = topic_features_to_remove(
                nlp, normalized_df, NUMBER_OF_FEATURES, topic_feature=False
            )  # topic features
            vectors_user = preprocessing_tfidf(
                stopwords, normalized_df_user, topic_features, 0.8, 3
            )

            vectors_system = preprocessing_tfidf(
                stopwords, normalized_df_system, topic_features, 0.8, 3
            )
        elif representation == "word2vec":
            vectors_user = word2vec(nlp, normalized_df_user)
            vectors_system = word2vec(nlp, normalized_df_system)
        elif representation == "sentenceTransformer":
            vectors_user = use_sentence_transformer(normalized_df_user, model)
            vectors_system = use_sentence_transformer(normalized_df_system, model)
        else:
            raise ValueError

        n_clusters_system = kmeansSystem.n_clusters
        n_clusters_user = kmeansUser.n_clusters

        print("Número de Clusters para USER: " + str(n_clusters_user))
        print("Número de Clusters para SYSTEM: " + str(n_clusters_system))

        normalized_df_user = normalized_df_user.reset_index(drop=True)
        normalized_df_system = normalized_df_system.reset_index(drop=True)
        
        # Filtra as utterances do User para codificação e previsão do cluster
        utterances_user = normalized_df_user["utterance"]
        dados_codificados_user = model.encode(utterances_user)
        clusters_idsUser = kmeansUser.predict(dados_codificados_user)

        # Filtra as utterances do System para codificação e previsão do cluster
        utterances_system = normalized_df_system["utterance"]
        dados_codificados_system = model.encode(utterances_system)
        clusters_idsSystem = kmeansSystem.predict(dados_codificados_system)

        # Cria o DataFrame com os dados codificados e os clusters
        df_user = normalized_df_user[["dialogue_id", "turn_id"]].copy()
        df_user["n_clusters_user"] = clusters_idsUser

        df_system = normalized_df_system[["dialogue_id", "turn_id"]].copy()
        df_system["n_clusters_system"] = clusters_idsSystem
        df_final = pd.concat([df_user, df_system])

        # Describe Clusters -> Bigrams | Verbs | Closest Document
        if labels_type == "verbs":
            verbs_user = describe_clusters_verbs(nlp, n_clusters_user, normalized_df_user, clusters_idsUser)
            verbs_system = describe_clusters_verbs(nlp, n_clusters_system, normalized_df_system, clusters_idsSystem)
            labels_type_user = verbs_user['labels'].to_dict()
            labels_type_system = verbs_system['labels'].to_dict()
            print("labels_type_user", labels_type_user)
            print("labels_type_system", labels_type_system)

        elif labels_type == "kBERT":
            kBERT_user = describe_clusters_kBERT(n_clusters_user, normalized_df_user, clusters_idsUser, stopwords, n_grams)
            kBERT_system = describe_clusters_kBERT(n_clusters_system, normalized_df_system, clusters_idsSystem, stopwords, n_grams)
            labels_type_user = kBERT_user['labels'].to_dict()
            labels_type_system = kBERT_system['labels'].to_dict()
       
        #criar as labels para os clusters, quando se faz system e utilizador juntos
        # get list of values
        listValuesUsers = (list(labels_type_user.values()))
        listValuesSystems = (list(labels_type_system.values()))

         # add prefix user and sys
        prefix_user = "user -> "
        listValuesUsers = list(map(lambda element: prefix_user + "" + element, listValuesUsers))
        prefix_system = "sys -> "
        listValuesSystems = list(map(lambda element: prefix_system + "" + element, listValuesSystems))
        listValuesClusters = listValuesUsers + listValuesSystems

        names = []
        names = listValuesClusters
        word1 = 'SOD'
        word2 = 'EOD'
        ttl = word1 , word2
        names.extend(ttl)

        # Criar a coluna 'n_clusters_final' somando as colunas 'n_clusters_user' e 'n_clusters_system'
        df_final['n_clusters_final'] = df_final['n_clusters_user'].fillna(0) + df_final['n_clusters_system'].fillna(0)
        df_final['sequence'] = range(len(df_final))

        # Ordena o DataFrame com base na coluna 'sequence'
        df_final.sort_values(by=['sequence'], inplace=True)

        # Remove a coluna 'sequence' após a ordenação, se não for mais necessária
        df_final.drop(columns=['sequence'], inplace=True)

        # Converter a coluna 'n_clusters_final' para o tipo inteiro
        df_final['n_clusters_final'] = df_final['n_clusters_final'].astype(int)

        #matriz criada para os dois juntos (system + user)
        nClustersBoth = n_clusters_user + n_clusters_system
        numberLabelsMatrix = (nClustersBoth+2) # +2 (EOD & SOD)
        occurrence_matrix = np.zeros((numberLabelsMatrix, numberLabelsMatrix)) #nclusters(system+user) + 2 (para EOD e SOD)
        #CRIAR MATRIZ DE OCORRÊNCIAS
        for i in range(int(normalized_df['dialogue_id'].iat[-1])+1):
            turno_anterior = 0
            fim_dialogo = -1
            inicio_dialogo = df_final['n_clusters_final'].loc[(df_final.dialogue_id == i) & (df_final.turn_id == 0)]
            occurrence_matrix[nClustersBoth, inicio_dialogo] = occurrence_matrix[nClustersBoth, inicio_dialogo] + 1
            for turno in df_final['turn_id'].loc[(df_final.dialogue_id == i) & (df_final.turn_id != 0)]:
                dialogo = df_final['n_clusters_final'].loc[(df_final.dialogue_id == i) & (df_final.turn_id == turno)]
                dialogo_anterior = df_final['n_clusters_final'].loc[(df_final.dialogue_id == i) & (df_final.turn_id == turno_anterior)]
                occurrence_matrix[dialogo_anterior, dialogo] = occurrence_matrix[dialogo_anterior, dialogo] + 1
                turno_anterior = turno
                fim_dialogo = dialogo
            occurrence_matrix[fim_dialogo, nClustersBoth+1] = occurrence_matrix[fim_dialogo, nClustersBoth+1]  + 1

        # Find the sum of each row
        row_sums = occurrence_matrix.sum(axis=1)

        # Divide each element in the matrix by the corresponding row sum
        transition_matrix = np.divide(occurrence_matrix, row_sums[:, np.newaxis])
        matrix = pd.DataFrame(transition_matrix, index=names, columns=names)
        matrix = matrix.round(decimals = 2)
        matrix = matrix.fillna(0.00)

        #AVALIAÇÃO
        #Ficheiro CSV dos dados de treino e de teste
        csv_treino = "data/file_name.csv"
        df_treino = pd.read_csv(csv_treino)
        csv_teste = "data/df_teste.csv"
        df_teste = pd.read_csv(csv_teste)

        def count_transitions_with_dict(df_teste, df_treino):
            # Preencher os valores ausentes com -1
            df_teste.fillna(-1, inplace=True)
            df_treino.fillna(-1, inplace=True)

            # Criar um dicionário para armazenar a contagem de transições
            transitions_dict = {}

            # Ordenar ambos os DataFrames pelo dialogue_id e turn_id
            df_teste_sorted = df_teste.sort_values(by=['dialogue_id', 'turn_id']).reset_index(drop=True)
            df_treino_sorted = df_treino.sort_values(by=['dialogue_id', 'turn_id']).reset_index(drop=True)
            print("df_teste_sorted", df_teste_sorted)
            print("df_treino_sorted", df_treino_sorted)

            # Loop externo: percorrer o df_teste
            for i, row_teste in df_teste_sorted.iterrows():
                # Obter a próxima linha do df_teste (se existir)
                if i < len(df_teste_sorted) - 1:
                    row_teste_next = df_teste_sorted.iloc[i + 1]
                else:
                    break

                # Loop interno: percorrer o df_treino
                for j, row_treino in df_treino_sorted.iterrows():
                    # Obter a próxima linha do df_treino (se existir)
                    if j < len(df_treino_sorted) - 1:
                        row_treino_next = df_treino_sorted.iloc[j + 1]
                    else:
                        break

                    # Verificar se é uma transição válida
                    if (row_teste['n_clusters_user'] == row_treino['cluster_user']) and (
                            row_teste['n_clusters_system'] == row_treino['cluster_system']) and (
                            row_teste_next['n_clusters_user'] == row_treino_next['cluster_user']) and (
                            row_teste_next['n_clusters_system'] == row_treino_next['cluster_system']):
                        # Criar a chave do dicionário com base nos clusters origem e destino
                        key = (row_teste['n_clusters_user'], row_teste_next['n_clusters_user'])

                        # Incrementar a contagem de transições para esta chave no dicionário
                        transitions_dict[key] = transitions_dict.get(key, 0) + 1

                    # Avançar para as próximas linhas do df_treino se a próxima do df_teste for maior
                    if row_treino_next['dialogue_id'] <= row_teste_next['dialogue_id'] and row_treino_next['turn_id'] <= \
                            row_teste_next['turn_id']:
                        break

            # Calcular a contagem total de transições
            total_transitions = sum(transitions_dict.values())

            return total_transitions, transitions_dict

        total_transitions, transitions_dict = count_transitions_with_dict(df_teste, df_treino)
        print("Contagem total de transições:", total_transitions)
        print("Transições encontradas:", transitions_dict)

        # Obter os dialogue_ids e turn_ids
        dialogue_ids = df_teste["dialogue_id"].tolist()
        turn_ids = df_teste["turn_id"].tolist()

        # Definir os clusters de interesse
        clusters_interesse = sorted(set(df_teste["n_clusters_final"]))
        SOD_cluster = clusters_interesse[0]  # Start of Dialogue (SOD)
        EOD_cluster = clusters_interesse[-1]  # End of Dialogue (EOD)

        def calcular_accuracy_transicoes(dialogue_ids, cluster_ids):
            num_dialogos = len(set(dialogue_ids))  # Número de diálogos distintos
            print("num_dialogos", num_dialogos)
            num_utterances = len(cluster_ids)  # Número total de utterances
            print("num_utterances", num_utterances)

            transicoes_previstas = 0
            transicoes_totais = 0

            for dialogue_id in set(dialogue_ids):
                dialogue_indices = df_teste[df_teste["dialogue_id"] == dialogue_id].index
                num_utterances_dialogo = len(dialogue_indices)

                for i in range(num_utterances_dialogo - 1):
                    cluster_atual = cluster_ids[dialogue_indices[i]]
                    cluster_proximo = cluster_ids[dialogue_indices[i + 1]]

                    if i == 0 and cluster_atual == SOD_cluster and cluster_proximo != SOD_cluster:
                        # Transição do SOD para o primeiro cluster
                        transicoes_previstas += 1
                        print("Transição do SOD para o primeiro cluster", transicoes_previstas)
                    elif cluster_atual != EOD_cluster and cluster_proximo != EOD_cluster:
                        # Transição entre clusters intermédios
                        transicoes_previstas += 1
                        print("Transição entre clusters intermédios", transicoes_previstas)

                    transicoes_totais += 1

                # Verificar transição do último cluster para o EOD
                ultimo_utterance_index = dialogue_indices[-1]
                if ultimo_utterance_index < num_utterances - 1:
                    if cluster_ids[ultimo_utterance_index] != cluster_ids[ultimo_utterance_index + 1]:
                        transicoes_previstas += 1
                        print("Verificar transição do último cluster para o EOD", transicoes_previstas)

                        transicoes_totais += 1

                # Verificar transição do SOD para o primeiro cluster em cada diálogo
                primeiro_utterance_index = dialogue_indices[0]
                if cluster_ids[primeiro_utterance_index] != SOD_cluster:
                    transicoes_previstas += 1
                    transicoes_totais += 1
                    print("transição do SOD para o 1º cluster", transicoes_previstas)

            # Verificar transição do último cluster para o EOD em cada diálogo
            for dialogue_id in set(dialogue_ids):
                dialogue_indices = df_teste[df_teste["dialogue_id"] == dialogue_id].index
                ultimo_utterance_index = dialogue_indices[-1]
                if cluster_ids[ultimo_utterance_index] != EOD_cluster:
                    transicoes_previstas += 1
                    transicoes_totais += 1
                    print("transição do último cluster para o EOD", transicoes_previstas)


            transicoes_totais += num_dialogos

            accuracy = transicoes_previstas / transicoes_totais
            return accuracy

        # Calcular a accuracy das transições
        #accuracy = calcular_accuracy_transicoes(dialogue_ids, clusters_ids)

        # Mostrar a accuracy
        #print("Precisão das transições:", accuracy)
        
        return generate_markov_chain_separately(n_clusters_user, n_clusters_system, matrix, df_final) #both_separetely

    return None

