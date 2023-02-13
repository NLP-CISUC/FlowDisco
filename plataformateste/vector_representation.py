import gensim
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_numeric,
    strip_punctuation,
)
from sklearn.feature_extraction.text import TfidfVectorizer


def sentences_to_words(sentences):
    """Convert sentences to words."""
    for sentence in sentences:
        # deacc=True -> meaning removes punctuations
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def make_bigrams(texts, data):
    """Make bigrams."""
    data_words = list(sentences_to_words(data))
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


def lemmatization(nlp, texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """Lemmatization, converting a word to its root word. 'walking' -> 'walk'."""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


def topic_features_to_remove(nlp, normalized_df, num_words, topic_feature=None):
    # words that represent a topic, can choose in PreProcessing if that words will be ignored

    if topic_feature is True:
        df_initial = normalized_df
        data = df_initial["utterance"].values.tolist()

        # convert sentences to words
        data_words = list(sentences_to_words(data))

        # form bigrams
        data_words_bigrams = make_bigrams(data_words, data)

        # do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(
            nlp, data_words_bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
        )

        # Create Dictionary
        id2word = gensim.corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=20,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha="auto",
            per_word_topics=True,
        )

        lda_topics = lda_model.show_topics(num_words=num_words)

        topics = []
        filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]
        for topic in lda_topics:
            topics.append(preprocess_string(topic[1], filters))
            flat_list = [item for sublist in topics for item in sublist]
            print(flat_list)
            return flat_list
    else:
        flat_list = []
        return flat_list


def use_sentence_transformer(normalized_df, model):
    """To use the models of sentenceTransformer."""
    # transform Series into list
    sentences = normalized_df["utterance"].tolist()

    # sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)
    # print(embeddings)

    vectors = np.array(embeddings)
    print(vectors.shape)
    # vectors = np.append(vectors, df_initial['turn_id'], axis=2)
    # final_vectors = np.column_stack((vectors,df_initial['turn_id'], df_initial['interlocutor']))
    # final_vectors = np.column_stack(vectors)

    # df_final = pd.DataFrame(final_vectors) #doubt if the return is final_vectors or df_final
    # print(final_vectors)
    print("--> sentence-transformers used with success!")
    # pd.DataFrame(final_vectors).to_csv(r'/content/drive/MyDrive/ColabNotebooks/datasets/DFProcessed.csv', index=False)
    # final_vectors = pd.concat((vectors, df_initial['turn_id'], df_initial['interlocutor']), axis = 1).values #axis 1 meaning concatenates in column wise

    return vectors


def word2vec(nlp, df: pd.DataFrame):
    """To convert word to vectors."""
    x = [d.vector for t in df["utterance"] for d in nlp(str(t))]
    vectors = np.array(x)

    print(vectors.shape)
    print("--> word2vec used with success!")
    return vectors


def preprocessing_tfidf(stopwordseng, normalized_df, topic_features, max_df, min_df):
    """To use TDF-IDF."""
    df_initial = normalized_df

    # turn 'utterance' column into list and all lowercase
    df_initial["utterance"] = [
        x for x in df_initial["utterance"].map(lambda x: str(x).lower())
    ]

    stopwords_and_topic = stopwordseng
    stopwords_and_topic.extend(topic_features)  # Stopwords + Topic Features

    # TfidfVectorizer
    count_vect_tfidf = TfidfVectorizer(
        stop_words=stopwords_and_topic, max_df=max_df, min_df=min_df, ngram_range=(1, 1)
    )
    count_matrix = count_vect_tfidf.fit_transform(df_initial["utterance"].tolist())
    count_array = count_matrix.toarray()
    df_tfidf = pd.DataFrame(
        data=count_array, columns=count_vect_tfidf.get_feature_names_out()
    )

    return df_tfidf
