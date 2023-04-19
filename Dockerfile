FROM python:3.9

ENV HOME=/app \
    PYTHONUNBUFFERED=1

WORKDIR ${HOME}

RUN apt-get update
RUN apt-get install graphviz libgraphviz-dev -y
RUN pip install numpy scipy pygraphviz

#RUN apt-get install -qqy \ 
#       python3.9-dev graphviz libgraphviz-dev pkg-config \
#       python-pydot python3-pydot python-pygraphviz python3-pygraphviz; \
#  rm -rf /var/lib/apt/lists/*
ADD requirements.txt ${HOME}
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md && \
    python -m spacy download en_core_web_lg && \
    python -m nltk.downloader stopwords punkt averaged_perceptron_tagger

RUN python -m spacy download pt_core_news_sm  && \
    python -m nltk.downloader stopwords punkt averaged_perceptron_tagger


COPY ./plataformateste ${HOME}/plataformateste

#CMD ["python", "-m", "plataformateste.main", "--data_filename", "twitter_full_dataset_v2.csv", "--package", "pt_core_news_sm", "--representation", "tfidf", "--n-clusters", "5", "--labels-type", "verbs"]

#Portuguese
#CMD ["python", "-m", "plataformateste.main", "--data_filename", "twitter_full_dataset_v2.csv", "--package", "pt_core_news_sm", "--representation", "tfidf", "--labels-type", "verbs"]

#English
CMD ["python", "-m", "plataformateste.main", "--data_filename", "MultiWOZ_DAs.csv", "--package", "en_core_web_md", "--representation", "sentenceTransformer", "--labels-type", "bigrams"]
