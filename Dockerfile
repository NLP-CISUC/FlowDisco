FROM python:3.9

ENV HOME=/app \
    PYTHONUNBUFFERED=1

WORKDIR ${HOME}

ADD requirements.txt ${HOME}
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords punkt averaged_perceptron_tagger

COPY ./plataformateste ${HOME}/plataformateste

CMD ["python", "-m", "plataformateste.main", "--data-filename", "twitter_full_dataset-pequeno.csv", "--package", "en_core_web_md", "--representation", "sentenceTransformer", "--n-clusters", "15", "--labels-type", "bigrams"]
