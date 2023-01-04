FROM python:3.9
ADD requirements.txt /
RUN pip install -r /requirements.txt
RUN pip install --upgrade kneed
RUN pip install -qU transformers sentence-transformers
RUN pip install clean-text
RUN pip install Unidecode
RUN pip install pandas
RUN pip install numpy
RUN pip install kmeans
RUN pip install -U gensim
RUN pip install spacy
RUN pip install nltk
RUN pip install sklearn
RUN pip install seaborn
RUN pip install networkx
RUN pip install pyparsing
RUN pip install pydot
RUN pip install graphviz
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en_core_web_md
RUN python3 -m spacy download en_core_web_lg
RUN pip install fsspec
ADD plataformateste.py /
ENV PYTHONUNBUFFERED=1
CMD [ "python", "./plataformateste.py", "--package", "en_core_web_md", "--representation", "sentenceTransformer", "--nClusters", "15", "--labelsType", "bigrams"]
