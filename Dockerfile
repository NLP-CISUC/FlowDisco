FROM python:3.9
ADD requirements.txt /
RUN pip install -r /requirements.txt
ADD plataformateste.py /
ENV PYTHONUNBUFFERED=1
CMD [ "python", "./plataformateste.py", "--package", "en_core_web_md", "--representation", "sentenceTransformer", "--n-clusters", "15", "--labels-type", "bigrams"]
