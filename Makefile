PYTHON = python3

run:
	docker-compose build platform-uplink && docker-compose run platform-uplink

install-locally:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -m spacy download en_core_web_md
	$(PYTHON) -m spacy download en_core_web_lg
	$(PYTHON) -m nltk.downloader stopwords punkt averaged_perceptron_tagger
	
run-locally:
	$(PYTHON) -m plataformateste.main --data-filename twitter_full_dataset-pequeno.csv --package en_core_web_md --representation sentenceTransformer --n-clusters 15 --labels-type bigrams
