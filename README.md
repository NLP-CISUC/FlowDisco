# FlowDisco

The dataset needs to have initially a dialogue_id, turn_id, utterance and a speaker (only if we want to split between user and system) for each utterance. 

Put both when there is no separation of user and system and both_separetely when this separation is made (default = both_separately).

## First Step

The first step is to set the parameters to the values we deem appropriate.
These values are set in the CMD of the Dockerfile where each argument (numbered)
contains only one value:

1. package of the Spacy library (--package):
    + en_core_web_sm;
    + en_core_web_md (default value);
    + en_core_web_lg.
2. vector representation (--representation):
    + tfidf;
    + word2vec;
    + sentenceTransformer (default value).
3. label type (--labelsType):
    + bigrams (default value);
    + verbs;
4. Number of clusters manually entered (--nClusters):
    + integer value.

## Start the service - commands

To start the service locally, we have to use two commands:

1. docker compose build platform-uplink
2. docker compose run platform-uplink

## Generate the PDF Markov flow

After running the code, we see that the .dot file has been added to the 'results' folder ;
To generate the PDF Markov flow:

1. Install graphviz and pygraphviz (pip install graphviz, e.g.) - only once
2. Run the command in terminal: python -m plataformateste.generate_pdf_markov results/FILE_NAME.dot
