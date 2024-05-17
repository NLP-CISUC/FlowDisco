# FlowDisco

The dataset needs to have initially a turn_id (an integer number), utterance and a speaker (only if we want to split between user and system) for each utterance. If you have dialogue_id it has to be an integer and sequential (0, 1, 2, 3, ...).

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
    + kBERT.
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

# How to cite
A [paper](https://arxiv.org/abs/2405.01403) proposing an innovative approach to unsupervised discovery of dialogue flows from conversation history and an automatic validation metric was presented at the 23rd International Conference on Hybrid Intelligent Systems (HIS 2023). See BibTex:

```
@inproceedings{ferreira2023unsupervised,
    title = {Unsupervised Flow Discovery from Task-oriented Dialogues},
    author = {Patrícia Ferreira and Daniel Martins and Ana Alves and Catarina Silva and Hugo {Gonçalo~Oliveira}},
    booktitle = {Proceedings of 23nd International Conference on Hybrid Intelligent Systems (HIS 2023)},
    year = {2023}
    publisher = {Springer}
}
```


