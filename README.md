# FlowDisco 
The FlowDisco repository contains:
- **Data folder**: This folder includes data from two different datasets (Schema-Guided Dialogue (SGD) - with dialogue acts (DAs) and intents, and Emowoz). This approach can be applied to any set of written/transcribed dialogues, in any language or domain. The dataset must initially have the columns "turn_id" (an integer) or "dialogue_id" (a sequential integer - 0,1,2,3,...), "utterance," and the "Speaker" of each expression. Use "both" when there is no separation of user and system, and "both_separately" when this separation is made (default = both_separately).
- **FlowDisco folder**: This folder contains the code for the automatic discovery of dialogue flows from a history of conversations. This approach follows three main steps where the utterances are (i) represented in a vector space, (ii) grouped according to their semantic similarity, and (iii) the discovered clusters, which can be seen as dialogue states, are used as vertices in a transition graph. We propose an automatic measure for computing the extent to which the transitions in the test portion of the dataset align with the flows discovered from the training portion.
- **Docker files**: The parameters with values we consider appropriate are defined in the CMD of the Dockerfile, where each argument contains only one value.

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


