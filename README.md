# FlowDisco 
This notebook contains the code for the automatic discovery of dialogue flows from a history of conversations. This approach follows three main steps where the utterances are (i) represented in a vector space, (ii) grouped according to their semantic similarity, and (iii) the discovered clusters, which can be seen as dialogue states, are used as vertices in a transition graph. We propose an automatic measure to compute the extent to which the transitions in the test portion of the dataset align with the flows discovered in the training portion. The flow can also be enriched with the sentiment of the interlocutors. We propose metrics aiming at quantifying key aspects in customer-support flows, thus enabling more objective comparison, e.g., between different entities or agents, and providing more detailed insights into the interactions.

## Repository Structure
```
.
├── data/
│   ├── MultiWOZ_DAs_test.xlsx   # Test dataset with dialogues without sentiment
│   ├── MultiWOZ_DAs_train.xlsx   # Training dataset with dialogues without sentiment
│   ├── twitter_final_teste.xlsx   # Test dataset with dialogues with sentiment
│   ├── twitter_final_teste_sem_sent.xlsx   # Test dataset with dialogues without sentiment
│   ├── twitter_final_treino.xlsx   # Training dataset with dialogues with sentiment
│   └── twitter_final_treino_sem_sent.xlsx # Training dataset with dialogues without sentiment
├── FlowDisco.ipynb            # Colab notebook with the implemented code
└── README.md                          # Project description file
```

# How to cite
This project was presented in two scientific papers, both proposing innovative approaches to dialogue flow discovery. Below are the BibTeX references for the two papers.

### Paper 1: **Unsupervised Flow Discovery from Task-oriented Dialogues**
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
### Paper 2: **Sentiment-Aware Dialogue Flow Discovery for Customer-Support Services**

This paper presents a generic approach to dialogue flow discovery, using clustering techniques to identify dialogue states and state transitions, as well as analyzing the prevailing sentiment. The approach aims to enhance interpretability and provide support to artificial agents in customer support scenarios. Below is the corresponding BibTeX:

```
@inproceedings{ferreira-etal-2024-sentiment,
    title = "Sentiment-Aware Dialogue Flow Discovery for Interpreting Communication Trends",
    author = "Ferreira, Patr{\'\i}cia Sofia Pereira  and
      Carvalho, Isabel  and
      Alves, Ana  and
      Silva, Catarina  and
      Oliveira, Hugo Gon{\c{c}}alo",
    editor = "Kawahara, Tatsuya  and
      Demberg, Vera  and
      Ultes, Stefan  and
      Inoue, Koji  and
      Mehri, Shikib  and
      Howcroft, David  and
      Komatani, Kazunori",
    booktitle = "Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = sep,
    year = "2024",
    address = "Kyoto, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.sigdial-1.24",
    doi = "10.18653/v1/2024.sigdial-1.24",
    pages = "274--288",
    abstract = "Customer-support services increasingly rely on automation, whether fully or with human intervention. Despite optimising resources, this may result in mechanical protocols and lack of human interaction, thus reducing customer loyalty. Our goal is to enhance interpretability and provide guidance in communication through novel tools for easier analysis of message trends and sentiment variations. Monitoring these contributes to more informed decision-making, enabling proactive mitigation of potential issues, such as protocol deviations or customer dissatisfaction. We propose a generic approach for dialogue flow discovery that leverages clustering techniques to identify dialogue states, represented by related utterances. State transitions are further analyzed to detect prevailing sentiments. Hence, we discover sentiment-aware dialogue flows that offer an interpretability layer to artificial agents, even those based on black-boxes, ultimately increasing trustworthiness. Experimental results demonstrate the effectiveness of our approach across different dialogue datasets, covering both human-human and human-machine exchanges, applicable in task-oriented contexts but also to social media, highlighting its potential impact across various customer-support settings.",
}
```
