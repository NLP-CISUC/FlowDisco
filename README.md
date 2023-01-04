The first step is to set the parameters to the values we deem appropriate. 
These values are set in the CMD of the Dockerfile where each argument (numbered) 
contains only one value (->):
1. package of the Spacy library (--package):
    -> en_core_web_sm; 
    -> en_core_web_md (default value);
    -> en_core_web_lg.
2. vector representation (--representation):
    -> tfidf;
    -> word2vec;
    -> sentenceTransformer (default value).
3. label type (--labelsType):
    -> bigrams (default value);
    -> verbs;
    -> closestDocuments.
4. Number of clusters manually entered (--nClusters):
    -> integer value (15 - default value).


To start the service locally, we have to use two commands:
1. docker compose build
2. docker run -v FolderWhereThePlatformIsStored:/mnt/mydata platform-uplink
In my example it is: docker run -v C:/Users/patricia/platform/:/mnt/mydata platform-uplink 


After running the code, we see that the .dot file has been added to the project folder;
To generate the PDF Markov flow:
    1. install graphviz (pip install graphviz, e.g.)
    2. Run the command in terminal: python generatePDFMarkov.py markov.dot
If we want to generate a .dot file for another type of label, we have to delete the old
dot file from the project folder and re-run the program to generate a new PDF.
