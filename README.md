# reconstructed-debate-project
Here is the debate project with the more modules, and baselines. 

dependencies:genism, numpy, spacy, nltk, pandas, pytorch=1.1, python=3.6

Put the data users.json, debates.json under a ./data directory.

First we generate the pretrained embeddings by running...
  * python big_issue_embedding.py
  * python user_aspect_embedding.py

Then we start the training and evaluating by running...
  * python debate.py
