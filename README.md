# reconstructed-debate-project
Here we reconstruct the debate project with a better structure, more modules, and baselines. 

Put the data users.json, debates.json under a ./data directory.

First we generate the pretrained embeddings by running...
  * python big_issue_embedding.py
  * python user_aspect_embedding.py

Then we start the training and evaluating by running...
  * python debate.py
