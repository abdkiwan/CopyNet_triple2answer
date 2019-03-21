# CopyNet_triple2answer
In 'Results' folder, all the training results are found. It contains the following files:
outputs.txt : The model outputs for the test dataset.
bleu_scores.txt : Bleu scores on the validation dataset for every epoch.
vocab.txt : All the words in the vocabulary.
vocab_no_embedding.txt : All the words in the vocabulary that didn't have an embedding (a random embedding vector were generated). 

# Accuracy on Test dataset:

Number of train pairs :  2471
Number of validation pairs : 326
Number of test pairs : 996

BLEU = 9.62, 51.4/20.8/9.6/6.2 (BP=0.606, ratio=0.666, hyp_len=5010, ref_len=7523)