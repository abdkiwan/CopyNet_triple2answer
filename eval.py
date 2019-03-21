

from __future__ import unicode_literals, print_function, division
import io
from io import open
import unicodedata
import string
import re
import random
import urllib
import requests 
import numpy as np
import os

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from dataset_loader import prepare_data
from copynet import CopyEncoder, CopyDecoder
from functions import numpy_to_var, toData, to_np, to_var, visualize, decoder_initial, update_logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]

def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(vocab, pair):
    input_tensor = tensorFromSentence(vocab, pair[0])
    target_tensor = tensorFromSentence(vocab, pair[1])
    return (input_tensor, target_tensor)


def evaluate_sentence(encoder, decoder, sentence, vocab):

    input = np.array([[vocab.word2index[word] for word in sentence]])
    x = numpy_to_var(input)
    encoded, _ = encoder(x)
    decoder_in, s, w = decoder_initial(x.size(0))
    out_list=[]

    for j in range(x.size(1)): # for all sequences

        if j==0:
            out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                encoded_idx=input, prev_state=s,
                                weighted=w, order=j)
        else:
            tmp_out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                encoded_idx=input, prev_state=s,
                                weighted=w, order=j)
            out = torch.cat([out,tmp_out],dim=1)

        decoder_in = out[:,-1].max(1)[1] # train with sequence outputs
        out_list.append(out[:,-1].max(1)[1].cpu().data.numpy())

    out_list = [vocab.index2word[b[0]] for b in out_list]
    out_list = ['.' if x == 'EOS' else x for x in out_list]

    return out_list
    

def load_embeddings(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        vector = tokens[1:]
        vector = [float(v) for v in vector]
        data[word] = vector
    return data

def prepare_embeddings_array(vocab, embeddings):
    embeddings_array = np.random.uniform(-0.25, 0.25, (vocab.n_words, len(embeddings['a'])))
    for idx in range(vocab.n_words):
        word = vocab.index2word[idx]
        if word in embeddings:
            embeddings_array[idx] = embeddings[word]
    return torch.FloatTensor(embeddings_array)

def save_eval_results(encoder, decoder, pairs):
    
    if os.path.isfile("Results/hypothesis.txt"):
        os.remove("Results/hypothesis.txt")
        os.remove("Results/reference.txt")

    hypothesis_f = open("Results/hypothesis.txt", "w")
    reference_f = open("Results/reference.txt", "w")

    for idx in range(len(pairs)):

        pair = pairs[idx]
        triple = pair[0]
        reference = pair[1]
        reference = ' '.join(reference)

        hypothesis = evaluate_sentence(encoder, decoder, triple, vocab)
        hypothesis = ' '.join(hypothesis)

        hypothesis_f.write(hypothesis)
        hypothesis_f.write('\n')

        reference_f.write(reference)
        reference_f.write('\n')

def accuracy(encoder, decoder, pairs):
    
    save_eval_results(encoder, decoder, pairs)

    result = subprocess.check_output('./bleu_score.sh', shell=True)
    score = result.decode("utf-8").rstrip()
    print(score)

import json
if __name__ == "__main__":

    import subprocess

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SOS_token = 0
    EOS_token = 1

    dataset_dir = 'webnlg-dataset/webnlg_challenge_2017'

    print('Reading train dataset.')
    vocab, train_pairs = prepare_data(dataset_dir, 'train', 1, MAX_LENGTH)
    print('Reading validation dataset.')
    _, valid_pairs = prepare_data(dataset_dir, 'dev', 1, MAX_LENGTH)
    print('Reading test dataset')
    _, test_pairs = prepare_data(dataset_dir, 'test', 1, MAX_LENGTH)

    # Adding valid dataset words into vocab
    for pair in valid_pairs:
        vocab.addSentence(pair[0])
        vocab.addSentence(pair[1])

    # Adding test dataset words into vocab
    for pair in test_pairs:
        vocab.addSentence(pair[0])
        vocab.addSentence(pair[1])

    print("Reading word embeddings.")
    embeddings = load_embeddings('../resources/cc.en.300.vec')
    embeddings_array = prepare_embeddings_array(vocab, embeddings)
    del embeddings
    
    hidden_size = 300
    embed_size = 300

    encoder = CopyEncoder(vocab.n_words, embed_size, hidden_size, embeddings_array).to(device)
    decoder = CopyDecoder(vocab.n_words, embed_size, hidden_size).to(device)

    encoder.load_state_dict(torch.load('Results/best_encoder.pt'))
    decoder.load_state_dict(torch.load('Results/best_decoder.pt'))

    outputs_f = open("Results/outputs.txt", "w")
    hypothesis_f = open("Results/hypothesis.txt", "w")
    reference_f = open("Results/reference.txt", "w")

    for idx in range(len(test_pairs)):
        
        print("Processed ", idx, " out of ", len(test_pairs))

        pair = test_pairs[idx]
        triple = pair[0]
        reference = pair[1]
        reference = ' '.join(reference)

        hypothesis = evaluate_sentence(encoder, decoder, triple, vocab)
        hypothesis = ' '.join(hypothesis)

        triple = ' '.join(triple)
        outputs_f.write(triple)
        outputs_f.write('\n')
        outputs_f.write(reference)
        outputs_f.write('\n')
        outputs_f.write(hypothesis)
        outputs_f.write('\n')
        outputs_f.write('#####################################################################################')
        outputs_f.write('\n')
        
    accuracy(encoder, decoder, test_pairs)

    print("Done ...")
