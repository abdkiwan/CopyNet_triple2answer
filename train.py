import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from copynet import CopyEncoder, CopyDecoder
from functions import numpy_to_var, toData, to_np, to_var, visualize, decoder_initial, update_logger
import random
import time
import sys
import math
import os
import subprocess

from dataset_loader import prepare_data
from eval import evaluate_sentence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1000)

# Hyperparameters
embed_size = 300
hidden_size = 300
num_epochs = 1000
batch_size = 16
lr = 0.001
weight_decay = 0.99
step = 0 # number of steps taken
teacher_forcing_ratio = 0.5

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

dataset_dir = 'webnlg-dataset/webnlg_challenge_2017'

print('Reading train dataset')
vocab, train_pairs = prepare_data(dataset_dir, 'train', 1, MAX_LENGTH)
print('Reading validation dataset.')
_, valid_pairs = prepare_data(dataset_dir, 'dev', 1, MAX_LENGTH)
print('Reading test dataset')
_, test_pairs = prepare_data(dataset_dir, 'test', 1, MAX_LENGTH)

# get number of batches
num_samples = len(train_pairs)
num_batches = int(num_samples/batch_size)

# Adding valid dataset words into vocab
for pair in valid_pairs:
    vocab.addSentence(pair[0])
    vocab.addSentence(pair[1])

# Adding test dataset words into vocab
for pair in test_pairs:
    vocab.addSentence(pair[0])
    vocab.addSentence(pair[1])

print("Writing vocabs into a text file.")
f= open("Results/vocab.txt","w+")
for i in range(vocab.n_words):
    f.write(vocab.index2word[i])
    f.write('\n')

import io
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

print("Reading word embeddings.")
embeddings = load_embeddings('../resources/cc.en.300.vec')
embeddings_array = prepare_embeddings_array(vocab, embeddings)

print("Writing vocabs that don't have an embedding vector into a text file.")
f = open("Results/vocab_no_embedding.txt","w+")
for i in range(vocab.n_words):
    if vocab.index2word[i] not in embeddings:
        f.write(vocab.index2word[i])
        f.write('\n')
del embeddings


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

    f = open("Results/bleu_scores.txt","a")
    f.write(score)
    f.write('\n')

    score = float(score.split(',')[0].split(' = ')[1])

    return score


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

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

################ load copynet model #####################
encoder = CopyEncoder(vocab.n_words, embed_size, hidden_size, embeddings_array)
decoder = CopyDecoder(vocab.n_words, embed_size, hidden_size)

if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()

################################# training ##################################

# set loss
criterion = nn.NLLLoss()

best_encoder = encoder
best_decoder = decoder
prev_valid_acc = 0

start = time.time()
for epoch in range(1, num_epochs+1):
    
    opt_e = optim.Adam(params=encoder.parameters(), lr=lr)
    opt_d = optim.Adam(params=decoder.parameters(), lr=lr)
    lr = lr * weight_decay # weight decay
    # shuffle data
    random.shuffle(train_pairs)
    samples_read = 0
    while(samples_read<len(train_pairs)):
        # initialize gradient buffers
        opt_e.zero_grad()
        opt_d.zero_grad()

        # obtain batch outputs
        batch = train_pairs[samples_read:min(samples_read+batch_size,len(train_pairs))]
        input_out, output_out, in_len, out_len = toData(batch, vocab)
        samples_read+=len(batch)

        # mask input to remove padding
        input_mask = np.array(input_out>0, dtype=int)

        # input and output in Variable form
        x = numpy_to_var(input_out)
        y = numpy_to_var(output_out)

        # apply to encoder
        encoded, _ = encoder(x)

        # get initial input of decoder
        decoder_in, s, w = decoder_initial(x.size(0))

        # out_list to store outputs
        out_list=[]
        for j in range(y.size(1)): # for all sequences

            # 1st state
            if j==0:
                out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                encoded_idx=input_out, prev_state=s,
                                weighted=w, order=j)
            # remaining states
            else:
                tmp_out, s, w = decoder(input_idx=decoder_in, encoded=encoded,
                                encoded_idx=input_out, prev_state=s,
                                weighted=w, order=j)
                out = torch.cat([out,tmp_out],dim=1)

            # for debugging: stop if nan
            if math.isnan(w[-1][0][0].item()):
                sys.exit()
            
            # select next input
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                decoder_in = out[:,-1].max(1)[1].squeeze() # train with sequence outputs
            else:
                decoder_in = y[:,j] # train with ground truth

        target = pack_padded_sequence(y,out_len.tolist(), batch_first=True)[0]
        pad_out = pack_padded_sequence(out,out_len.tolist(), batch_first=True)[0]
        
        # include log computation as we are using log-softmax and NLL
        pad_out = torch.log(pad_out)
        loss = criterion(pad_out, target)
        loss.backward()
                
        opt_e.step()
        opt_d.step()
        step += 1

    print("Validation")

    # Evaluation on validation set    
    valid_acc = accuracy(encoder, decoder, valid_pairs)
    if valid_acc > prev_valid_acc:
        best_encoder = encoder
        best_decoder = decoder
        print('Prev Accuracy : ', prev_valid_acc, ' , New Accuracy : ', valid_acc, ' => Saving a new best model.')
        prev_valid_acc = valid_acc
    else:
        print('Prev Accuracy : ', prev_valid_acc, ' , New Accuracy : ', valid_acc, ' => Keeping the current best model.')
    
    print('%s (%d %d%%)' % (timeSince(start, epoch / num_epochs), epoch, epoch / num_epochs * 100))
    print()

    if epoch % 5 == 0:
        torch.save(best_encoder.state_dict(), 'Results/best_encoder.pt')
        torch.save(best_decoder.state_dict(), 'Results/best_decoder.pt')

torch.save(best_encoder.state_dict(), 'Results/best_encoder.pt')
torch.save(best_decoder.state_dict(), 'Results/best_decoder.pt')
