# -*- coding: utf-8 -*-

import os
import re
from bs4 import BeautifulSoup
import unicodedata
import nltk

SOS_token = 0
EOS_token = 1

def preprocess_triple(triple):
    # Removing camel casing
    new_triple = []
    for entity in triple:
        entity = entity.lower()
        entity = entity.strip()
        
        # Removing accents and converting into pure ascii charactetrs
        entity = unicodedata.normalize('NFD', entity).encode('ascii', 'ignore')
        entity = str(entity, 'utf-8')

        # Removing non-letter and non-digit characters
        entity = re.sub(r"[^a-zA-Z1234567890.!?]+", r" ", entity)
        # Removing camel casing
        entity = re.sub("([a-z])([A-Z])","\g<1> \g<2>",entity)
        entity = entity.split(' ')
        for e in entity:
            if not e.isspace() and e: new_triple.append(e)

    return new_triple

def preprocess_sentence(sentence):
    sentence = [s.strip() for s in sentence]
    sentence = [s.lower() for s in sentence]

    return sentence

def parse_xml(file_dir, n_triples):

    pairs = []
    handler = open(file_dir).read()
    soup = BeautifulSoup(handler, 'lxml')
    entries = soup.findAll('entry', {"size":str(n_triples)})
    for entry in entries:
        triple = entry.find('mtriple').text.split('|')
        triple = preprocess_triple(triple)
       
        for sentence in entry.findAll('lex'):

            sentence = nltk.word_tokenize(sentence.text.lower())
            sentence = preprocess_sentence(sentence)
            if sentence[-1] == '.': sentence = sentence[:-1] # Remove the dot at the end of the sentence, because we will add it anyway during training.
            pairs.append((triple, sentence))
    
    return pairs

def filter_pair(p, max_length):
    return len(p[0]) < max_length and len(p[1]) < max_length

def filter_pairs(pairs, max_length):
    return [pair for pair in pairs if filter_pair(pair, max_length)]

class Vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def create_vocab(all_pairs):
    vocab = Vocab()
    for pair in all_pairs:
        vocab.addSentence(pair[0])
        vocab.addSentence(pair[1])
    
    print("Counted words: ", vocab.n_words)

    return vocab

def get_test_data(dataset_dir, n_triples=1, to_get_vocab=True, max_length=10):

    all_pairs = []
    xml_file_name = dataset_dir + '/testdata_unseen_with_lex.xml'
    pairs = parse_xml(xml_file_name, n_triples)
    all_pairs += pairs
    
    xml_file_name = dataset_dir + '/testdata_with_lex.xml'
    pairs = parse_xml(xml_file_name, n_triples)
    all_pairs += pairs

    all_pairs = filter_pairs(all_pairs, max_length)

    print("Read %s sentence pairs" % len(all_pairs))    
    
    vocab = None
    if to_get_vocab:
        vocab = create_vocab(all_pairs)
    
    return vocab, all_pairs

def get_normal_data(dataset_dir, mode, n_triples=1, to_get_vocab=True, max_length=10):
    xml_file_dir = dataset_dir+'/'+mode+'/'+str(n_triples)+'triples/'
    all_pairs = []
    for xml_file_name in os.listdir(xml_file_dir):
        pairs = parse_xml(xml_file_dir+'/'+xml_file_name, n_triples)
        all_pairs += pairs
    
    all_pairs = filter_pairs(all_pairs, max_length)

    print("Read %s sentence pairs" % len(all_pairs))    
    
    vocab = None
    if to_get_vocab:
        vocab = create_vocab(all_pairs)

    return vocab, all_pairs

def prepare_data(dataset_dir, mode='train', n_triples=1, to_get_vocab=True, max_length=10):
    
    if mode == 'test':
        dataset_dir = dataset_dir + '/test'
        vocab, all_pairs = get_test_data(dataset_dir, n_triples, to_get_vocab, max_length)
    else:
        vocab, all_pairs = get_normal_data(dataset_dir, mode, n_triples, to_get_vocab, max_length)

    return vocab, all_pairs
    


if __name__ == '__main__':
    dataset_dir = 'webnlg-dataset/webnlg_challenge_2017'
    vocab, pairs = prepare_data(dataset_dir, mode='train')
    print(pairs[0])
    vocab, pairs = prepare_data(dataset_dir, mode='dev')
    print(pairs[0])
    vocab, pairs = prepare_data(dataset_dir, mode='test')
    print(pairs[0])
