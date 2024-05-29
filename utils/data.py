from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split

import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for char in sentence:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2):
    print("Reading lines...")

    lines = open(config.data_path, encoding='utf-8').\
        read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [pair[:2] for pair in pairs]

    if config.reverse:
        pairs = [list(reversed(p)) for p in pairs]
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0]) < config.max_length and len(p[1]) < config.max_length

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def indexesFromSentence(lang, sentence):
    return [lang.char2index[char] for char in sentence if char in lang.char2index]

def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs)
    if not pairs:
        print("Error: All pairs were filtered out. Check your max_length and data file.")
        return
    print(f"Trimmed to {len(pairs)} sentence pairs")
    print("Counting characters...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted characters:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    
    print("Random pair: ", random.choice(pairs))
    return input_lang, output_lang, pairs


def get_dataloader():
    input_lang, output_lang, pairs = prepareData(config.input_language, config.output_language)

    # If the dataset is empty after filtering, return None for loaders
    if not pairs:
        print("Dataset is empty after filtering.")
        return input_lang, output_lang, None, None, None

    # Transform to numerical data: one hot vector 
    n = len(pairs)
    input_ids = np.zeros((n, config.max_length), dtype=np.int32)
    target_ids = np.zeros((n, config.max_length), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    data = TensorDataset(torch.LongTensor(input_ids).to(device),
                         torch.LongTensor(target_ids).to(device))
    
    # Print the size of the dataset
    print("Total number of examples in the dataset: ", len(data))

    # Access a valid index within the bounds
    if len(data) > 0:
        example_index = min(1000, len(data) - 1)
        print("Train data tensor example: ", data[example_index]) 

    val_size = int(config.validation_split * len(data))
    test_size = int(config.test_split * len(data))
    train_size = len(data) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return input_lang, output_lang, train_loader, val_loader, test_loader
