import torch
import pandas as pd
from utils.training import translate
import config
import wandb
import pandas as pd
import torch
import torch.nn as nn
from utils.training import *
from utils.data import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

def tensorFromSentence(lang, sentence):
    indexes = []
    for char in sentence:
        if char in lang.char2index:
            indexes.append(lang.char2index[char])
        else:
            break  # Use <UNK> token for out-of-vocabulary characters
    indexes.append(EOS_token)  # Append the End-of-Sentence token
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def loadEncoderDecoderModel(input_lang, output_lang):
    encoder = EncoderRNN(input_size=input_lang.n_chars, hidden_size=config.latent_dim).to(device)
    decoder = DecoderRNN(hidden_size=config.latent_dim, output_size=output_lang.n_chars).to(device)
    encoder.load_state_dict(torch.load(config.encoder_path))
    decoder.load_state_dict(torch.load(config.decoder_path))
    return encoder, decoder

def translate(input_lang, output_lang, input_tensor, decoded_outputs, target_tensor=None):
    def get_chars(lang, tensor):
        _, topi = tensor.topk(1)
        ids = topi.squeeze()
        chars = []
        for idx in ids:
            if idx.item() == EOS_token:
                chars.append('EOS')
                break
            chars.append(lang.index2char[idx.item()])
        return chars

    input_chars = [input_lang.index2char[idx.item()] for idx in input_tensor]
    decoded_chars = get_chars(output_lang, decoded_outputs)
    
    if target_tensor is not None:
        target_chars = [output_lang.index2char[idx.item()] for idx in target_tensor]
    else:
        target_chars = None

    return input_chars, decoded_chars, target_chars

input_lang, output_lang, train_loader, val_loader, test_loader = get_dataloader()
encoder, decoder = loadEncoderDecoderModel(input_lang, output_lang)
sentence = "your sentence here"

# Convert sentence to tensor
input_tensor = tensorFromSentence(input_lang, sentence)

# Pass the input tensor through the encoder
encoder_outputs, encoder_hidden = encoder(input_tensor)

# Pass the encoder outputs to the decoder
decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor=None)

# Translate the decoder outputs
input_chars, decoded_chars, _ = translate(input_lang, output_lang, input_tensor, decoder_outputs)

# Print the translated sentence
print(''.join(decoded_chars))
