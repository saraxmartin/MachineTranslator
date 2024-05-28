import config
import wandb
import pandas as pd
import torch
import torch.nn as nn
from utils.training import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadEncoderDecoderModel(input_lang, output_lang):
    encoder = EncoderRNN(input_size=input_lang, hidden_size=config.latent_dim).to(device)
    decoder = DecoderRNN(hidden_size=config.latent_dim, output_size=output_lang).to(device)
    encoder.load_state_dict(torch.load(config.encoder_path))
    decoder.load_state_dict(torch.load(config.decoder_path))
    return encoder, decoder


def test(input_lang, output_lang, data_loader, type='test'):
    # Load Encoder and Decoder model
    encoder, decoder = loadEncoderDecoderModel(input_lang.n_chars, output_lang.n_chars)

    # Test
    encoder.eval()
    decoder.eval()

    criterion = {'NLLLoss': nn.NLLLoss, 'CrossEntropyLoss': nn.CrossEntropyLoss}[config.criterion]()
    
    translated_sentences = []
    total_loss = []
    total_acc = []


    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):

            input_tensor, target_tensor = data
            input_tensor.to(device), target_tensor.to(device)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            acc = compute_accuracy(decoder_outputs, target_tensor, output_lang, EOS_token)

            total_loss.append(loss.item())
            total_acc.append(acc)

            
            for input, output, target in zip(input_tensor, decoder_outputs, target_tensor):
                input_words, decoded_words, target_words = translate(input_lang, output_lang, 
                                                                    input, output, target)
                translated_sentences.append((input_words, decoded_words, target_words))

            if type == 'test':
                if batch_idx % config.batch_size == 0:
                    print(f'    Step [{batch_idx+1}/{len(data_loader)}], ' 
                        f' Loss: {loss.item():.4f}, '
                        f'Accuracy: {acc:.4f}')

    avg_loss = sum(total_loss) / len(data_loader)
    avg_acc = sum(total_acc) / len(data_loader)     
    
    # Print final metrics
    print(f'Average loss of {type} data: {avg_loss}, '
          f'Average accuracy of {type} data: {avg_acc}')
    
    # Store loss and accuracy evolution
    if type == 'test':
        wandb.log({'test/loss': avg_loss, 'test/accuracy': avg_acc})

    # Store translated sentences in csv
    df = pd.DataFrame(translated_sentences, columns=['Input', 'Output', 'Target'])
    
    if type == 'train':
        path = config.results_path_train
    elif type == 'val':
        path = config.results_path_val
    elif type == 'test':
        path = config.results_path_test

    df.to_csv(path, index=False)

    

