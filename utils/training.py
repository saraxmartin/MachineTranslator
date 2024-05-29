from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import wandb
import jiwer
import config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

# FUNCTIONS

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

# EVALUATIONS
def compute_accuracy(predictions, targets, output_lang, eos_token):
    def tensor_to_chars(tensor, lang, eos_token):
        chars = []
        for idx in tensor:
            char = lang.index2char[idx.item()]
            if idx == eos_token:
                break
            chars.append(char)
        return chars

    batch_size = predictions.size(0)
    total_correct = 0
    total_chars = 0
    
    for i in range(batch_size):
        predicted_ids = predictions[i].max(dim=-1)[1]  # Get the predicted char indices
        reference_chars = tensor_to_chars(targets[i], output_lang, eos_token)
        predicted_chars = tensor_to_chars(predicted_ids, output_lang, eos_token)
        
        for pred_char, ref_char in zip(predicted_chars, reference_chars):
            if pred_char == "EOS" or ref_char == "EOS":
                break
            if pred_char == ref_char:
                total_correct += 1
            total_chars += 1

    accuracy = total_correct / total_chars if total_chars > 0 else 0
    return accuracy
# Function to calculate Character Error Rate (CER)
def cer(reference, hypothesis, eos_token="EOS"):
    total_error = 0
    total_chars = 0
    # Check if both reference and hypothesis are not the end-of-sequence token
    if reference != eos_token and hypothesis != eos_token:  
        # Calculate the error using jiwer.wer
        error = jiwer.wer(reference, hypothesis)
        # Count the number of characters in the reference
        total_chars += len(reference)
        # Accumulate the error
        total_error += error
    # Calculate the CER value
    cer_value = total_error / total_chars if total_chars > 0 else 0
    return cer_value

# Function to evaluate CER for a batch of predictions
def evaluate_cer(predictions, targets, output_lang, eos_token):
    # Function to convert tensor to characters
    def tensor_to_chars(tensor, lang, eos_token):
        chars = []
        for idx in tensor:
            char = lang.index2char[idx.item()]
            if char == eos_token:
                break
            chars.append(char)
        return chars

    total_cer = 0
    batch_size = predictions.size(0)
    
    # Loop over each item in the batch
    for i in range(batch_size):
        # Get predicted IDs
        predicted_ids = predictions[i].max(dim=-1)[1] 
        # Convert target and predicted sequences to characters
        reference_chars = tensor_to_chars(targets[i], output_lang, eos_token)
        hypothesis_chars = tensor_to_chars(predicted_ids, output_lang, eos_token)
        # Convert characters to strings
        reference_str = ''.join(reference_chars)
        hypothesis_str = ''.join(hypothesis_chars)
        # Calculate CER for the current pair
        cer_value = cer(reference_str, hypothesis_str)
        total_cer += cer_value
    
    # Calculate average CER for the batch
    average_cer = total_cer / batch_size
    return average_cer


def translate(input_lang, output_lang, 
              input_tensor, decoded_outputs, target_tensor):

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
    target_chars = [output_lang.index2char[idx.item()] for idx in target_tensor]

    return input_chars, decoded_chars, target_chars


# ENCODER / DECODER
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=config.dropouts):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)        
        if config.cell_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif config.cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded)
        return output, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        if config.cell_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif config.cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(config.max_length):
            assert decoder_input.max() < self.embedding.num_embeddings, "Decoder input is out of range"
            
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output)
        return output, hidden



# TRAINING AND VALIDATION EPOCHS

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion, output_lang):

    total_loss = 0
    total_acc = 0
    total_cer = 0

    for batch_idx, data in enumerate(dataloader):
        input_tensor, target_tensor = data
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        acc = compute_accuracy(decoder_outputs, target_tensor, output_lang,EOS_token)
        total_acc += acc
        cer_value = evaluate_cer(decoder_outputs, target_tensor, output_lang, EOS_token)
        total_cer +=cer_value

        if batch_idx % config.batch_size == 0:
            print(f'    Step [{batch_idx+1}/{len(dataloader)}], ' 
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {acc:.4f}, '
                  f'CER:{cer_value}')

    if len(dataloader) > 0:
        average_loss = total_loss / len(dataloader)  # Calculate average loss over all batches
        average_acc = total_acc / len(dataloader)   # Calculate average accuracy over all batches
        average_cer = total_cer / len(dataloader)   # Calculate average cer over all batches
    else:
        average_loss = 0
        average_acc = 0
        average_cer= 0

    return average_loss, average_acc, average_cer

def val_epoch(dataloader, encoder, decoder, criterion, input_lang, output_lang):

    total_loss = 0
    total_acc = 0
    total_cer = 0


    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            input_tensor, target_tensor = data
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )

            total_loss += loss.item()
            acc = compute_accuracy(decoder_outputs, target_tensor, output_lang,EOS_token)
            total_acc += acc
            cer_value = evaluate_cer(decoder_outputs, target_tensor, output_lang, EOS_token)
            total_cer +=cer_value
            if batch_idx % config.batch_size == 0:
                    print(f'        Step [{batch_idx+1}/{len(dataloader)}], ' 
                        f'Loss: {loss.item():.4f}, '
                        f'Accuracy: {acc:.4f}')


                    # Get translation examples
                    input_words, decoded_words, target_words = translate(input_lang, output_lang, 
                                                                        input_tensor[0], 
                                                                        decoder_outputs[0], 
                                                                        target_tensor[0])
                    
                    print(f'{input_lang.name}: {input_words}')
                    print(f'{output_lang.name} translation: {decoded_words}')
                    print(f'{output_lang.name} ground truth: {target_words}')

    average_loss = total_loss / len(dataloader)
    average_acc = total_acc / len(dataloader)
    average_cer = total_cer / len(dataloader)

    return average_loss, average_acc, average_cer

def trainSeq2Seq(train_loader, val_loader, encoder, decoder,
                 input_lang, output_lang):
    
    start = time.time()
    
    losses_train, acc_train, cer_train = [],[],[]
    losses_val, acc_val, cer_val= [],[],[]

    print(f"Cell type: {config.cell_type}")
    print(f"Hidden dimensions: {config.latent_dim}\n")
    # Define optimizer and criterion
    encoder_optimizer = getattr(torch.optim, config.opt)(encoder.parameters(), lr=config.learning_rate)
    print("Encoder optimizer:",encoder_optimizer)

    decoder_optimizer = getattr(torch.optim, config.opt)(decoder.parameters(), lr=config.learning_rate)
    print("Decoder optimizer:",decoder_optimizer)
    
    criterion = {'NLLLoss': nn.NLLLoss, 'CrossEntropyLoss': nn.CrossEntropyLoss}[config.criterion]()
    print("Loss function: ", criterion)

    # Training
    encoder.train()
    decoder.train()

    for epoch in range(1, config.epochs + 1):

        print("\nEpoch:",epoch)

        loss, acc, cer = train_epoch(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, output_lang)

        losses_train.append(loss)
        acc_train.append(acc)
        cer_train.append(cer)


        print(f'    Time: {timeSince(start, epoch / config.epochs)}, '
              f'Epochs completed: {epoch / config.epochs * 100}%, '
              f'Epoch loss: {loss:.4f}, '
              f'Epoch accuracy: {acc:.4f},'
              f'Epoch CER: {cer:.4f}')

        wandb.log({'epoch': epoch, 'train/loss': loss, 'train/accuracy': acc, 'train/CER':cer})

        # Validation
        encoder.eval()
        decoder.eval()

        with torch.no_grad():

            print(f'\n   Validation: epoch {epoch}')

            val_loss, val_acc, val_cer = val_epoch(val_loader, encoder, decoder, criterion, input_lang, output_lang)

            losses_val.append(val_loss)
            acc_val.append(val_acc)
            cer_val.append(val_cer)


            wandb.log({'epoch': epoch, 'validation/loss': val_loss, 'validation/accuracy': val_acc, 'validation/CER':cer})
        
        encoder.train()
        decoder.train()

    # Save the trained models
    torch.save(encoder.state_dict(), config.encoder_path)
    torch.save(decoder.state_dict(), config.decoder_path)


def train(input_lang, output_lang, train_loader, val_loader):
    encoder = EncoderRNN(input_size=input_lang.n_chars, hidden_size=config.latent_dim).to(device)
    decoder = DecoderRNN(hidden_size=config.latent_dim, output_size=output_lang.n_chars).to(device)
    print("Encoder and decoder created.\n")
    # Train the decoder and encoder
    trainSeq2Seq(train_loader, val_loader, encoder, decoder, input_lang, output_lang)
    print("\nModel trained successfully.")
