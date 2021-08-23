import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle
import random
import copy
import time
from Model import EncoderCNN, DecoderRNN
from dataHandler import get_loader
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import lr_scheduler
from VQGKFold import *
from VocabularyFromPreTrained import *

def train(data_sets, config, encoder = None, decoder = None):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
     # Create model directory
    if not os.path.exists(config['Model']['model path']):
        os.makedirs(config['Model']['model path'])
    
    # Load vocabulary object
    vocab = get_vocab(config)
    
    data_loaders = {label: get_loader(config['Data']['image dir'], data_sets[label], vocab, None, int(config['Data']['batch size']), True, int(config['Data']['num workers'])) for label in ['train', 'val']}
    
    if encoder is None or decoder is None:
        encoder, decoder = get_models(config, vocab)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    #params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=float(config['Model']['learning rate']))

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the models
    since = time.time()

    best_model_encoder_wts = copy.deepcopy(encoder.state_dict())
    best_model_decoder_wts = copy.deepcopy(decoder.state_dict())
    lowest_loss = 100000
    
    model_path_encoder = os.path.join(config['Model']['model path'], f"encoder_{config['Model']['label']}")
    model_path_decoder = os.path.join(config['Model']['model path'], f"decoder_{config['Model']['label']}")

    epochs = int(config['Model']['num epochs'])
    dataset_sizes = {x: len(data_sets[x]) for x in ['train', 'val']}
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        for phase in ['train', 'val']:
            print(f'Phase: {phase}')
           
            if phase == 'train':
                encoder.train()  # Set model to training mode
            else:
                encoder.eval()  # Set model to evaluate mode

            running_loss = 0.0

            for images, questions, lengths in data_loaders[phase]:
                
                # Set mini-batch dataset
                images = images.to(device)
                questions = questions.to(device)
                targets = pack_padded_sequence(questions, lengths, batch_first=True)[0]

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward, backward and optimize
                    features = encoder(images)
                    outputs = decoder(features, questions, lengths)
                    loss = criterion(outputs, targets)
                    decoder.zero_grad()
                    encoder.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        exp_lr_scheduler.step()
                    
                    # statistics
                    running_loss += loss.item()*images.size(0)
            
            epoch_loss = running_loss / dataset_sizes[phase]

            # deep copy the model
            if phase == 'val' and epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                best_model_encoder_wts = copy.deepcopy(encoder.state_dict())
                best_model_decoder_wts = copy.deepcopy(decoder.state_dict())
                torch.save(best_model_encoder_wts, model_path_encoder)
                torch.save(best_model_decoder_wts, model_path_decoder)
                
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Lowest loss: {:4f}'.format(lowest_loss))

def get_json_corpus(config, corpus_file):
    corpus_file = os.path.join(config['Data']['data set dir'], corpus_file)                
    with open(corpus_file, 'r') as corpus_file:
        corpus = json.load(corpus_file)
    return corpus

def get_vocab(config):
    with open(config['Data']['vocab'], 'rb') as vocab:
        vocab = pickle.load(vocab)
    return vocab

def get_models(config, vocab=None):
    if not vocab:
        vocab = get_vocab(config)
    encoder = EncoderCNN(vocab.embed_size)
    decoder = DecoderRNN(vocab.embedding_layer, vocab.embed_size, int(config['Model']['hidden size']), len(vocab), int(config['Model']['num layers']))
    return encoder, decoder
    
def main(args, config):
    
    # Load corpus and create data loaders
    corpus_file = get_json_corpus(config, f'{args.corpus}.json')
    
    vqgKFold = pickle.load(args.fold_file)    
    data_sets = [VQGKFold.get_data_for_split(data_split, corpus) for data_split in vqgKFold.get(args.fold)]
    data_sets = dict(zip(('train', 'val', 'test'), data_sets))
    
    # Load pretrained models
    encoder = None
    decoder = None
    
    """
    encoder, decoder = get_model(config)
    if os.path.exists(args.pretrained_encoder):
        encoder.load_state_dict(torch.load(args.pretrained_encoder))
    if os.path.exists(args.pretrained_decoder):
        decoder.load_state_dict(torch.load(args.pretrained_decoder))
    """
    
    config['Model']['label'] = f"{args.corpus}_{args.fold}"    
    train(data_sets, encoder, decoder, config)
    
if __name__ == '__main__':
    import argparse, configparser
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, help='Which corpus JSON file to use')
    parser.add_argument('--fold', type=int, help='The fold to use for the validation set')
    parser.add_argument('--fold_file', type=argparse.FileType('rb'), default='./data_sets/folds.pkl', help='pickled data structure that keeps track of data folds')
    parser.add_argument('--config', type=str, default='./model/config.ini', help='The config file')
    parser.add_argument('--config', type=str, default='./model/config.ini', help='The config file')
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    
    print(f'Training corpus {args.corpus} with fold {args.fold}')
    main(args, config)
