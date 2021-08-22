import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle
import random
import copy
import time
from model.Model import EncoderCNN, DecoderRNN
from model.dataHandler import get_loader
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import lr_scheduler
from model.VQGKFold import *
from model.VocabularyFromPreTrained import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(epochs, data_loaders, vocab):

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        for phase in ['train', 'val']:
            print(f'Phase: {phase}')
           
            #if phase == 'train':
                #encoder.train()  # Set model to training mode
            #else:
                #encoder.eval()  # Set model to evaluate mode

            running_loss = 0.0

            for images, targets, lengths in data_loaders[phase]:
                targets = [vocab.idxs_to_sentence(target.tolist()) for target in targets]
                print(f'Targets: {targets}')
                print(f'Lengths: {lengths}')
                break
    
    """
                # Set mini-batch dataset
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward, backward and optimize
                    features = encoder(images)
                    outputs = decoder(features, captions, lengths)
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
    """

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Load vocabulary object
    vocab = pickle.load(args.vocab)
    print(vocab.idx2word)

    # Load corpus and create data loaders
    corpus_file = os.path.join(args.data_set_dir, f'{args.corpus}.json')                
    with open(corpus_file, 'r') as corpus_file:
        corpus = json.load(corpus_file)
    
    vqgKFold = pickle.load(args.fold_file)    
    data_sets = [VQGKFold.get_data_for_split(data_split, corpus) for data_split in vqgKFold.get(args.fold)]
    data_sets = dict(zip(('train', 'val', 'test'), data_sets))
    data_loaders = {label: get_loader(args.image_dir, data_sets[label], vocab, None, args.batch_size, True, args.num_workers) for label in ['train', 'val', 'test']}

    """
    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(vocab.embedding_layer, vocab.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    #params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the models
    since = time.time()

    best_model_encoder_wts = copy.deepcopy(encoder.state_dict())
    best_model_decoder_wts = copy.deepcopy(decoder.state_dict())
    lowest_loss = 100000
    
    model_path_encoder = os.path.join(args.model_path, f"encoder_{args.corpus}_{args.fold}")
    model_path_decoder = os.path.join(args.model_path, f"decoder_{args.corpus}_{args.fold}")
    """
    
    train(args.num_epochs, data_loaders, vocab)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
 
    # Arguments with no default
    parser.add_argument('--corpus', type=str, help='which corpus JSON file to use')
    parser.add_argument('--fold', type=int, help='The fold to use for the validation set')
    
    # Data loader parameters
    parser.add_argument('--data_set_dir', type=str, default='./data_sets/', help='directory with json data sets')
    parser.add_argument('--fold_file', type=argparse.FileType('rb'), default='./data_sets/folds.pkl', help='pickled data structure that keeps track of data folds')
    parser.add_argument('--image_dir', type=str, default='./images/', help='directory for images')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=2)

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # Other details               
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--model_path', type=str, default='./saved_models/', help='path for saving trained models')
    parser.add_argument('--vocab', type=argparse.FileType('rb'), default='./data_sets/glove.6B.300d.pkl', help='The pickled Vocabulary object with pre-trained embeddings')

    args = parser.parse_args()
    print(args)
    main(args)
