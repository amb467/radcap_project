import torch
import numpy as np
import os
from train import get_json_corpus, get_vocab, get_models
from dataHandler import image_transform

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generateCaption(image, vocab, encoder, decoder):
    # Generate an caption from the image
    image = image.unsqueeze(0)
    features = encoder(image)
    sampled_ids = decoder.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    return sentence



def main(args, config):

    test_corpus = get_json_corpus(config, args.test_corpus)
    vocab = get_vocab(config)
    encoder, decoder = get_models(config, vocab)

    encoder.load_state_dict(torch.load(args.encoder_path))
    encoder.to(device)
    encoder.eval()
    decoder.load_state_dict(torch.load(args.decoder_path))
    decoder.to(device)

    for image_id, image_dict in test_corpus.items():

        image = image_transform(config['Data']['image dir'], image_dict[0]['image_path'])
        image = image.to(device)

        ai_question = generateCaption(image, vocab, encoder, decoder)
        human_questions = [item['question'] for item in image_dict]

        print(f'For image id {image_id}')
        print(f'Human questions are: {human_questions}')
        print(f'AI questions: {ai_question}')
        print('\n')


if __name__ == '__main__':
    import argparse, configparser

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, help='path for trained decoder')
    parser.add_argument('--test_corpus', type=str, help='path for json file with test imgs')
    parser.add_argument('--config', type=str, default='./model/config.ini', help='The config file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    print(f'Testing corpus {args.test_corpus}')
    main(args, config)