import os
from train import get_json_corpus, get_vocab, get_models, train

def main(args, config):
        
    training_corpus = get_json_corpus(config, os.path.join(config['Data']['data set dir'], args.training_corpus))
    validation_corpus = get_json_corpus(config, os.path.join(config['Data']['data set dir'], args.validation_corpus))
    
    data_sets = {
        'train': list(training_corpus.values()),
        'val': list(validation_corpus.values())
    }
    
    config['Model']['label'] = "pretraining"
    train(data_sets, config)

if __name__ == '__main__':
    import argparse, configparser
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_corpus', type=str, default='pretraining.json', help='pretraining corpus')
    parser.add_argument('--validation_corpus', type=str, default='pretraining_validation.json', help='pretraining validation corpus]')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--config', type=str, default='./model/config.ini', help='The config file')
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    
    print(f'Pretraining')
    main(args, config)