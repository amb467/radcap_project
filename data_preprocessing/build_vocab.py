import json
import nltk
import pickle
from collections import defaultdict

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def sentence_to_idx(self, sentence):
        tokens = Vocabulary.tokenize(sentence)
        idxs = [self(token) for token in tokens]
        idxs.append(self('<end>'))
        idxs.insert(0, self('<start>'))
        return idxs
    
    def idxs_to_sentence(self, idxs):
        tokens = [self.idx2word[idx] for idx in idxs]
        return " ".join(tokens)
        
    def tokenize(sentence):
        return nltk.tokenize.word_tokenize(sentence.lower())
               
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(data_set, count_thr):

    counts = defaultdict(int)
    questions = [q_dict['question'] for q_dict_list in data_set.values() for q_dict in q_dict_list]
    
    for question in questions:
        tokens = Vocabulary.tokenize(question)
        for word in tokens:
            counts[word] +=  1

    words = [w for w, n in counts.items() if n > count_thr]  #reject words that occur less than 'word count threshhold'

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    vocab.add_word('.')
    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=argparse.FileType('r'), help='path for JSON data set file')
    parser.add_argument('--vocab_path', type=str, help='path for saving vocabulary wrapper pickle file')
    parser.add_argument('--threshold', type=int, default=3, help='minimum word count threshold')
    args = parser.parse_args()
    
    data_set = json.load(args.data_set)
    vocab = build_vocab(data_set, args.threshold)
    
    with open(args.vocab_path, 'wb') as vocab_path:
        pickle.dump(vocab, vocab_path)
        
    print("Total vocabulary size: {}".format(len(vocab)))
    #print("Saved the vocabulary wrapper to '{}'".format(args.vocab_path.name))
    print("Saved the vocabulary wrapper to '{}'".format(args.vocab_path))



