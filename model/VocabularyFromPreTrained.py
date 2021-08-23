import nltk
import numpy as np
import os
import pickle
import torch

# Based on the tutorial found here:
# https://medium.com/mlearning-ai/load-pre-trained-glove-embeddings-in-torch-nn-embedding-layer-in-under-2-minutes-f5af8f57416a 
class VocabularyFromPreTrained():

    def __init__(self, embedding_file):
        
        self.embed_size = None
        
        # Build the embeddings from the GloVe file
        vocab,embeddings = [],[]
        with open(embedding_file,'rt') as fi:
            full_content = fi.read().strip().split('\n')
        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [int(val) for val in full_content[i].split(' ')[1:]]
            
            if self.embed_size is None:
                self.embed_size = len(i_embeddings)
            else:
                assert len(i_embeddings) == self.embed_size
                
            vocab.append(i_word)
            embeddings.append(i_embeddings)
            
        self.vocab = np.array(vocab)
        self.embeddings = np.array(embeddings)
        
        #insert '<pad>' and '<unk>' tokens at start of vocab
        unk_embedding = np.mean(self.embeddings,axis=0,keepdims=True)
        self._add_word('<pad>', 0)
        self._add_word('<unk>', 1, unk_embedding)
        self._add_word('<start>',2)
        self._add_word('<end>', 3)
        
        self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(self.embeddings))
        self.idx2word = dict(enumerate(self.vocab))
        self.word2idx = {self.idx2word[i]: i for i in self.idx2word}
        assert self.embedding_layer.weight.shape == self.embeddings.shape

    def tokenize(sentence):
        return nltk.tokenize.word_tokenize(sentence.lower())

    def sentence_to_idx(self, sentence):
        tokens = VocabularyFromPreTrained.tokenize(sentence)
        idxs = [self(token) for token in tokens]
        idxs.append(self('<end>'))
        idxs.insert(0, self('<start>'))
        return idxs
                
    def idxs_to_sentence(self, idxs):
        tokens = [self.idx2word[idx] for idx in idxs]
        return " ".join(tokens)
                
    def _add_word(self, word, idx, embedding=None):
        self.vocab = np.insert(self.vocab, idx, word)
        embedding = (np.zeros((1,self.embeddings.shape[1])) + idx) if embedding is None else embedding
        self.embeddings = np.vstack((embedding,self.embeddings))
    
    def __call__(self, word):
        word = word if word in self.vocab else '<unk>'
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('embedding_file', type=str, help='GloVe word embedding file')
    parser.add_argument('--output_dir', type=str, default='./data_sets/', help='path for saving vocabulary wrapper pickle file')
    args = parser.parse_args()
    
    output_file = os.path.basename(args.embedding_file)
    output_file = output_file[:output_file.rindex('.')]
    output_file = os.path.join(args.output_dir, f'{output_file}.pkl')

    vocab = VocabularyFromPreTrained(args.embedding_file)
    
    with open(output_file, 'wb') as output:
        pickle.dump(vocab, output)
        
    print("Total vocabulary size: {}".format(len(vocab)))
    print(f"Saved the vocabulary wrapper to '{output_file}'")



