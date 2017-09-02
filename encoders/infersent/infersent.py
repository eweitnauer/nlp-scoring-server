import sys
sys.path = ['encoders/infersent'] + sys.path
import torch
import numpy as np

class InferSent(object):
    def __init__(self, file='encoders/infersent/infersent.allnli.pickle'
                     , K=100000
                     , use_cuda=False):
        if use_cuda:
            self.model = torch.load(file)
        else:
            self.model = torch.load(file, map_location=lambda storage, loc: storage)
            self.model.use_cuda = False
        self.model.set_glove_path('pretrained/GloVe/glove.840B.300d.txt')
        self.model.build_vocab_k_words(K=K)
        return

    def encode(self, sentences):
        return self.model.encode(sentences)

    def pairFeatures(self, target, response):
        a = self.model.encode([target])
        b = self.model.encode([response])
        f = np.c_[np.abs(a - b), a * b]
        return f[0]
