from gensim.models import KeyedVectors
import numpy as np
from sklearn.preprocessing import normalize
from scipy import spatial

class Bow(object):
    """
    The distributional bag of word model of sentence meaning:
    vector representation of a sentence is obtained by adding up
    the vectors of its constituting words.
    """
    w2v = None

    def __init__(self, modelFile, limit=100000):
        print("bow init: loading word2vec model")
        self.w2v = KeyedVectors.load_word2vec_format(modelFile, binary=True, limit=limit)
        self.w2v.init_sims(replace=True)

        return
    def encode(self, sentences, verbose=False, use_eos=True):
        sentenceVecs = list()
        sentenceVec = None
        for index, sentence in enumerate(sentences):
            #print(sentence)
            #print(type(sentence))
            sentence = sentence.lower().split()
            wordCount = 0
            for word in sentence:
                if word in self.w2v.vocab:
                    if wordCount == 0:
                        sentenceVec = self.w2v[word]
                    else:
                        sentenceVec = np.add(sentenceVec, self.w2v[word])
                    wordCount+=1
            if(wordCount == 0):
                #print(str(sentence))
                raise ValueError("Cannot encode sentence " + str(index) + " : all words unknown to model!  ::" + str(sentence))
            else:
                sentenceVecs.append(normalize(sentenceVec[:,np.newaxis], axis=0).ravel())
        return np.array(sentenceVecs)
    def sentence_similarity(self, sentenceA, sentenceB):
        a = self.encode([sentenceA])
        b = self.encode([sentenceB])
        return 1 - spatial.distance.cosine(a[0],b[0])
    def pairFeatures(self, sentenceA, sentenceB):
        a = self.encode([sentenceA])
        b = self.encode([sentenceB])
        f = np.c_[np.abs(a - b), a * b]
        return f[0]
