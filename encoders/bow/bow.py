from gensim.models import KeyedVectors
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from encoders.feature_based.spell import spellChecker

STOPLIST = stopwords.words('english')
spell_checker = spellChecker()

class Bow(object):
    """
    The distributional bag of word model of sentence meaning:
    vector representation of a sentence is obtained by adding up
    the vectors of its constituting words.
    """
    w2v = None

    def __init__(self, modelFile, limit=40000):
        print("bow init: loading word2vec model")
        self.w2v = KeyedVectors.load_word2vec_format(modelFile, binary=True, limit=limit)
        self.w2v.init_sims(replace=True)

        return

    def encode(self, sentences, spell_check=True):
        sentenceVecs = list()
        for index, sentence in enumerate(sentences):
            sentence = sentence.lower().split()
            sentenceVec = []
            for word in sentence:
                if word in STOPLIST: continue
                if (word in self.w2v.vocab):
                    if len(sentenceVec) == 0:
                        sentenceVec = self.w2v[word]
                    else:
                        sentenceVec = np.add(sentenceVec, self.w2v[word])
                elif spell_check:
                    replacements = [w for w in spell_checker.spellCorrect(word)
                                    if (w in self.w2v.vocab) and not (w in STOPLIST)]
                    for r_word in replacements:
                        if len(sentenceVec) == 0:
                            sentenceVec = self.w2v[r_word] / len(replacements)
                        else:
                            sentenceVec = np.add(sentenceVec, self.w2v[r_word] / len(replacements))
            if len(sentenceVec) == 0:
                sentenceVecs.append(np.array([]))
            else:
                sentenceVecs.append(normalize(sentenceVec[:,np.newaxis], axis=0).ravel())
        return np.array(sentenceVecs)

    def pairFeatures(self, sentenceA, sentenceB):
        a = self.encode([sentenceA])
        b = self.encode([sentenceB])
        f = np.c_[np.abs(a - b), a * b]
        return f[0]
