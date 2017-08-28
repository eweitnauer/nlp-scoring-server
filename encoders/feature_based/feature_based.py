import numpy as np
from fuzzywuzzy import fuzz
import similarity_measures as sf
from quick_score import quickScore
from nltk.corpus import stopwords

class featureBased(object):
    """
    The distributional bag of word model of sentence meaning:
    vector representation of a sentence is obtained by adding up
    the vectors of its constituting words.
    """

    stoplist = None
    qs = None

    def __init__(self):
        print("featureBased init: loading feature-based model")
        self.stoplist = stopwords.words('english')
        self.qs = quickScore()
        return

    def pairFeatures(self, sentenceA, sentenceB):
        features = list()

        ## len features all, chars, word
        features.append( np.log(len(sentenceA)+1) )
        features.append( np.log(len(sentenceB)+1) )
        features.append( np.log(abs(len(sentenceA) - len(sentenceB))+1) )
        features.append( np.log(len(''.join(set(sentenceA.replace(' ', ''))))+1 ))
        features.append( np.log(len(''.join(set(sentenceB.replace(' ', ''))))+1 ))
        features.append( np.log(len(sentenceA.split())+1 ))
        features.append( np.log(len(sentenceB.split())+1 ))

        features.append(np.log(sf.longestCommonsubstring(sentenceA, sentenceB)+1))
        features.append(np.log(sf.longestCommonSubseq(sentenceA, sentenceB)+1))

        ## token features
        features.append( len(set(sentenceA.lower().split()).intersection(set(sentenceB.lower().split()))) )
        features.append( np.log(fuzz.QRatio(sentenceA, sentenceB)+1) )
        features.append( np.log(fuzz.WRatio(sentenceA, sentenceB)+1) )
        features.append( np.log(fuzz.partial_ratio(sentenceA, sentenceB)+1) )
        features.append( np.log(fuzz.partial_token_set_ratio(sentenceA, sentenceB)+1) )
        features.append( np.log(fuzz.partial_token_sort_ratio(sentenceA, sentenceB)+1) )
        features.append( np.log(fuzz.token_set_ratio(sentenceA, sentenceB)+1) )
        features.append( np.log(fuzz.token_sort_ratio(sentenceA, sentenceB)+1) )

        ## word semantic features
        for f in self.qs.pairFeatures(sentenceA, sentenceB, stemming = 0):
            features.append(f)
        for f in self.qs.pairFeatures(sentenceA, sentenceB, stemming = 1):
            features.append(f)

        return features
