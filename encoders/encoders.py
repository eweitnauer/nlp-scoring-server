import sys
sys.path = ['encoders/infersent'] + sys.path
import torch
import itertools
from bow.bow import Bow
from feature_based.quick_score import quickScore
from feature_based.feature_based import featureBased

# get infersent model with vocab of K
def loadInfersent(K=100000):
	infersent = torch.load('encoders/infersent/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
	infersent.use_cuda = False
	infersent.set_glove_path('pretrained/GloVe/glove.840B.300d.txt')
	infersent.build_vocab_k_words(K=100000)
	return infersent

def loadBow():
	return Bow('pretrained/word2vec/GoogleNews-vectors-negative300.bin')

def loadQuickScore():
	return quickScore()

def loadFeatureBased():
	return featureBased()

## find features (a vector) describing the relation between two sentences
def pairFeatures(models, a, b):
    print "using method pairFeatures!"
    result = list()
    for sentenceA,sentenceB in itertools.izip(a,b):
        try:
            vector = list()
            for index , model in enumerate(models):
                part = model.pairFeatures(sentenceA,sentenceB)
                vector.extend(part)
            result.append(vector)
        except:
            print("ERROR: " + sentenceA + " & " +  sentenceB)
            result.append(errorFlag)
    return result
