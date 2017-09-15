import itertools
from infersent.infersent import InferSent
from bow.bow import Bow
from feature_based.quick_score import quickScore
from feature_based.feature_based import featureBased
from scipy import spatial
import numpy as np

bow_model = None
inf_model = None
qs_model = None
fb_model = None

# infersent creates a 4096 dim. vector embedding for a sentence
# we can use the cosine to judge similarity, but training a classifier for
# our specific tasks works much better
def loadInfersent(K=500000):
    global inf_model
    if inf_model: return inf_model
    inf_model = InferSent(file='encoders/infersent/infersent.allnli.pickle'
                         ,K=K
                         ,use_cuda=False)
    return inf_model

# adds up 300 dim. vector embeddings for each word in the sentences
# we can use the cosine to judge similarity, but training a classifier for
# our specific tasks works much better
def loadBow():
    global bow_model
    if bow_model: return bow_model
    bow_model = Bow('pretrained/word2vec/GoogleNews-vectors-negative300.bin', limit=500000)
    return bow_model

# the 'old', internal algorithm that just looks at overlap between words,
# using stemming and synonyms. Can only be used for comparing sentences directly.
def loadQuickScore():
    global qs_model
    if qs_model: return qs_model
    qs_model = quickScore()
    return qs_model

# builds a vector of features describing both sentences; this can only be used
# together with a classifier; includes the output of quickScore
def loadFeatureBased():
    global fb_model
    if fb_model: return fb_model
    fb_model = featureBased()
    return fb_model

# Directly compare separately encoded sentences using model.sentence_similarity
# if available, and the cosine distance between the sentence encodings otherwise.
# Returns a list of average model scores per sentence pair.
# Uses a value of numpy.nan to mark error cases in which none of the models could be applied.
def sentenceSimilarity(models, targets, responses):
    result = list()
    i = 0
    for target, response in itertools.izip(targets, responses):
        i += 1
        if i%100==1: print i, 'of', len(targets)
        try:
            total = 0.0
            votes = 0
            for index, model in enumerate(models):
                if hasattr(model, 'sentence_similarity'):
                    sim = model.sentence_similarity(target, response)
                    if np.isnan(sim): continue
                    total += sim
                else:
                    encA = model.encode([target])
                    encB = model.encode([response])
                    if (len(encA[0])==0) or (len(encB[0])==0): continue
                    total += 1 - spatial.distance.cosine(encA[0], encB[0])
                votes += 1
            result.append((total / votes) if votes > 0 else np.nan)
        except:
            raise
    return result

## find features (a vector) describing the relation between two sentences
# appends an empty np array if a target-response pair could not be encoded
# by at least one of the models
def pairFeatures(models, targets, responses):
    result = list()
    i = 0
    for target, response in itertools.izip(targets, responses):
        i += 1
        if i%100==1: print i, 'of', len(targets)
        try:
            vector = list()
            for index , model in enumerate(models):
                part = model.pairFeatures(target,response)
                if len(part) == 0: raise ValueError
                vector.extend(part)
            result.append(vector)
        except:
            result.append([])
    return result
