import encoders
import pickle
import keras.models
import re
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import tensorflow as tf

available_models = ['quickscore', 'bow', 'infersent', 'feature_based']
cache = {}

def CachedClassifier(model_names=[], classifier_name=None, from_cache_only=False):
    global cache, available_models
    if not model_names:
        raise ValueError("no models specified")
    if not all(name in available_models for name in model_names):
        raise ValueError("unknown model name")
    if classifier_name and not re.match('^[a-zA-Z+0-9_\\-]*$', classifier_name):
        raise ValueError("invalid character in classifier name")
    key = '+'.join(model_names)
    if classifier_name: key += '@' + classifier_name
    if not (key in cache):
        if from_cache_only: return False
        print('cache miss', key)
        cache[key] = Classifier(model_names, (classifier_name+'.h5') if classifier_name else None)
    else: print('cache hit', key)
    return cache[key]

# Initialize with a list of model names (bow, feature_based, infersent, quickscore),
# and optionally the file name of a pretrained classifier for that model combination.
class Classifier(object):
    def __init__(self, model_names=[], classifier_file=None, use_pickle=False):
        self.model_names = model_names
        self.classifier_file = classifier_file
        if classifier_file:
            print 'Loading pretrained classifier', classifier_file
            if use_pickle: self.classifier = pickle.load(open('pretrained/classifiers/' + classifier_file, 'rb'))
            else: self.classifier = keras.models.load_model('pretrained/classifiers/' + classifier_file)
            self.classifier._make_predict_function()
            self.graph = tf.get_default_graph()
        else:
            self.classifier = None

        self.models = list()
        for model_name in model_names:
            if (model_name == 'bow'):
                self.models.append(encoders.loadBow())
            elif (model_name == 'feature_based'):
                self.models.append(encoders.loadFeatureBased())
            elif (model_name == 'infersent'):
                self.models.append(encoders.loadInfersent())
            elif (model_name == 'quickscore'):
                self.models.append(encoders.loadQuickScore())
            else: raise "unknown model name", model_name
        if not self.models:
            raise 'no models specified'

    def get_score(self, targets, responses):
        if self.classifier: # supervised
            #import pdb; pdb.set_trace()
            feature_list = encoders.pairFeatures(self.models, process(targets), process(responses))
            valid_pos = getNotEmptyPositions(feature_list, verbose=True)
            valid_features = np.asarray(feature_list)[valid_pos]
            valid_features = np.asarray(valid_features.tolist()) # turn array of lists into ndarray (in case an empty array was included in all_trainF)

            print 'valid features', valid_features.shape

            with self.graph.as_default():
                valid_yhat = np.dot(self.classifier.predict(valid_features, verbose=0), [1])

            # now return an array as big as the targets array, with NAN where we found an error
            yhat = np.empty(len(targets))
            yhat[:] = np.nan
            yhat[valid_pos] = valid_yhat
            return yhat

        else: # unsupervised
            return encoders.sentenceSimilarity(self.models, process(targets), process(responses))

    def train(self, trainSet, devSet, use_labels=False, seed=1234):
        #K.set_session(self.sess)
        ## Takes an input model that can calculate similarity features for sentence pairs
        ## Returns a linear regression classifier on provided (gold) similarity scores (in 0...1)
        print 'Preparing data...'
        trainSet[0], trainSet[1], trainSet[2] = shuffle(trainSet[0], trainSet[1], trainSet[2], random_state=seed)

        all_trainF = encoders.pairFeatures(self.models, process(trainSet[0]), process(trainSet[1]))
        all_trainY = trainSet[2]
        valid_pos = getNotEmptyPositions(all_trainF, verbose=True)
        trainF = np.asarray(all_trainF)[valid_pos]
        trainF = np.asarray(trainF.tolist()) # turn array of lists into ndarray (in case an empty array was included in all_trainF)
        trainY = np.asarray(all_trainY)[valid_pos]
        trainY = np.asarray(trainY.tolist()) # turn array of lists into ndarray (in case an empty array was included in all_trainF)

        all_devF = encoders.pairFeatures(self.models, process(devSet[0]), process(devSet[1]))
        all_devY = devSet[2]
        valid_pos = getNotEmptyPositions(all_devF, verbose=True)
        devF = np.asarray(all_devF)[valid_pos]
        devF = np.asarray(devF.tolist()) # turn array of lists into ndarray (in case an empty array was included in all_trainF)
        devY = np.asarray(all_devY)[valid_pos]
        devY = np.asarray(devY.tolist()) # turn array of lists into ndarray (in case an empty array was included in all_trainF)

        print 'Compiling model...'
        print(trainF.shape, trainF.dtype)

        lrmodel = prepare_model(dim=trainF.shape[1])#, ninputs=trainF.shape[0])

        print 'Training...'
        bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY)

        #r = np.arange(1,6)
        #yhat = np.dot(bestlrmodel.predict_proba(devF, verbose=0), r)
        yhat = bestlrmodel.predict(devF, verbose=0)
        se = mse(yhat, devY)

        print("\n************ SUMMARY ***********")
        print 'Train data size: ' + str(len(trainY))
        print 'Dev data size: ' + str(len(devY))
        #print 'Dev Pearson: ' + str(pr)
        #print 'Dev Spearman: ' + str(sr)
        print 'Dev MSE: ' + str(se)
        print("********************************")

        return bestlrmodel

    def test(self, testSet):
        #K.set_session(self.sess)
        if (self.classifier):
            all_testF = np.asarray(encoders.pairFeatures(self.models, process(testSet[0]), process(testSet[1])))
            valid_pos = getNotEmptyPositions(all_testF, verbose=True)
            testF = np.asarray(all_testF)[valid_pos]
            testF = np.asarray(testF.tolist()) # turn array of lists into ndarray (in case an empty array was included in all_trainF)
            yhat = np.dot(self.classifier.predict(testF, verbose=0), [1]) # change to 1d array
            testY = np.asarray(testSet[2])[valid_pos]
            testY = np.asarray(testY.tolist()) # turn array of lists into ndarray (in case an empty array was included in all_trainF)
            test_score = testY #*5.0
        else:
            all_yhat = encoders.sentenceSimilarity(self.models, process(testSet[0]), process(testSet[1]))
            valid_pos = getNotNaNPositions(all_yhat, verbose=True)
            yhat = np.asarray(all_yhat)[valid_pos]
            testY = np.asarray(testSet[2])[valid_pos]
            test_score = testY #*5.0

        #import pdb; pdb.set_trace();

        yhat_score = np.clip(yhat, 0.0, 1.0)#5.0)
        pr = pearsonr(yhat_score, test_score)[0]
        sr = spearmanr(yhat_score, test_score)[0]
        se = mse(yhat, testY)

        print("\n************ SUMMARY ***********")
        print 'Test data size: ' + str(len(testY))
        print 'Test Pearson: ' + str(pr)
        print 'Test Spearman: ' + str(sr)
        print 'Test MSE: ' + str(se)
        print("********************************")

def getNotNaNPositions(data_list, verbose=False):
    valid_pos = [not np.isnan(j) for i,j in enumerate(data_list)]
    if verbose:
        unique, counts = np.unique(valid_pos, return_counts=True)
        print 'valid counts', dict(zip(unique, counts))
    return valid_pos

def getNotEmptyPositions(data_list, verbose=False):
    valid_pos = [len(j)>0 for i,j in enumerate(data_list)]
    if verbose:
        unique, counts = np.unique(valid_pos, return_counts=True)
        print 'valid counts', dict(zip(unique, counts))
    return valid_pos

def process(strings):
    return [re.sub('[!"#$%&\'()*+,-./:;<=>?@\\[\\]\\^_`{|}~]', '', string).lower().strip() for string in strings]

def prepare_label_model(dim, nclass):
    lrmodel = Sequential()
    lrmodel.add(Dense(nclass, input_dim=dim)) #set this to twice the size of sentence vector or equal to the final feature vector size
    lrmodel.add(Activation('softmax'))
    lrmodel.compile(loss='categorical_crossentropy', optimizer='adam') # or rather mse?
    return lrmodel

def prepare_model(dim, nclass=1):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    lrmodel = Sequential()
    lrmodel.add(Dense(5, input_dim=dim, activation='sigmoid', name='input_layer')) # named layers help with loading & saving
    lrmodel.add(Dense(nclass, input_dim=5, activation='sigmoid', name='output_layer'))
    lrmodel.compile(loss='mse', optimizer='adam')
    return lrmodel

def train_model(lrmodel, X, Y, devX, devY):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -10000
    early_stop_count = 0
    max_epoch = 1000
    epoch = 0

    while not done and epoch < max_epoch:
        lrmodel.fit(X, Y, epochs=20, verbose=0, shuffle=False, validation_data=(devX, devY))
        epoch += 20
        yhat = np.dot(lrmodel.predict(devX, verbose=0), [1])
        score = r2_score(devY, yhat)
        #score = -mse(devY, yhat)
        #score = pearsonr(yhat, devY)[0]
        if score > best:
            print 'new best score = ' + str(score)
            best = score
            lrmodel.save('temp_best_model.h5')
        else:
            print 'perf. degraded:' + str(score)
            if early_stop_count >= 5: done = True
            early_stop_count += 1
    lrmodel = keras.models.load_model('temp_best_model.h5')
    yhat = np.dot(lrmodel.predict(devX, verbose=0), [1])
    score = mse(yhat, devY)
    print 'Dev MSE: ' + str(score)
    return lrmodel

def decode_labels(labels):
    dim = labels.shape[1]
    if dim == 1: return labels
    return np.dot(labels, np.arange(1, dim))

def encode_labels(labels, nclass=5):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j,i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j,i] = np.floor(y) - y + 1
    return Y
