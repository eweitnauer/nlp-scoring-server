from encoders import encoders
import pickle
from string import punctuation
import numpy as np

classifier_map = {
    'bow,features on sts1214': { 'file': 'bow+fb+sts1214.file', 'models': ['bow', 'feature_based'] },
    'features on sts2014': { 'file': 'fb+sts1214.file', 'models': ['feature_based'] }
}

def loadStoredClassifier(options):
    print 'Loading pretrained classifier', options['file']
    classifier = pickle.load(open('pretrained/classifiers/' + options['file'], 'rb'))
    models = list()
    for model_name in options['models']:
        if (model_name == 'bow'):
            models.append(encoders.loadBowModel())
        elif (model_name == 'feature_based'):
            models.append(encoders.loadFeatureBased())
        elif (model_name == 'infersent'):
            models.append(encoders.loadInfersent())
        elif (model_name == 'quickscore'):
            models.append(encoders.loadQuickScore())
        else: raise "unknown model name", model_name
    return models, classifier

## lower case and removes punctuation from the input text
def process(s): return [i.lower().translate(None, punctuation).strip() for i in s]

def get_score(models, classifier, testSet):
    ## Takes a linear regression classifier already trained for scoring similarity between two sentences based on the model
    ## Returns predicted scores for the input dataset
    print 'Computing feature vectors directly through model.pairFeatures() ...'
    feature_list = encoders.pairFeatures(models, process(testSet[0]), process(testSet[1]))
    testF = np.asarray( feature_list )
    #index = [i for i, j in enumerate(testF) if j ==  errorFlag]
    #testF = np.asarray([x for i, x in enumerate(testF) if i not in index])

    r = np.arange(1,6) # the classifier is trained to give probabilities for label 1, 2, 3, 4, 5
    res = classifier.predict_proba(testF, verbose=0)
    print res
    yhat = np.dot(classifier.predict_proba(testF, verbose=0), r)
    return yhat

if __name__ == '__main__':
    print '\nWelcome to Automatic Short Answer Grading system. \n'
    models, classifier = loadStoredClassifier(classifier_map['features on sts2014'])

    while True:
        try:
            goldA = raw_input('Enter gold Answer: \n')
            studA = raw_input('Enter students\'s Answer: \n')

            testSet = [[goldA], [studA]]
            result = get_score(models, classifier, testSet)
            print 'score: ', result
        except KeyboardInterrupt:
            raise
        except Exception, e:
            print "Error: %s" % e
            print 'Please correct the error and try again.'
