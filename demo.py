from encoders.classifier import Classifier

## Uncomment one of the following lines to load that classfier

# unsupervised:
#classifier = Classifier(model_names=['quickscore'], classifier_file=None)
#classifier = Classifier(model_names=['bow'], classifier_file=None)
#classifier = Classifier(model_names=['infersent'], classifier_file=None)
#classifier = Classifier(model_names=['quickscore', 'bow'], classifier_file=None)


#classifier = Classifier(model_names=['bow', 'feature_based'], classifier_file='bow+fb+sts1214.file', use_pickle=True)
#classifier = Classifier(model_names=['feature_based'], classifier_file='fb+sts1214.file')
#classifier = Classifier(model_names=['infersent'], classifier_file='feed+college.file')

print '\nWelcome to Automatic Short Answer Grading system.'
print 'Using models ', classifier.model_names
if classifier.classifier_file: print 'with pretrained classifier ' + classifier.classifier_file
else: print 'without a trained classifier'
print ''

while True:
    try:
        goldA = raw_input('Enter gold Answer: \n')
        studA = raw_input('Enter students\'s Answer: \n')
        print 'score: ', classifier.get_score([goldA], [studA])
    except KeyboardInterrupt:
        raise
    # except Exception, e:
    #     print "Error: %s" % e
    #     print 'Please correct the error and try again.'
