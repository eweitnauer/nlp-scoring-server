from encoders.classifier import Classifier

## Uncomment one of the following lines to load that classfier

# untrained:
classifier = Classifier(model_names=['quickscore'], classifier_file=None)
#classifier = Classifier(model_names=['bow'], classifier_file=None)
#classifier = Classifier(model_names=['infersent'], classifier_file=None)
#classifier = Classifier(model_names=['quickscore', 'bow'], classifier_file=None)

# trained:
#classifier = Classifier(model_names=['bow', 'feature_based'], classifier_file='bow_fb-sick.h5')
#classifier = Classifier(model_names=['infersent'], classifier_file='infersent-sick.h5')

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
