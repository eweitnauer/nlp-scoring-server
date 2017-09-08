def load_data_SICK(loc='data/SICK/', rescale=True):
    """
    Load the SICK semantic-relatedness dataset
    """
    trainA, trainB, devA, devB, testA, testB = [],[],[],[],[],[]
    trainS, devS, testS = [],[],[]

    with open(loc + 'SICK_train.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            trainA.append(text[1])
            trainB.append(text[2])
            trainS.append(text[3])
    with open(loc + 'SICK_trial.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            devA.append(text[1])
            devB.append(text[2])
            devS.append(text[3])
    with open(loc + 'SICK_test_annotated.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            testA.append(text[1])
            testB.append(text[2])
            testS.append(text[3])

    if rescale:
        trainS = [(float(s)-1.0)/4.0 for s in trainS[1:]]
        devS = [(float(s)-1.0)/4.0 for s in devS[1:]]
        testS = [(float(s)-1.0)/4.0 for s in testS[1:]]
    else:
        trainS = [float(s) for s in trainS[1:]]
        devS = [float(s) for s in devS[1:]]
        testS = [float(s) for s in testS[1:]]


    return [trainA[1:], trainB[1:], trainS], [devA[1:], devB[1:], devS], [testA[1:], testB[1:], testS]

def clip_data(data, N):
	data[0] = data[0][0:N]
	data[1] = data[1][0:N]
	data[2] = data[2][0:N]

###################################################################################################
# print "Training BOW+LR model on SICK"
# from encoders.classifier import Classifier;
# c = Classifier(['bow']);
# train, dev, test = load_data_SICK();
# print "Untrained performance: "
# c.test(test)
# # ************ SUMMARY ***********
# # Test data size: 4927
# # Test Pearson: 0.694481553078
# # Test Spearman: 0.577979770056
# # Test MSE: 0.0298842735276
# # ********************************
# print "Training..."
# c.classifier = c.train(train, dev)
# print "Trained performance: "
# c.test(test)
# # ************ SUMMARY *********** WITH 1 LINEAR LAYER
# # Test data size: 4927
# # Test Pearson: 0.738651127214
# # Test Spearman: 0.627919944578
# # Test MSE: 0.0186701774748
# # ********************************
# # ************ SUMMARY *********** WITH 1 SIGMOID LAYER
# # Test data size: 4927
# # Test Pearson: 0.746720520786
# # Test Spearman: 0.625170110855
# # Test MSE: 0.0181161723794
# # ********************************
# # ************ SUMMARY *********** WITH 2 SIGMOID LAYERS
# # Test data size: 4927
# # Test Pearson: 0.768042370046
# # Test Spearman: 0.66275561178
# # Test MSE: 0.0168185726496
# # ********************************

###################################################################################################
print "Training infersent based model on SICK"
from encoders.classifier import Classifier;
c = Classifier(['infersent']);
train, dev, test = load_data_SICK();
N = 2000
# clip_data(train, N)
# clip_data(dev, N/2)
# clip_data(test, N)
print "Untrained performance: "
c.test(test)
# # ************ SUMMARY ***********
# # Test data size: 2000
# # Test Pearson: 0.705948727038
# # Test Spearman: 0.638738528351
# # Test MSE: 0.0255961994753
# # ********************************
# print "Training..."
# c.classifier = c.train(train, dev)
# print "Saving classifier as infersent_scaled-sick.h5"
# c.classifier.save('pretrained/classifiers/infersent_scaled-sick.h5')
# print "Trained performance: "
# c.test(test)
# ************ SUMMARY *********** 2 layers; scaled to 0..1; saved as infersent-sick.h5
# Test data size: 4927
# Test Pearson: 0.872362947586
# Test Spearman: 0.823647106453
# Test MSE: 0.0154452371796
# ********************************

###################################################################################################
# print "Training bow + feature based model on SICK"
# from encoders.classifier import Classifier;
# c = Classifier(['bow', 'feature_based']);
# train, dev, test = load_data_SICK();
# print "Training..."
# c.classifier = c.train(train, dev)
# c.classifier.save('pretrained/classifiers/bow_fb_scaled-sick.h5')
# print "Trained performance: "
# c.test(test)
# # ************ SUMMARY *********** 2 layers; scaled to 0...1; saved as bow_fb_scaled-sick.h5
# # Test data size: 4927
# # Test Pearson: 0.775201503039
# # Test Spearman: 0.685546582637
# # Test MSE: 0.0261244947089
# # ********************************
