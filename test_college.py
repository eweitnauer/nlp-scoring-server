import numpy as np
from sklearn.utils import shuffle
import math

def load_data_college(loc='data/local/CollegeOldData_HighAgreementPartialScoring.txt', skip_first_line=True):
		allA, allB, allS = [],[],[]

		with open(loc, 'rb') as f:
				for line in f:
						text = line.strip().split('\t')
						allA.append(text[1])
						allB.append(text[2])
						allS.append(text[3])

		if skip_first_line:
			allA = allA[1:]
			allB = allB[1:]
			allS = allS[1:]

		allS = [float(s) for s in allS]

		## remove useless datapoints
		index = [i for i, j in enumerate(allB) if (j == "empty" or ("I don't" in j))]
		print("No. of empty and 'i don't know' cases': " , len(index))
		index = [i for i, j in enumerate(allB) if (j == "empty" or ("I don't" in j) or ("\n" in j) or ('\"' in j) )]
		print("No. of empty and 'i don't know' , 'i don't' and multi-line (suspicious) cases': " , len(index))
		allA = np.asarray([x for i, x in enumerate(allA) if i not in index])
		allB = np.asarray([x for i, x in enumerate(allB) if i not in index])
		allS = np.asarray([x for i, x in enumerate(allS) if i not in index])
		print("Average length of sentenceA ", sum(map(len, allA))/float(len(allA)))
		print("Average length of sentenceB ", sum(map(len, allB))/float(len(allB)))
		print len(allA), len(allB), len(allS)

		## shuffle the data
		allS, allA, allB = shuffle(allS, allA, allB, random_state=12345)

		## split into 45% train, 5% dev and remaining ~50% test
		trainA, devA, testA = allA[0 : int(math.floor(0.45 * len(allA)))], allA[int(math.floor(0.45 * len(allA))) + 1 : int(math.floor(0.5 * len(allA))) ], allA[int(math.floor(0.5 * len(allA))) + 1 : ]
		trainB, devB, testB = allB[0 : int(math.floor(0.45 * len(allB)))], allB[int(math.floor(0.45 * len(allB))) + 1 : int(math.floor(0.5 * len(allB))) ], allB[int(math.floor(0.5 * len(allB))) + 1 : ]
		trainS, devS, testS = allS[0 : int(math.floor(0.45 * len(allS)))], allS[int(math.floor(0.45 * len(allS))) + 1 : int(math.floor(0.5 * len(allS))) ], allS[int(math.floor(0.5 * len(allS))) + 1 : ]

		print len(allA)
		print len(trainA)+len(devA)+len(testA)
		print len(trainA), len(devA), len(testA)
		return [trainA, trainB, trainS], [devA, devB, devS], [testA, testB, testS]

def clip_data(data, N):
	data[0] = data[0][0:N]
	data[1] = data[1][0:N]
	data[2] = data[2][0:N]

train, dev, test = load_data_college();
from encoders.classifier import Classifier;

###################################################################################################
c = Classifier(['feature_based']);
print "Training (feature_based)..."
c.classifier = c.train(train, dev)
c.classifier.save('pretrained/classifiers/fb-college.h5')
print "Trained performance (fb): "
c.test(test)
# ************ SUMMARY ***********
# Test data size: 2377
# Test Pearson: 0.780165116222
# Test Spearman: 0.773093078166
# Test MSE: 0.0618294011138
# ********************************

###################################################################################################
c = Classifier(['bow']);
# print "Untrained performance: "
# c.test(test)
# ************ SUMMARY ***********
# Test data size: 2366
# Test Pearson: 0.609879183339
# Test Spearman: 0.647898161358
# Test MSE: 0.146632924546
# ********************************
print "Training (bow)..."
c.classifier = c.train(train, dev)
c.classifier.save('pretrained/classifiers/bow-college.h5')
print "Trained performance (bow): "
c.test(test)
# ************ SUMMARY ***********
# Test data size: 2371
# Test Pearson: 0.815727992469
# Test Spearman: 0.814669747831
# Test MSE: 0.0529697648042
# ********************************

###################################################################################################
c = Classifier(['infersent']);
# N = 2000
# # clip_data(train, N)
# # clip_data(dev, N/2)
# # clip_data(test, N)
# print "Untrained performance: "
# c.test(test)
# # ************ SUMMARY ***********
# # Test data size: 2377
# # Test Pearson: 0.614370830801
# # Test Spearman: 0.704158202389
# # Test MSE: 0.142912132193
# # ********************************
print "Training (infersent)..."
c.classifier = c.train(train, dev)
c.classifier.save('pretrained/classifiers/infersent-college.h5')
print "Trained performance (infersent): "
c.test(test)
# ************ SUMMARY ***********
# Test data size: 2377
# Test Pearson: 0.864566020796
# Test Spearman: 0.85160629806
# Test MSE: 0.0403718484387
# ********************************

# Results with infersent trained on SICK data:
# (actually worse than untrained!)
# ************ SUMMARY ***********
# Test data size: 2377
# Test Pearson: 0.575628460689
# Test Spearman: 0.590724986585
# Test MSE: 0.155956669515
# ********************************
# ************ SUMMARY *********** with 0 score for contradiction cases
# Test data size: 2377
# Test Pearson: 0.425360783695
# Test Spearman: 0.445096454362
# Test MSE: 0.134747540396
# ********************************

###################################################################################################
c = Classifier(['bow', 'feature_based']);
print "Training (bow + feature_based)..."
c.classifier = c.train(train, dev)
c.classifier.save('pretrained/classifiers/bow_fb-college.h5');
print "Trained performance (bow + feature_based): "
c.test(test)
# ************ SUMMARY ***********
# Test data size: 2371
# Test Pearson: 0.851198922554
# Test Spearman: 0.838510775468
# Test MSE: 0.0441557772239
# ********************************
#
# # Results with bow+fb trained on SICK data:
# ************ SUMMARY ***********
# Test data size: 2366
# Test Pearson: 0.455408578396
# Test Spearman: 0.409406750153
# Test MSE: 0.142136707951
# ********************************

###################################################################################################
c = Classifier(['infersent', 'feature_based']);
print "Training (infersent + feature_based)..."
c.classifier = c.train(train, dev)
c.classifier.save('pretrained/classifiers/infersent_fb-college.h5');
print "Trained performance (infersent + feature based): "
c.test(test)
# ************ SUMMARY ***********
# Test data size: 2377
# Test Pearson: 0.886186401746
# Test Spearman: 0.868906840594
# Test MSE: 0.0339921869165
# ********************************
