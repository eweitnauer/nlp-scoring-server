import numpy as np
from sklearn.utils import shuffle
import math

def load_data_exp1A():
	return load_data(loc='data/local/IES-2Exp1A_AVG.txt')

def load_data_exp2A():
	return load_data(loc='data/local/IES-2Exp2A_AVG.txt')

def load_data(loc, skip_first_line=True):
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

###################################################################################################
# print "Training BOW+LR model on exp1A"
# from encoders.classifier import Classifier;
# c = Classifier(['bow']);
# train, dev, test = load_data_exp1A();
# print "Untrained performance: "
# c.test(test)
# # ************ SUMMARY ***********
# # Test data size: 784
# # Test Pearson: 0.793993001472
# # Test Spearman: 0.782309363231
# # Test MSE: 0.103506707689
# # ********************************
# print "Training..."
# c.classifier = c.train(train, dev)
# print "Trained performance: "
# c.test(test)
# # ************ SUMMARY *********** 2 sigmoid layers
# # Test data size: 784
# # Test Pearson: 0.88036287173
# # Test Spearman: 0.84489853873
# # Test MSE: 0.0512770979582
# # ********************************

###################################################################################################
# print "Training BOW model on exp2A"
# from encoders.classifier import Classifier;
# c = Classifier(['bow']);
# train, dev, test = load_data_exp2A();
# print "Untrained performance: "
# c.test(test)
# # ************ SUMMARY ***********
# # Test data size: 802
# # Test Pearson: 0.949398561489
# # Test Spearman: 0.839376854333
# # Test MSE: 0.0257896074761
# # ********************************
# print "Training..."
# c.classifier = c.train(train, dev)
# print "Trained performance: "
# c.test(test)
# # ************ SUMMARY *********** 2 sig layers
# # Test data size: 802
# # Test Pearson: 0.987660586873
# # Test Spearman: 0.82193199993
# # Test MSE: 0.00545393423369
# # ********************************

###################################################################################################
# print "Training infersent based model on college"
# from encoders.classifier import Classifier;
# c = Classifier(['infersent']);
# train, dev, test = load_data_college();
# N = 500
# clip_data(train, N)
# clip_data(dev, N/2)
# clip_data(test, N)
# print "Untrained performance: "
# c.test(test)
# print "Training..."
# c.classifier = c.train(train, dev)
# print "Trained performance: "
# c.test(test)

###################################################################################################
# print "Training bow + feature based model on exp1A"
# from encoders.classifier import Classifier;
# c = Classifier(['bow', 'feature_based']);
# train, dev, test = load_data_exp1A();
# print "Training..."
# c.classifier = c.train(train, dev)
# print "Trained performance: "
# c.test(test)
# # ************ SUMMARY *********** 2 sigmoid layers
# # Test data size: 784
# # Test Pearson: 0.954879481061
# # Test Spearman: 0.877234397781
# # Test MSE: 0.0200705280446
# # ********************************

# ###################################################################################################
# print "Training bow + feature based model on exp2A"
# from encoders.classifier import Classifier;
# c = Classifier(['bow', 'feature_based']);
# train, dev, test = load_data_exp2A();
# print "Training..."
# c.classifier = c.train(train, dev)
# print "Trained performance: "
# c.test(test)
# # ************ SUMMARY *********** 2 sigmoid layers
# # Test data size: 802
# # Test Pearson: 0.981349343557
# # Test Spearman: 0.822492784817
# # Test MSE: 0.00820888379743
# # ********************************
