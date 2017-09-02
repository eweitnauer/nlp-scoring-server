import re
from spell import spellChecker
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer
from numpy import nan as NaN

class quickScore(object):
    stemmer = None
    stoplist = None
    spell_checker = None

    def __init__(self):
        print("quickScore init: initializing the stemmer, stoplist and spellchecker")
        self.stemmer = SnowballStemmer('english')
        self.stoplist = stopwords.words('english')
        self.spell_checker = spellChecker()
        return

    def synonyms(self, word):
        syns = set()
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                syns.add(str(lemma.name()))
        return syns

    def stem(self, wordSet):
        stemmedSet = set()
        for word in wordSet:
            stemmedSet.add(self.stemmer.stem(word))
        return stemmedSet


    def sentence_similarity(self, sentenceA, sentenceB, stemming=True):
        if sentenceA == sentenceB: return 1
        exactMatch , corMatch, synMatch, keywords = self.pairFeatures(sentenceA, sentenceB, stemming)
        if keywords==0: return NaN
        return (exactMatch + corMatch + synMatch) * 1.0 / keywords

    def pairFeatures(self, sentenceA, sentenceB, stemming=True):
        ## Note that in quickScore, there is a difference between sentenceA and senteceB
        ## sentenceA is considered as the target (correct response)
        ## sentenceB is considered as the student response to be evaluated

        ##preprocess sentenceA
        keywords = re.findall('[a-z_]+', sentenceA.lower())
        keywords = [i for i in  keywords if i not in self.stoplist]
        if len(keywords) == 0:
            return 0,0,0,0;
        ##preprocess sentenceB
        exactsentenceB = re.findall('[a-z_]+', sentenceB.lower())
        exactsentenceB = [i for i in  exactsentenceB if i not in self.stoplist]
        exactsentenceB = set(exactsentenceB)
        correctedsentenceB = set()
        for i in  exactsentenceB:
            candidates = self.spell_checker.spellCorrect(i)
            for word in candidates:
                #print ("word "+ word + " is added as a correct candidate for " + i)
                correctedsentenceB.add(word)
        exactMatch, corMatch, synMatch = 0,0,0
        if not stemming:
            for word in keywords:
                syns = self.synonyms(word)
                if word in exactsentenceB:
                    exactMatch = exactMatch + 1
                elif( word in correctedsentenceB):
                    corMatch = corMatch + 1
                elif (len( correctedsentenceB   & syns ) >= 1 ):
                        synMatch = synMatch + 1
        else:
            exactsentenceB = self.stem(exactsentenceB)
            correctedsentenceB = self.stem(correctedsentenceB)
            for word in keywords:
                syns = self.synonyms(word)
                syns = self.stem(syns)
                word = self.stemmer.stem(word)
                if word in exactsentenceB:
                    exactMatch = exactMatch + 1
                elif( word in correctedsentenceB):
                    corMatch = corMatch + 1
                elif (len( correctedsentenceB   & syns ) >= 1 ):
                        synMatch = synMatch + 1
        # print("\nExact matches: " + str(exactMatch)+
        #     "\nCorrected mathces: " + str(corMatch)+
        #     "\nSynonymy matches: " + str(synMatch))

        return exactMatch , corMatch , synMatch, len(keywords)
