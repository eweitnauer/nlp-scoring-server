from encoders import encoders

# infersent creates a 4096 dim. vector embedding for a sentence
# we can use the cosine to judge similarity, but training a classifier for
# our specific tasks works much better
infersent = encoders.loadInfersent()
print "infersent", infersent.encode(['the cat eats a mouse'], verbose=True)

# adds up 300 dim. vector embeddings for each word in the sentences
# we can use the cosine to judge similarity, but training a classifier for
# our specific tasks works much better
bow = encoders.loadBow()
print "bow", bow.encode(['the cat eats a mouse'])

# the 'old', internal algorithm that just looks at overlap between words,
# using stemming and synonyms. Can only be used for comparing sentences directly.
qs = encoders.loadQuickScore()
print "quickscore", qs.sentence_similarity('the cats eat a mouse', 'the catt eats the mouse', stemming=True)

# builds a vector of features describing both sentences; this can only be used
# together with a classifier; includes the output of quickScore
fb = encoders.loadFeatureBased()
print "feature based", fb.pairFeatures('the cats eat a mouse', 'the catt eats the mouse')
