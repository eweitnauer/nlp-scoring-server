from nltk import sent_tokenize

def splitIntoSentences(text, split_on_newline=True, min_sent_words=3, min_sent_chars=5):
	sents = sent_tokenize(text)
	if split_on_newline:
		all_sents = []
		for sent in sents:
			all_sents.extend([s.strip() for s in sent.splitlines()])
	else:
		all_sents = sents
	results = []
	curr = ''
	for sent in all_sents:
		if len(curr) > 0: curr = curr + ' ' + sent
		else: curr = sent
		if len(curr.split(' ')) >= min_sent_words and len(curr) > min_sent_chars:
			results.append(curr)
			curr = ''
	return results
