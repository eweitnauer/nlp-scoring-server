from flask import jsonify, request
import requests
from encoders.classifier import CachedClassifier
from nltk import sent_tokenize
import math
import json

class PreloadError(Exception):
	pass

''' Scoring a paragraph by comparing it to idea units sentence by sentence.
'''
class ScoreParController(object):
	def __init__(self, max_par_len=1000, max_targets=5, max_responses=5):
		self.max_par_len = max_par_len
		self.max_targets = max_targets
		self.max_responses = max_responses

	def authenticate(self):
		req = request.args if request.method == "GET" else request.form
		api_key = req.get('api_key', None)
		if api_key:
			payload = {"apiKey": api_key}
			r = requests.get('http://ultron.psych.purdue.edu/checkKey', params=payload)
			if r.text != "1":
				raise self.errors.append("invalid api key")
		else:
			pass # CHANGEME FOR PRODUCTION
			#raise self.errors.append("api key not supplied")

	def extractInfo(self):
		req = request.args if request.method == "GET" else request.form

		# targets (json array of idea units)
		self.targets = req.get('targets', None)
		if not self.targets: self.errors.append("'targets' not present")
		elif len(self.targets) > self.max_par_len: self.errors.append("'targets' too long")
		try:
			self.targets = json.loads(self.targets)
			if not isinstance(self.targets, list): self.errors.append("'targets' has wrong format")
			if len(self.targets) > self.max_targets: self.error.append("too many 'targets'")
		except JSONDecodeError:
			self.errors.append('Parsing error for targets')

		# response (sentences separated by full-stops)
		self.response = req.get('response', None)
		if not self.response: self.errors.append("'response' not present")
		elif len(self.response) > self.max_par_len: self.errors.append("'response' too long")
		else:
			self.sentences = sent_tokenize(self.response)
			if not isinstance(self.sentences, list): self.errors.append("'response' counld not be split into sentences")
			if len(self.sentences) > self.max_responses: self.errors.append("too many response sentences")
			if len(self.sentences) == 0: self.errors.append("found no sentences in response")

		# model_names (comma separated)
		self.model_names = req.get('models', None)
		if not self.model_names: self.errors.append("'models' not present")
		else:
			self.model_names = [name.strip() for name in self.model_names.split(',')]

		# classifier
		self.classifier_name = req.get('classifier', None)
		if self.classifier_name == 'untrained': self.classifier_name = None
		if self.errors: raise ValueError()

	def score(self):
		try:
			cache_only = True if self.classifier_name else False
			classifier = CachedClassifier(self.model_names, self.classifier_name, from_cache_only=cache_only)
			if not classifier: raise PreloadError()
			scores = []
			for resp in self.sentences:
				row = []
				for target in self.targets:
					score = classifier.get_score([target], [resp])[0]
					if math.isnan(score): score = "NaN"
					row.append(score)
				scores.append(row)
			return scores
		except PreloadError:
			self.errors.append("trained classfiers have to be preloaded before they can be used")
			raise
		except Exception as ex:
			self.errors.append("internal error: " + ex.message)
			raise

	def route(self):
		self.errors = []
		resp = {
			'name': "Automated Scoring",
		    'version': "1.2"
		}
		try:
			self.extractInfo()
			if self.model_names: resp['models'] = ", ".join(self.model_names)
			if self.classifier_name: resp['classifier'] = self.classifier_name
			self.authenticate()
			resp['sentences'] = self.sentences
			resp['scores'] = self.score()
			return jsonify(resp)
		except:
			resp['errors'] = self.errors
			return jsonify(resp)
