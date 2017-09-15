from flask import jsonify, request
import requests
from encoders.classifier import CachedClassifier

# need to preload all models with trained classifiers
CachedClassifier(['bow', 'feature_based'], 'bow_fb-sick')
CachedClassifier(['bow', 'feature_based'], 'bow_fb-college')
CachedClassifier(['infersent'], 'infersent-sick')
CachedClassifier(['infersent'], 'infersent-college')
CachedClassifier(['infersent', 'feature_based'], 'infersent_fb-college')
CachedClassifier(['feature_based'], 'fb-college')
CachedClassifier(['bow'], 'bow-college')

# optionally preload untrained models to make the first request faster
CachedClassifier(['bow'])
CachedClassifier(['quickscore'])
CachedClassifier(['infersent'])

class PreloadError(Exception):
	pass

class ScoreController(object):
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
		self.target = req.get('target', None)
		if not self.target: self.errors.append("'target' not present")
		self.response = req.get('response', None)
		if not self.response: self.errors.append("'response' not present")
		self.model_names = req.get('models', None)
		if not self.model_names: self.errors.append("'models' not present")
		else:
			self.model_names = [name.strip() for name in self.model_names.split(',')]
		self.classifier_name = req.get('classifier', None)
		if self.classifier_name == 'untrained': self.classifier_name = None
		if self.errors: raise ValueError()

	def score(self):
		try:
			cache_only = True if self.classifier_name else False
			classifier = CachedClassifier(self.model_names, self.classifier_name, from_cache_only=cache_only)
			if not classifier: raise PreloadError()
			return classifier.get_score([self.target], [self.response])[0]
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
		  'version': "1.1"
		}
		try:
			self.extractInfo()
			if self.model_names: resp['models'] = ", ".join(self.model_names)
			if self.classifier_name: resp['classifier'] = self.classifier_name
			self.authenticate()
			resp['score'] = self.score()
			return jsonify(resp)
		except:
			resp['errors'] = self.errors
			return jsonify(resp)
