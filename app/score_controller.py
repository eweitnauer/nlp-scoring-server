from flask import jsonify, request
import requests
from encoders.classifier import CachedClassifier

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
		if self.errors: raise ValueError()

	def score(self):
		try:
			classifier = CachedClassifier(self.model_names, self.classifier_name)
			return classifier.get_score([self.target], [self.response])[0]
		except Exception as err:
			raise
			raise self.errors.append("internal error: could not apply classifier")

	def route(self):
		self.errors = []
		try:
			self.extractInfo()
			self.authenticate()
			score = self.score()
			return jsonify(
				{ 'name': "Automated Scoring",
		      'version': "1.1",
		      'errors': [],
		      'score': score,
		      'models': ", ".join(self.model_names),
		      'classifier': self.classifier_name
		    });
		except:
			raise
			return jsonify({'errors': self.errors})
