import flask
#import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from app.score_controller import ScoreController

app = Flask(__name__)

CORS(app)

score_controller = ScoreController()
app.add_url_rule('/api/score', 'score', score_controller.route, methods=['GET', 'POST'])

if __name__ == "__main__":
	#if os.environ.get('ENV') == 'PROD':
		app.run(host='0.0.0.0', debug=False, threaded=True)
	#else:
		#app.run(host='localhost', debug=True, threaded=True)

