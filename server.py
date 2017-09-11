from flask import Flask, jsonify, request, redirect, url_for, render_template
from flask_cors import CORS
from app.score_controller import ScoreController

models = {
  'quickscore': ['untrained']
, 'bow': ['untrained']
, 'infersent': ['untrained', 'infersent-sick']
, 'bow, feature_based': ['bow_fb-sick']
}

app = Flask(__name__)
CORS(app)

@app.route('/')
@app.route('/index')
def index():
	return redirect(url_for('score'))

@app.route('/score')
def score():
	return render_template('score.html', models=models)

score_controller = ScoreController()
app.add_url_rule('/api/score', 'api_score', score_controller.route, methods=['GET', 'POST'])

if __name__ == "__main__":
	app.run(host='127.0.0.1', port=5001, debug=False, threaded=False)
