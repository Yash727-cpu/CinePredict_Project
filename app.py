from pathlib import Path
from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / 'model' / 'movie_success_model.pkl'
FEATURES = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'release_year']

@app.get('/')
def home():
    return render_template('index.html')

@app.post('/api/predict')
def predict():
    if not MODEL_PATH.is_file():
        return jsonify(error='Trained model not found. Place movie_success_model.pkl inside the model folder.'), 503
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='Please submit movie details as JSON.'), 400
    try:
        values = {name: float(data[name]) for name in FEATURES}
    except (KeyError, TypeError, ValueError):
        return jsonify(error='All six features must be valid numbers.'), 400
    if any(not np.isfinite(v) for v in values.values()):
        return jsonify(error='Inputs must be finite numbers.'), 400
    if any(values[k] < 0 for k in ['budget','popularity','vote_count']) or not 1 <= values['runtime'] or not 0 <= values['vote_average'] <= 10 or not 1888 <= values['release_year'] <= 2100:
        return jsonify(error='Check the input ranges and try again.'), 400
    try:
        # Load only model files you generated or trust. joblib/pickle files can execute code.
        model = joblib.load(MODEL_PATH)
        X = pd.DataFrame([values], columns=FEATURES)
        prediction = int(model.predict(X)[0])
        if prediction not in (0, 1):
            return jsonify(error='Model must predict labels 0 or 1.'), 500
        output = {'prediction': prediction, 'label': 'Successful' if prediction else 'Not Successful'}
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            index = list(model.classes_).index(prediction)
            output['confidence'] = float(probabilities[index])
        return jsonify(output)
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify(error='Model could not generate a prediction. Check the server terminal.'), 500

if __name__ == '__main__':
    app.run(debug=True)
