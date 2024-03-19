from flask import Flask, jsonify, request
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

app = Flask(__name__)

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)

# Train a simple model
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        query_df = pd.DataFrame(json_)
        prediction = model.predict(query_df)
        return jsonify({'prediction': list(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)

