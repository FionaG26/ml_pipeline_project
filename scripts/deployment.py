from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Save the model and preprocessors
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(selector, 'selector.pkl')

# Load the model and preprocessors
model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
selector = joblib.load('selector.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    data_preprocessed = preprocessor.transform(data)
    data_selected = selector.transform(data_preprocessed)
    prediction = model.predict(data_selected)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
