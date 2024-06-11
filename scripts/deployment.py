from flask import Flask, request, jsonify
import joblib
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and preprocessors
model = joblib.load('../models/best_model.pkl')
preprocessor = joblib.load('../models/preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        
        # Input validation
        if not isinstance(data, list):
            return jsonify({'error': 'Data must be a list of feature values'}), 400
        
        data_preprocessed = preprocessor.transform(data)
        data_selected = selector.transform(data_preprocessed)
        prediction = model.predict(data_selected)
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logger.error(f'Error during prediction: {e}')
        return jsonify({'error': 'An unexpected error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
