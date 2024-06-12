from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessors
model = joblib.load('models/best_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

@app.route('/')
def index():
    return 'Server is running!'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.json['data']
            print("Received data:", data)  # Add this debug statement
            data_df = pd.DataFrame(data)
            print("Data DataFrame:", data_df)  # Add this debug statement
            data_preprocessed = preprocessor.transform(data_df)
            print("Preprocessed data:", data_preprocessed)  # Add this debug statement
            print("Type of preprocessed data:", type(data_preprocessed))  # Add this debug statement
            prediction = model.predict(data_preprocessed)
            print("Prediction:", prediction)  # Add this debug statement
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)})
    elif request.method == 'GET':
        return 'GET method is not supported for this endpoint. Please use POST method to make predictions.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
