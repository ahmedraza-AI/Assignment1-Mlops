from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('iris_model.h5')
scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions based on input data."""
    try:
        data = request.get_json()
        features = data['features']

        # Scale the features
        features = scaler.transform([features])

        # Make a prediction
        prediction = model.predict(features)

        # Get the predicted class (0, 1, or 2)
        predicted_class = np.argmax(prediction)

        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
