from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load trained models and preprocessing pipeline
models = {
    'knn': joblib.load('models/knn_model.pkl'),
    'stacking': joblib.load('models/stacking_model.pkl'),
    'random_forest': joblib.load('models/random_forest_model.pkl')
}
preprocessor = joblib.load('models/preprocessor.pkl')

# Initialize Flask app
app = Flask(__name__)

# Route for home
@app.route('/')
def home():
    return render_template('index.html', prediction_result=None)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from form submission
        input_data = {key: float(value) for key, value in request.form.items()}
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400

        # Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess the input features
        processed_data = preprocessor.transform(input_df)

        # Get predictions from all models
        predictions = {
            'knn': int(models['knn'].predict(processed_data)[0]),
            'stacking': int(models['stacking'].predict(processed_data)[0]),
            'random_forest': int(models['random_forest'].predict(processed_data)[0])
        }

        # Prepare prediction result
        if predictions['knn'] == 1 or predictions['stacking'] == 1 or predictions['random_forest'] == 1:
            result_message = "\u2764\ufe0f You are at risk of heart disease \u1f622"
            result_color = "danger"
        else:
            result_message = "\u2764\ufe0f You are not having symptoms of heart disease \u1f604"
            result_color = "success"

        return render_template('index.html', prediction_result=result_message, result_color=result_color)

    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {str(e)}", result_color="danger")

if __name__ == '__main__':
    app.run(debug=True)
