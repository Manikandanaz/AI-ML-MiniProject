from flask import Flask, request, render_template, jsonify
import pickle
import os

# Load the pre-trained models and vectorizer
MODEL_PATHS = {
    'voting': 'mdl.pkl',
    'logistic': 'mdl_logistic.pkl',
    'random_forest': 'mdl_random_forest.pkl'
}
VECTORIZER_PATH = 'count_vectorizer.pkl'

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Vectorizer file '{VECTORIZER_PATH}' not found. Make sure to save the vectorizer.")

with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
    count_vectorizer = pickle.load(vectorizer_file)

# Create a Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Render a simple HTML form for input

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the email content and model choice from the JSON request
        data = request.get_json()
        if 'email' not in data or 'model' not in data:
            return jsonify({'error': 'Invalid input'}), 400

        email_content = data['email']
        model_choice = data['model']

        if not email_content:
            return jsonify({'error': 'Please enter the email content to predict'}), 400

        if model_choice not in MODEL_PATHS:
            return jsonify({'error': 'Invalid model choice. Please select a valid model.'}), 400

        # Load the chosen model
        model_path = MODEL_PATHS[model_choice]
        if not os.path.exists(model_path):
            return jsonify({'error': f"Model file '{model_path}' not found. Make sure to train and save the model."}), 400

        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        # Preprocess email
        email_content_vectorized = count_vectorizer.transform([email_content]).toarray()

        # Predict if the email is spam or not
        prediction = model.predict(email_content_vectorized)

        # Return the prediction result
        result = 'Spam' if prediction[0] == 1 else 'Not Spam'
        return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
