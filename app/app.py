from flask import Flask, request, render_template, jsonify, send_file
import pickle
import os
import logging
import lime
import lime.lime_text

# Set up logging for error handling
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained models and vectorizers
MODEL_PATHS = {
    'voting': 'mdl.pkl',
    'logistic': 'mdl_logistic.pkl',
    'random_forest': 'mdl_random.pkl',
    'naive_bayes': 'nb_mdl.pkl',
    'voting_cnt':'mdl_voting_cnt.pkl'
}
VECTORIZER_PATHS = {
    'count_vectorizer': 'count_vectorizer.pkl',
    'tfidf_vectorizer': 'nb_vectorizer.pkl',
    'logistic': 'tfidf_vectorizer.pkl',
    'voting_cnt_pkl':'mdl_cnt_vectorizer.pkl'
}

# Load vectorizers
vectorizers = {}
for vec_name, vec_path in VECTORIZER_PATHS.items():
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Vectorizer file '{vec_path}' not found.")
    with open(vec_path, 'rb') as vectorizer_file:
        vectorizers[vec_name] = pickle.load(vectorizer_file)

# Load models with accuracy
models = {}
model_accuracies = {}
for model_name, model_path in MODEL_PATHS.items():
    if not os.path.exists(model_path):
        logging.warning(f"Model file '{model_path}' not found. Skipping this model.")
        continue
    with open(model_path, 'rb') as model_file:
        model_data = pickle.load(model_file)
        models[model_name] = model_data['model']
        model_accuracies[model_name] = {
            'train_accuracy': model_data.get('train_accuracy', 'N/A'),
            'test_accuracy': model_data.get('test_accuracy', 'N/A')
        }

# Create a Flask app
app = Flask(__name__)

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Developer information
developer_info = {
    'name': 'Manikandan V',
    'email': 'maniatramco@gmail.com',
    'github': 'https://github.com/Manikandanaz'
}
# Define the home route
@app.route('/')
def home():
    return render_template('index.html', accuracies=model_accuracies,developer=developer_info)  # Pass accuracies to frontend

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the email content and model choice from the JSON request
            data = request.get_json()
            if 'email' not in data or 'model' not in data:
                logging.error("Invalid input. Email content or model choice missing.")
                return jsonify({'error': 'Invalid input. Please provide both email content and model choice.'}), 400

            email_content = data['email']
            model_choice = data['model']

            if not email_content:
                logging.error("Email content is empty.")
                return jsonify({'error': 'Please enter the email content to predict'}), 400

            if model_choice not in models:
                logging.error(f"Invalid model choice: {model_choice}.")
                return jsonify({'error': 'Invalid model choice or model not found. Please select a valid model.'}), 400

            # Load the chosen model
            model = models[model_choice]
            logging.info(f"Using model: {model_choice}")

            #vectorizer = vectorizers['tfidf_vectorizer'] if model_choice in ('naive_bayes', 'voting') else vectorizers['count_vectorizer']

            # Select the appropriate vectorizer for the chosen model
            if model_choice in ('naive_bayes', 'voting'):
                vectorizer = vectorizers['tfidf_vectorizer']
            elif model_choice =='voting_cnt':
                vectorizer = vectorizers['voting_cnt_pkl']
            else:
                vectorizer=vectorizers['count_vectorizer']

            # Preprocess email content
            email_content_vectorized = vectorizer.transform([email_content])

            # Predict if the email is spam or not
            prediction = model.predict(email_content_vectorized)
            result = 'Spam' if prediction[0] == 1 else 'Not Spam'

            # Get model accuracies
            train_accuracy = model_accuracies[model_choice]['train_accuracy']
            test_accuracy = model_accuracies[model_choice]['test_accuracy']

            # Generate LIME explanation
            explainer = lime.lime_text.LimeTextExplainer(class_names=['Not Spam', 'Spam'])

            # LIME requires a function that takes in raw text and returns prediction probabilities
            def predict_probabilities(texts):
                vectorized_texts = vectorizer.transform(texts)
                return model.predict_proba(vectorized_texts)

            # Explain the model's prediction for the given text
            explanation = explainer.explain_instance(email_content, predict_probabilities, num_features=10)
            explanation_file = os.path.join(BASE_DIR, 'lime_explanation.html')
            explanation.save_to_file(explanation_file)

            # Determine matched words contributing to the prediction
            feature_names = vectorizer.get_feature_names_out()
            weights = explanation.as_list()
            matched_words_sorted = [word for word, weight in weights if weight > 0]

            # Prepare the appropriate message for matched words
            if result == 'Spam' and matched_words_sorted:
                matched_words_message = matched_words_sorted[:10]
            elif result == 'Spam':
                matched_words_message = 'No significant spam words found.'
            else:
                matched_words_message = 'No spam words found. Good to go!'

            return jsonify({
                'prediction': result,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'matched_words': matched_words_message,
                'lime_html': f'/lime/lime_explanation.html'  # Send URL for the LIME explanation
            })

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify({'error': 'An error occurred during prediction', 'details': str(e)}), 500

@app.route('/lime/<path:filename>')
def serve_lime_explanation(filename):
    return send_file(os.path.join(BASE_DIR, filename))

if __name__ == '__main__':
    app.run(debug=True)
