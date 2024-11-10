from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Step 1: Load the saved model and scaler
with open('knn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Step 2: Initialize Flask app
app = Flask(__name__)

# Step 3: Define home route to display input form with initial values
@app.route('/')
def home():
    # Set initial values for the form fields
    initial_values = {
        'age': 76,
        'billing_amount': 27954,
        'medical_condition_Cancer': 1,
        'admission_year': 2020,
        'medication_Lipitor': 1,
        'medication_Penicillin': 1,
        'admission_type_Urgent': 1,
        'admission_type_Emergency': 0,
        'length_of_stay': 12,
        'medical_condition_Obesity': 1,
        'admission_month': 12
    }
    # Set options for dropdown fields
    dropdown_options = {
        'age': list(range(1, 91)),
        'medical_condition_Cancer': [0, 1],
        'medication_Lipitor': [0, 1],
        'medication_Penicillin': [0, 1],
        'admission_type_Urgent': [0, 1],
        'admission_type_Emergency': [0, 1],
        'medical_condition_Obesity': [0, 1],
        'admission_month': list(range(1, 13)),
        'admission_year': [2019, 2020, 2021, 2022]
    }
    return render_template('index.html', initial_values=initial_values, dropdown_options=dropdown_options)  # Pass initial values and dropdown options to the HTML form

# Step 4: Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        age = float(request.form.get('age', ''))
        billing_amount = float(request.form.get('billing_amount', ''))
        medical_condition_Cancer = int(request.form.get('medical_condition_Cancer', ''))
        admission_year = int(request.form.get('admission_year', ''))
        medication_Lipitor = int(request.form.get('medication_Lipitor', ''))
        medication_Penicillin = int(request.form.get('medication_Penicillin', ''))
        admission_type_Urgent = int(request.form.get('admission_type_Urgent', ''))
        admission_type_Emergency = int(request.form.get('admission_type_Emergency', ''))
        length_of_stay = float(request.form.get('length_of_stay', ''))
        medical_condition_Obesity = int(request.form.get('medical_condition_Obesity', ''))
        admission_month = int(request.form.get('admission_month', ''))

        # Create the feature list in the correct order
        features = np.array([age, billing_amount, medical_condition_Cancer, admission_year, medication_Lipitor,
                             medication_Penicillin, admission_type_Urgent, admission_type_Emergency,
                             length_of_stay, medical_condition_Obesity, admission_month]).reshape(1, -1)

        # Standardize the features using the loaded scaler
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)

        # Convert the prediction to a readable format
        classes = ['Abnormal', 'Inconclusive', 'Normal']
        predicted_class = classes[prediction.argmax()] if len(prediction.shape) > 1 else classes[prediction[0]]

        # Set dropdown options for the form
        dropdown_options = {
            'age': list(range(1, 91)),
            'medical_condition_Cancer': [0, 1],
            'medication_Lipitor': [0, 1],
            'medication_Penicillin': [0, 1],
            'admission_type_Urgent': [0, 1],
            'admission_type_Emergency': [0, 1],
            'medical_condition_Obesity': [0, 1],
            'admission_month': list(range(1, 13)),
            'admission_year': [2019, 2020, 2021, 2022]
        }

        # Return the result to the user
        return render_template('index.html', prediction_text=f'Test Result: {predicted_class}', initial_values=request.form, dropdown_options=dropdown_options)

    except Exception as e:
        # Set dropdown options for the form
        dropdown_options = {
            'age': list(range(1, 91)),
            'medical_condition_Cancer': [0, 1],
            'medication_Lipitor': [0, 1],
            'medication_Penicillin': [0, 1],
            'admission_type_Urgent': [0, 1],
            'admission_type_Emergency': [0, 1],
            'medical_condition_Obesity': [0, 1],
            'admission_month': list(range(1, 13)),
            'admission_year': [2019, 2020, 2021, 2022]
        }

        return render_template('index.html', prediction_text=f'Error: Please make sure all inputs are correct. Details: {e}', initial_values=request.form, dropdown_options=dropdown_options)

# Step 5: Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
