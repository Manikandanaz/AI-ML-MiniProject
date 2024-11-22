from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, BaggingClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, roc_auc_score

# Load dataset and split into features and target
df = pd.read_csv('data/heart.csv')
df.rename(columns={
            'age': 'Age',
            'sex': 'Gender',
            'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure',
            'chol': 'Cholesterol',
            'fbs': 'Fasting Blood Sugar',
            'restecg': 'Resting ECG Results',
            'thalach': 'Max Heart Rate Achieved',
            'exang': 'Exercise-Induced Angina',
            'oldpeak': 'ST Depression',
            'slope': 'ST Slope',
            'ca': 'Major Vessels Blocked',
            'thal': 'Thalassemia',
            'target': 'Heart Disease'
        }, inplace=True)
df.rename(columns={
            'Age': 'age',
            'Gender': 'gender',
            'Chest Pain Type': 'chest_pain_type',
            'Resting Blood Pressure': 'resting_blood_pressure',
            'Cholesterol': 'cholesterol',
            'Fasting Blood Sugar': 'fasting_blood_sugar',
            'Resting ECG Results': 'resting_ecg_results',
            'Max Heart Rate Achieved': 'max_heart_rate',
            'Exercise-Induced Angina': 'exercise_induced_angina',
            'ST Depression': 'st_depression',
            'ST Slope': 'st_slope',
            'Major Vessels Blocked': 'major_vessels_blocked',
            'Thalassemia': 'thalassemia',
            'Heart Disease': 'heart_disease'
        }, inplace=True)
print(df.columns)
X = df[['age', 'cholesterol', 'max_heart_rate', 'st_depression', 'chest_pain_type', 'major_vessels_blocked', 'thalassemia']]
y = df['heart_disease']

# Preprocessing pipeline
numerical_features = ['age', 'cholesterol', 'max_heart_rate', 'st_depression']
categorical_features = ['chest_pain_type', 'major_vessels_blocked', 'thalassemia']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for KNN
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
knn_grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
knn_grid_search.fit(preprocessor.fit_transform(X_train), y_train)
knn_model = knn_grid_search.best_estimator_
joblib.dump(knn_model, 'models/knn_model.pkl')

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(preprocessor.transform(X_train), y_train)
rf_model = rf_grid_search.best_estimator_
joblib.dump(rf_model, 'models/random_forest_model.pkl')

# Train and save Stacking model
stacking_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42, **rf_grid_search.best_params_)),
        ('svc', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(**knn_grid_search.best_params_)),
        ('bagging', BaggingClassifier(random_state=42, n_estimators=50))
    ],
    final_estimator=GradientBoostingClassifier(random_state=42)
)
stacking_model.fit(preprocessor.transform(X_train), y_train)
joblib.dump(stacking_model, 'models/stacking_model.pkl')

# Calculate and print train and test accuracy for all models
print("Train and Test Accuracy of Models:")

# KNN Model
knn_train_acc = accuracy_score(y_train, knn_model.predict(preprocessor.transform(X_train)))
knn_test_acc = accuracy_score(y_test, knn_model.predict(preprocessor.transform(X_test)))
print(f"KNN Model - Training Accuracy: {knn_train_acc:.3f}, Testing Accuracy: {knn_test_acc:.3f}")

# Random Forest Model
rf_train_acc = accuracy_score(y_train, rf_model.predict(preprocessor.transform(X_train)))
rf_test_acc = accuracy_score(y_test, rf_model.predict(preprocessor.transform(X_test)))
print(f"Random Forest Model - Training Accuracy: {rf_train_acc:.3f}, Testing Accuracy: {rf_test_acc:.3f}")

# Stacking Model
stacking_train_acc = accuracy_score(y_train, stacking_model.predict(preprocessor.transform(X_train)))
stacking_test_acc = accuracy_score(y_test, stacking_model.predict(preprocessor.transform(X_test)))
print(f"Stacking Model - Training Accuracy: {stacking_train_acc:.3f}, Testing Accuracy: {stacking_test_acc:.3f}")

# Save the preprocessor
joblib.dump(preprocessor, 'models/preprocessor.pkl')

# Load trained models and preprocessing pipeline
models = {
    'knn': joblib.load('models/knn_model.pkl'),
    'stacking': joblib.load('models/stacking_model.pkl'),
    'random_forest': joblib.load('models/random_forest_model.pkl')
}
preprocessor = joblib.load('models/preprocessor.pkl')