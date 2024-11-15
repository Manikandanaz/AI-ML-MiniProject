import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,VotingClassifier
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
#read
df = pd.read_csv('./data/spam_or_not_spam.csv')
df.dropna(subset=['email'],inplace=True)
# Defining the feature and label columns
X = df['email']
y = df['label']
with open('./scripts/best_model_params.json', 'r') as json_file:
    best_params = json.load(json_file)

# Load hyperparameters for Logistic Regression
with open('./scripts/best_model_params.json', 'r') as json_file:
    best_params = json.load(json_file)

# TF-IDF Vectorizer with bigrams
tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_tfidf = tfidf_vectorizer.fit_transform(X)


# Handle class imbalance using SMOTETomek
smote = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42,stratify=y_resampled)

# Train Logistic Regression model with balanced class weights
mdl = LogisticRegression( class_weight='balanced', random_state=42)
mdl.fit(X_train, y_train)

# Evaluate on training data
y_pred_train = mdl.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Train Accuracy:", accuracy_train)

# Evaluate on test data
y_pred_test = mdl.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
report = classification_report(y_test, y_pred_test)
print("Test Accuracy:", accuracy_test)
print("Classification Report:\n", report)

# Evaluate model
train_accuracy = accuracy_score(y_train, mdl.predict(X_train))
test_accuracy = accuracy_score(y_test, mdl.predict(X_test))

# Save model and metadata
model_metadata = {
    'model': mdl,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy
}
with open('mdl_logistic.pkl', 'wb') as f:
    pickle.dump(model_metadata, f)

# Save vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
