import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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
# Converting the text data into numerical features using Count Vectorizer
count_vectorizer = CountVectorizer(stop_words='english',lowercase=True)
X_count = count_vectorizer.fit_transform(X)

# Using SMOTE to oversample the minority class
smote = SMOTETomek(random_state=42)
X_resampled_count, y_resampled_count = smote.fit_resample(X_count, y)

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled_count, y_resampled_count, test_size=0.3, random_state=42)

if hasattr(X_train_smote, 'toarray'):
        X_train_smote = X_train_smote.toarray()
if hasattr(X_test_smote, 'toarray'):
        X_test_smote = X_test_smote.toarray()

mdl=RandomForestClassifier(**best_params.get("Random Forest", {}),class_weight='balanced',random_state=42)
# Train the model
mdl.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred = mdl.predict(X_test_smote)

# Evaluate model performance
accuracy = accuracy_score(y_test_smote, y_pred)
report = classification_report(y_test_smote, y_pred)

train_accuracy = accuracy_score(y_train_smote, mdl.predict(X_train_smote))
test_accuracy = accuracy_score(y_test_smote, mdl.predict(X_test_smote))
# Save model and metadata
model_metadata = {
    'model': mdl,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy
}
with open('mdl_random.pkl', 'wb') as f:
    pickle.dump(model_metadata, f)

# Save vectorizer
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(count_vectorizer, f)