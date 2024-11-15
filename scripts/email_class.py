import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json

# Step 1: Read and preprocess data
df = pd.read_csv('./data/spam_or_not_spam.csv')
df.dropna(subset=['email'], inplace=True)

# Defining the feature and label columns
X = df['email']
y = df['label']

'''
# Converting the text data into numerical features using Count Vectorizer
count_vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_count = count_vectorizer.fit_transform(X)

# Using SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled_count, y_resampled_count = smote.fit_resample(X_count, y)

# Train-test split
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled_count, y_resampled_count, test_size=0.3, random_state=42)

# Convert sparse matrix to dense for models that require dense input
X_train_smote = X_train_smote.toarray()
X_test_smote = X_test_smote.toarray()
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.combine import SMOTETomek
# Use TfidfVectorizer for feature extraction
tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2), max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Handle class imbalance with SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_tfidf, y)
with open('./scripts/best_model_params.json', 'r') as json_file:
    best_params = json.load(json_file)
# Train-test split
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Define the voting classifier with Logistic Regression, Random Forest, and GaussianNB
mdl = VotingClassifier(estimators=[
    ('lr', LogisticRegression(**best_params.get("Logistic Regression", {}),class_weight='balanced',random_state=42)),
    #('rf', RandomForestClassifier(**best_params.get("Random Forest", {}),class_weight='balanced',random_state=42)),
    ('gnb', MultinomialNB(alpha=0.5))
], voting='soft')

# Train the model
mdl.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred = mdl.predict(X_test_smote)

# Evaluate model performance
accuracy = accuracy_score(y_test_smote, y_pred)
report = classification_report(y_test_smote, y_pred)

print(f"Accuracy: {accuracy}")
print(report)


# Evaluate on test data
y_pred_test = mdl.predict(X_test_smote)
test_accuracy = accuracy_score(y_test_smote, y_pred_test)
report = classification_report(y_test_smote, y_pred_test)
# Evaluate on training data
y_pred_train = mdl.predict(X_train_smote)
train_accuracy = accuracy_score(y_train_smote, y_pred_train)
# Print evaluation results
print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print(report)

# Save the model and vectorizer along with metadata
model_metadata = {
    'model': mdl,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'vectorizer': tfidf_vectorizer  # Include the vectorizer for consistent preprocessing
}

with open('mdl.pkl', 'wb') as model_file:
    pickle.dump(model_metadata, model_file)

#with open('count_vectorizer.pkl', 'wb') as vectorizer_file:
#    pickle.dump(count_vectorizer, vectorizer_file)
