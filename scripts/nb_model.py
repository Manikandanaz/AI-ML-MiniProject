import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Read and preprocess data
df = pd.read_csv('./data/spam_or_not_spam.csv')
df.dropna(subset=['email'], inplace=True)

# Defining the feature and label columns
X = df['email']
y = df['label']

# Use TfidfVectorizer for feature extraction
tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2), max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Handle class imbalance with SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_tfidf, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Evaluate model
train_accuracy = accuracy_score(y_train, nb_model.predict(X_train))
test_accuracy = accuracy_score(y_test, nb_model.predict(X_test))

# Save model and metadata
model_metadata = {
    'model': nb_model,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy
}
with open('nb_mdl.pkl', 'wb') as f:
    pickle.dump(model_metadata, f)

# Save vectorizer
with open('nb_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
