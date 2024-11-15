import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,VotingClassifier
import numpy as np
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
#read
df = pd.read_csv('./data/spam_or_not_spam.csv')
df.dropna(subset=['email'],inplace=True)
# Defining the feature and label columns
X = df['email']
y = df['label']

# Converting the text data into numerical features using Count Vectorizer
count_vectorizer = CountVectorizer(stop_words='english',lowercase=True,max_features=10000)
X_count = count_vectorizer.fit_transform(X)

# Using SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled_count, y_resampled_count = smote.fit_resample(X_count, y)
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled_count, y_resampled_count, test_size=0.3, random_state=42,stratify=y_resampled_count)


mdl=VotingClassifier(estimators=[
            ('lr', LogisticRegression(random_state=42,class_weight='balanced')),
            ('rf', RandomForestClassifier(random_state=42,class_weight='balanced')),
            ('gnb', MultinomialNB())
        ], voting='soft')
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
print(train_accuracy,test_accuracy)

with open('mdl_voting_cnt.pkl', 'wb') as f:
    pickle.dump(model_metadata, f)

# Save vectorizer
with open('mdl_cnt_vectorizer.pkl', 'wb') as f:
    pickle.dump(count_vectorizer, f)