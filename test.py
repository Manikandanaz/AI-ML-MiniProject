import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# Step 1: Load dataset
df = pd.read_csv('./data/healthcare_dataset.csv')
print("Dataset loaded successfully")
print(df.head())  # Display the first few rows to verify

# Clean and preprocess dataset
df.columns = df.columns.str.lower().str.replace(' ', '_')
df['date_of_admission'] = pd.to_datetime(df['date_of_admission'], errors='coerce')
df['discharge_date'] = pd.to_datetime(df['discharge_date'], errors='coerce')
df['length_of_stay'] = (df['discharge_date'] - df['date_of_admission']).dt.days
df['admission_year'] = df['date_of_admission'].dt.year
df['admission_month'] = df['date_of_admission'].dt.month
# Check for rows with invalid date parsing or inconsistencies
invalid_dates = df[df['date_of_admission'].isnull() | df['discharge_date'].isnull()]
print("Invalid dates:")
print(invalid_dates)

# Define encoding functions
def encode_features(dataframe, cols):
    labelencoder = LabelEncoder()
    for column in cols:
        if dataframe[column].dtype == 'object':
            dataframe[column] = labelencoder.fit_transform(dataframe[column])
    return dataframe

def apply_one_hot_encoding(dataframe, cols):
    # Initialize the encoder
    encoder = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')

    # Fit and transform the specified columns
    encoded_data = encoder.fit_transform(dataframe[cols])

    # Create a DataFrame with the encoded data
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cols), index=dataframe.index)

    # Drop the original columns and concatenate the encoded DataFrame
    dataframe = dataframe.drop(columns=cols)
    dataframe = pd.concat([dataframe, encoded_df], axis=1)

    return dataframe

# Apply one-hot encoding to categorical columns
df.drop(columns=['gender','name', 'doctor', 'hospital','insurance_provider','room_number'],inplace=True)
category_col = df.select_dtypes(include=['object', 'category']).columns.tolist()
onehot_df = apply_one_hot_encoding(df, category_col)
print(f"Categorical columns being one-hot encoded: {category_col}")
print("Dataset after one-hot encoding:")
print(onehot_df.head())

# Step 2: Select features and target
X = onehot_df[['age', 'billing_amount', 'medical_condition_Cancer', 'admission_year', 'medication_Lipitor', 'medication_Penicillin','admission_type_Urgent','admission_type_Emergency', 'length_of_stay', 'medical_condition_Obesity', 'admission_month']]
y = onehot_df[['test_results_Abnormal', 'test_results_Inconclusive', 'test_results_Normal']]

# Standardize the features
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the scaler on the training data
X[['billing_amount']] = scaler.fit_transform(X[['billing_amount']])

# Fit and transform the scaler on the training data
X[['age']] = scaler.fit_transform(X[['age']])

# Fit and transform the scaler on the training data
X[['admission_year']] = scaler.fit_transform(X[['admission_year']])

X.to_csv('tst.csv',index=False)
# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
print("Features and target split into train and test sets successfully",X_train.head(1))

# Step 4: Train K-Neighbors Classifier model
knn = KNeighborsClassifier(algorithm='auto', n_neighbors=3, p=2, weights='distance')
knn.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))