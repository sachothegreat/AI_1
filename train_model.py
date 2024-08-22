import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import xgboost as xgb
import os

# Define the required features
REQUIRED_FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'D2', 'DFA', 'spread1', 'spread2', 'PPE'
]

# Load the dataset from the uploads directory
file_path = os.path.join('uploads', 'parkinsons.data')
df = pd.read_csv(file_path)

# Check if all required features are present
missing_columns = [col for col in REQUIRED_FEATURES if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

# Preprocess the data
X = df[REQUIRED_FEATURES]
y = df['status']

# Save the feature names
with open(os.path.join('uploads', 'feature_names.txt'), 'w') as f:
    f.write(','.join(REQUIRED_FEATURES))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
scaler_path = os.path.join('uploads', 'scaler.pkl')
joblib.dump(scaler, scaler_path)

# TensorFlow Model
model_tf = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model_tf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the TensorFlow model
model_tf.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_data=(X_test_scaled, y_test), verbose=1)

# Save the TensorFlow model
model_tf_path = os.path.join('uploads', 'parkinsons_model.h5')
model_tf.save(model_tf_path)

# scikit-learn Model (Logistic Regression)
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(random_state=42, max_iter=1000)
model_lr.fit(X_train_scaled, y_train)

# Save the scikit-learn model
model_lr_path = os.path.join('uploads', 'parkinsons_logreg.pkl')
joblib.dump(model_lr, model_lr_path)

# XGBoost Model
model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xgb.fit(X_train_scaled, y_train)

# Save the XGBoost model
model_xgb_path = os.path.join('uploads', 'parkinsons_xgb_model.json')
model_xgb.save_model(model_xgb_path)

print(f"Models and feature names saved in: {scaler_path}, {model_tf_path}, {model_lr_path}, {model_xgb_path}")
