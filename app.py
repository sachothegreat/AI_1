from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import joblib
import tensorflow as tf
import xgboost as xgb
import os
import re

app = Flask(__name__)

# Define the directory where models are stored
MODEL_DIR = 'uploads'

# Load the trained models and scaler
model_tf = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'parkinsons_model.h5'))
model_lr = joblib.load(os.path.join(MODEL_DIR, 'parkinsons_logreg.pkl'))
model_xgb = xgb.XGBClassifier()
model_xgb.load_model(os.path.join(MODEL_DIR, 'parkinsons_xgb_model.json'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

# Load the feature names used during training
with open(os.path.join(MODEL_DIR, 'feature_names.txt'), 'r') as f:
    feature_names = f.read().split(',')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename, file_extension = os.path.splitext(file.filename)
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            if file_extension == '.csv' or file_extension == '.data':
                data = pd.read_csv(filepath)

                if 'name' not in data.columns:
                    return "Error: 'name' column is missing from the uploaded file.", 400

                # Extract person identifier (SXX) from the 'name' column
                def extract_person_id(name):
                    person_id = re.search(r'S\d+', name)
                    return person_id.group() if person_id else "Unknown"

                data['Person_ID'] = data['name'].apply(extract_person_id)

                # Drop non-feature columns
                data_for_prediction = data.drop(columns=['name', 'Person_ID', 'status'])

                # Ensure the feature columns are in the correct order
                data_for_prediction = data_for_prediction[feature_names]

                # Scale the features
                X_new = scaler.transform(data_for_prediction)

                # Get predictions from all models
                predictions_tf = model_tf.predict(X_new)
                predictions_lr = model_lr.predict(X_new)
                predictions_xgb = model_xgb.predict(X_new)

                data['Prediction_TensorFlow'] = (predictions_tf > 0.5).astype(int)
                data['Prediction_LogReg'] = predictions_lr
                data['Prediction_XGBoost'] = predictions_xgb

                # Aggregate predictions by person (Person_ID)
                aggregated = data.groupby('Person_ID').agg({
                    'Prediction_TensorFlow': 'mean',
                    'Prediction_LogReg': 'mean',
                    'Prediction_XGBoost': 'mean'
                })

                # Determine the final prediction for each person (majority vote)
                aggregated['Final_Prediction'] = (aggregated.mean(axis=1) > 0.5).astype(int)

                # Identify which persons have the disease
                individuals_with_disease = aggregated[aggregated['Final_Prediction'] == 1].index.tolist()
                individuals_without_disease = aggregated[aggregated['Final_Prediction'] == 0].index.tolist()

                # Count how many individuals have the disease and how many don't
                count_with_disease = len(individuals_with_disease)
                count_without_disease = len(individuals_without_disease)

                # Print out the results in the console
                print(f"Individuals predicted to have Parkinson's disease: {individuals_with_disease} (Total: {count_with_disease})")
                print(f"Individuals predicted NOT to have Parkinson's disease: {individuals_without_disease} (Total: {count_without_disease})")

                # Add the counts to the aggregated DataFrame
                summary_row = pd.DataFrame({
                    'Prediction_TensorFlow': [""],
                    'Prediction_LogReg': [""],
                    'Prediction_XGBoost': [""],
                    'Final_Prediction': [
                        f"Total with disease: {count_with_disease}",
                        f"Total without disease: {count_without_disease}"
                    ]
                }, index=['Summary1', 'Summary2'])

                # Append the summary row to the DataFrame
                aggregated = pd.concat([aggregated, summary_row])

                # Save the aggregated results to an Excel file
                output_filepath = os.path.join('uploads', 'aggregated_predictions_by_person.xlsx')
                aggregated.to_excel(output_filepath)

                return send_file(output_filepath, as_attachment=True)

    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='127.0.0.1', port=5000, debug=True)
