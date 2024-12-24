from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def train_model():
    """Train and save the model if it doesn't exist"""
    try:
        # Read the dataset
        df = pd.read_csv('Rainfall.csv')
        
        # Print the column names to debug
        print("Available columns:", df.columns.tolist())
        
        # Clean the data
        df = df.dropna()  # Remove any rows with missing values
        
        # Prepare features and target
        # Adjust these column names to match your Rainfall.csv
        feature_columns = ['temperature', 'humidity', 'pressure', 'wind_speed']
        X = df[feature_columns]
        y = df['rainfall']  # Make sure this column exists in your CSV
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save both the model and scaler
        joblib.dump(model, 'models/rainfall_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        # Print model accuracy
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Training accuracy: {train_score:.2f}")
        print(f"Testing accuracy: {test_score:.2f}")
        
        return model, scaler
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None, None

# Try to load the model and scaler, train if they don't exist
try:
    model = joblib.load('models/rainfall_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("Model and scaler loaded successfully!")
except:
    print("Model not found. Training new model...")
    model, scaler = train_model()
    if model is None:
        print("Failed to train model!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not available'})
            
        # Get data from the form
        data = {
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'pressure': float(request.form['pressure']),
            'wind_speed': float(request.form['wind_speed'])
        }
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        return jsonify({
            'success': True,
            'prediction': 'Rain' if prediction == 1 else 'No Rain',
            'probability': float(probability)
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Add debug print
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 