import pandas as pd
import numpy as np

def prepare_rainfall_data():
    try:
        # Read the CSV file
        df = pd.read_csv('Rainfall.csv')
        print("Original columns:", df.columns.tolist())
        
        # Rename columns if needed
        # Adjust this mapping according to your actual column names
        column_mapping = {
            'Temperature': 'temperature',
            'Humidity': 'humidity',
            'Pressure': 'pressure',
            'Wind Speed': 'wind_speed',
            'Rainfall': 'rainfall'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert rainfall to binary (0 or 1)
        if 'rainfall' in df.columns:
            df['rainfall'] = df['rainfall'].apply(lambda x: 1 if x > 0 else 0)
        
        # Save the processed data
        df.to_csv('Rainfall_processed.csv', index=False)
        print("Data processed successfully!")
        print("Final columns:", df.columns.tolist())
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    prepare_rainfall_data() 