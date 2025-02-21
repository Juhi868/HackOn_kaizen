# 1. Library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import joblib
from flask import Flask, request, render_template, jsonify
import io
import base64
import os # Import the os module

# Flask app initialization
app = Flask(__name__) # Changed _name_ to __name__

# 2. Setting up path for training data
data_path1 = 'train_genetic_disorders.csv'
data_path2 = 'train.csv'
data_path3 = 'sample_submission.csv'

# 3. Data loading for training
try:
    # Check if the files exist before attempting to read them
    if not all(os.path.exists(path) for path in [data_path1, data_path2, data_path3]):
        raise FileNotFoundError("One or more data files are missing.") # Raise an exception if any file is missing

    data1 = pd.read_csv(data_path1)
    data2 = pd.read_csv(data_path2)
    data3 = pd.read_csv(data_path3)
    
    # Now, explore the data and merge the datasets inside the try block:
    # 4. Data exploration
    def explore_data(data):
        print(data.info())
        print(data.describe())
        print(data.isnull().sum())
    
    for i, data in enumerate([data1, data2, data3], 1):
        print(f"Dataset {i}:")
        explore_data(data)
    
    merged_data = pd.concat([data1, data2, data3], ignore_index=True)
    # ... (Rest of your code using merged_data) ...
    
    print(merged_data.columns)
    merged_data.rename(columns=lambda x: x.strip(), inplace=True)  # Remove leading/trailing spaces
    # Assuming 'genetic_disorder' is the actual column name in your datasets
    # If not, replace 'genetic_disorder' with the actual name
    merged_data.rename(columns={'genetic_disorder': 'genetic_disorder'}, inplace=True)
    

    if 'genetic_disorder' in merged_data.columns:
        print("Column 'genetic_disorder' exists.")
        print(merged_data['genetic_disorder'].head())  # Print first few values
    else:
        print("Column 'genetic_disorder' does NOT exist.")

    if 'genetic_disorder' not in data1.columns:
        data1['genetic_disorder'] = np.nan  # Add a default value (e.g., NaN)
    if 'genetic_disorder' not in data2.columns:
        data2['genetic_disorder'] = np.nan
    if 'genetic_disorder' not in data3.columns:
        data3['genetic_disorder'] = np.nan

    print("Unique values in 'genetic_disorder':", merged_data['genetic_disorder'].unique())

    # 6. Data visualization
    def visualize_data(data):
        if 'genetic_disorder' not in data.columns:
            print("Error: Column 'genetic_disorder' not found in the dataset.")
            return

        plt.figure(figsize=(10, 6))
        sns.countplot(x='genetic_disorder', data=data)
        plt.title('Distribution of Genetic Disorders')
        plt.xlabel('Genetic Disorder')
        plt.ylabel('Count')
        plt.savefig('static/genetic_disorders_distribution.png')
        plt.close()

    # Call the function after confirming 'merged_data' is correct
    visualize_data(merged_data)

except FileNotFoundError as e:
    # ... (Existing code for handling FileNotFoundError) ...
    print(f"Error: {e}. Please ensure all data files are in the correct location.")
    exit(1)  # Exit the script if files are not found

# Call visualize_data only if merged_data is defined
if 'merged_data' in locals():
    visualize_data(merged_data)
else:
    print("Error: merged_data is not defined. Visualization skipped.")


print(merged_data.columns)
merged_data.rename(columns=lambda x: x.strip(), inplace=True)  # Remove leading/trailing spaces
merged_data.rename(columns={'actual_column_name': 'genetic_disorder'}, inplace=True)
if 'genetic_disorder' in merged_data.columns:
    print("Column 'genetic_disorder' exists.")
    print(merged_data['genetic_disorder'].head())  # Print first few values
else:
    print("Column 'genetic_disorder' does NOT exist.")

if 'genetic_disorder' not in data1.columns:
    data1['genetic_disorder'] = np.nan  # Add a default value (e.g., NaN)
if 'genetic_disorder' not in data2.columns:
    data2['genetic_disorder'] = np.nan
if 'genetic_disorder' not in data3.columns:
    data3['genetic_disorder'] = np.nan

print("Unique values in 'genetic_disorder':", merged_data['genetic_disorder'].unique())

import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(data):
    if 'genetic_disorder' not in data.columns:
        print("Error: Column 'genetic_disorder' not found in the dataset.")
        return

    plt.figure(figsize=(10, 6))
    sns.countplot(x='genetic_disorder', data=data)
    plt.title('Distribution of Genetic Disorders')
    plt.xlabel('Genetic Disorder')
    plt.ylabel('Count')
    plt.savefig('static/genetic_disorders_distribution.png')
    plt.close()

# Call the function after confirming 'merged_data' is correct
visualize_data(merged_data)
