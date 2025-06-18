import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import pickle

np.random.seed(42)  # For reproducibility

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # train
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # test model
    y_pred = model.predict(X_test)
    print("Model's accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    
    return model, scaler

def get_clean_data():
    # Get the absolute path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "data.csv")
    
    data = pd.read_csv(data_path)    
    data = data.drop(['Unnamed: 32', 'id'], axis=1)    
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    return data

def main():
    data = get_clean_data()
    
    model, scaler = create_model(data)
    
    # Create models directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the model and scaler with absolute paths
    model_path = os.path.join(models_dir, "logistic_regression_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    # Save using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()