import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    data_path = 'tourism_project/data/tourism.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please ensure tourism.csv is in the data folder.")

    df = pd.read_csv(data_path)

    # Drop unnecessary columns identified during data exploration
    # 'Unnamed: 0' appears to be an old index column, 'CustomerID' is an identifier
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    # Handle missing values
    # Numerical columns: fill with mean
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # Categorical columns: fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Define features (X) and target (y)
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create a directory for preprocessed data if it doesn't exist
    os.makedirs('tourism_project/model_building/preprocessed_data', exist_ok=True)

    # Save the splits locally as CSV files
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    print("Data preparation complete. Training and testing sets saved.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_data()
