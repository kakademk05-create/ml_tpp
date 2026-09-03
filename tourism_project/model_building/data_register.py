import pandas as pd
import os

def register_data():
    data_path = 'tourism_project/data/tourism.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please upload tourism.csv.")

    df = pd.read_csv(data_path)

    # Define expected columns based on the Data Description
    expected_columns = [
        'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'Occupation',
        'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus',
        'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation',
        'MonthlyIncome', 'PitchSatisfactionScore', 'ProductPitched', 'NumberOfFollowups',
        'DurationOfPitch'
    ]

    # Check if all expected columns are present
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns in the dataset: {', '.join(missing_columns)}")

    print("Dataset loaded successfully.")
    print("Data shape:", df.shape)
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nData Info:")
    df.info()
    return df

if __name__ == '__main__':
    df = register_data()
