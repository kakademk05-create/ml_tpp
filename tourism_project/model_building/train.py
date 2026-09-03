import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
import os

def train_model():
    # Load the preprocessed data
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').squeeze() # Use .squeeze() to convert DataFrame to Series
    y_test = pd.read_csv('y_test.csv').squeeze()

    # Identify categorical and numerical features for preprocessing
    categorical_features = X_train.select_dtypes(include=['object']).columns
    numerical_features = X_train.select_dtypes(include=np.number).columns

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Define the model pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])

    # Define hyperparameters for tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }

    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns") # Explicitly set local tracking URI
    mlflow.set_experiment("Tourism Package Prediction")

    with mlflow.start_run():
        print("Starting MLflow run...")
        # Log parameters
        mlflow.log_params({
            'n_estimators_candidates': param_grid['classifier__n_estimators'],
            'learning_rate_candidates': param_grid['classifier__learning_rate'],
            'max_depth_candidates': param_grid['classifier__max_depth']
        })

        # Perform GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Evaluate the best model
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Log metrics to MLflow
        mlflow.log_param("best_n_estimators", grid_search.best_params_['classifier__n_estimators'])
        mlflow.log_param("best_learning_rate", grid_search.best_params_['classifier__learning_rate'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['classifier__max_depth'])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save the best model
        deployment_path = 'tourism_project/deployment'
        os.makedirs(deployment_path, exist_ok=True)
        model_save_path = os.path.join(deployment_path, 'model.joblib')
        joblib.dump(best_model, model_save_path)
        print(f"Best model saved to {model_save_path}")

        # Log the model to MLflow
        mlflow.sklearn.log_model(best_model, "model",
                                  registered_model_name="XGBoostTourismClassifier")
        print("Model logged to MLflow.")

if __name__ == '__main__':
    train_model()
