import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

X_train = pd.DataFrame()

y_train = pd.DataFrame()


def encode_job(x):
    arr = ['blue-collar', 'entrepreneur', 'technician', 'housemaid',
           'services', 'management', 'admin', 'student', 'unemployed',
           'self-employed', 'retired']
    x = x.strip().lower()
    if x in arr:
        return arr.index(x)
    else:
        return -1


def encode_marital(x):
    arr = ['divorced', 'married', 'single']
    x = x.strip().lower()
    if x in arr:
        return arr.index(x)
    else:
        return -1


def encode_education(x):
    arr = ['basic.4y', 'university.degree', 'basic.9y', 'professional.course',
           'high.school', 'basic.6y', 'illiterate']
    x = x.strip().lower()
    if x in arr:
        return arr.index(x)
    else:
        return -1


def encode_binary(x):
    arr = ['yes', 'no']
    x = x.strip().lower()
    if x in arr:
        return arr.index(x)
    else:
        return -1


def encode_contact(x):
    arr = ['telephone', 'cellular']
    x = x.strip().lower()
    if x in arr:
        return arr.index(x)
    else:
        return -1


def encode_month(x):
    arr = ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr',
           'sep', 'jan', 'feb']
    x = x.strip().lower()
    if x in arr:
        return arr.index(x)
    else:
        return -1


def encode_day_of_week(x):
    arr = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    x = x.strip().lower()
    if x in arr:
        return arr.index(x)
    else:
        return -1


def encode_poutcome(x):
    arr = ['nonexistent', 'failure', 'success']
    x = x.strip().lower()
    if x in arr:
        return arr.index(x)
    else:
        return -1


def prerequisites():
    df1 = pd.read_csv("trainset.csv")
    print(df1)

    df1.isnull().sum()
    df2 = df1.replace("unknown", pd.NA)
    df2.dropna(inplace=True)
    df2.reset_index(drop=True, inplace=True)
    print(df2.columns)
    df2['marital'].unique()
    df2["education"].unique()
    # binary encoding
    df2["housing"].unique()
    # binary encoding
    df2["loan"].unique()
    df2["contact"].unique()
    df2["month"].unique()
    df2["day_of_week"].unique()
    df2["poutcome"].unique()
    # binary encoding
    df2["Subscribed"].unique()
    df3 = df2.copy()

    columns_to_encode = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                         'poutcome', 'Subscribed']

    # Apply encoding functions using lambda function
    for col in columns_to_encode:
        if col not in ('housing', 'loan', 'Subscribed') and col in df3:
            df3[col] = df3[col].apply(lambda x: globals()[f'encode_{col}'](x))
        elif col == 'housing' or col == 'loan':
            df3[col] = df3[col].apply(lambda x: globals()['encode_binary'](x))
        elif col == 'Subscribed':
            df3[col] = df3[col].apply(lambda x: globals()['encode_binary'](x))

    # Prepare the data
    X = df3[['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration',
             'campaign', 'pdays', 'poutcome', 'nr.employed']]
    y = df3['Subscribed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    # Create pipelines for each classifier
    pipelines = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }

    # Train and evaluate each classifier
    for name, pipeline in pipelines.items():
        print(f"Training and evaluating {name}...")
        # Train the model using the pipeline
        pipeline.fit(X_train, y_train)
        # Make predictions on the testing set
        y_pred = pipeline.predict(X_test)
        # Evaluate the classifier's performance
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("---------------------------------------------------------")

    # Train each model
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

    # Extract feature importances or coefficients
    feature_importances = {}
    for name, pipeline in pipelines.items():
        if 'Logistic Regression' in name:
            feature_importances[name] = pipeline.named_steps['logisticregression'].coef_[0]

    df_test = pd.read_csv("testset.csv")

    columns_to_encode = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                         'poutcome', 'Subscribed']

    # Apply encoding functions using lambda function
    for col in columns_to_encode:
        if col not in ('housing', 'loan', 'Subscribed') and col in df3:
            df_test[col] = df_test[col].apply(lambda x: globals()[f'encode_{col}'](x))
        elif col == 'housing' or col == 'loan':
            df_test[col] = df_test[col].apply(lambda x: globals()['encode_binary'](x))
        elif col == 'Subscribed':
            df_test[col] = df_test[col].apply(lambda x: globals()['encode_binary'](x))

    X_test = df_test[
        ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration',
         'campaign', 'pdays', 'poutcome', 'nr.employed']]
    y_test = df_test['Subscribed']

    X_train = df3[
        ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration',
         'campaign', 'pdays', 'poutcome', 'nr.employed']]
    y_train = df3['Subscribed']

    return X_train, y_train


# Define the encode_binary function
def encode_binary(x):
    arr = ['yes', 'no']
    x = x.strip().lower()
    if x in arr:
        return arr.index(x)
    else:
        return -1


def decode_binary(x):
    arr = ['yes', 'no']
    return arr[x]


from joblib import dump

model_file_path = 'logistic_regression_model.joblib'


def train_model():
    X_train, y_train = prerequisites()
    # Instantiate logistic regression model with maximum iteration 1000
    log_reg_model = LogisticRegression(max_iter=1000)

    # Fit the model to the training data
    log_reg_model.fit(X_train, y_train)

    # Save the trained model
    dump(log_reg_model, model_file_path)

    print("Model trained and saved successfully at:", model_file_path)


import os


def delete_model():
    # Check if the model file exists
    if os.path.exists(model_file_path):
        # Delete the model file
        os.remove(model_file_path)
        print("Model file deleted successfully.")
    else:
        print("Model file does not exist.")


# delete_model()

# train_model()
import json
columns_to_encode = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'Subscribed']
from joblib import load


def prediction(single_data_point1):
    loaded_model = None
    try:
        loaded_model = load('logistic_regression_model.joblib')
    except FileNotFoundError:
        print("Model file 'logistic_regression_model.joblib' not found. Training a new model...")
        train_model()
        loaded_model = load('logistic_regression_model.joblib')
    if loaded_model:
        single_data_point = json.loads(json.dumps(single_data_point1))

        # Encode categorical variables
        for col in columns_to_encode:
            if col not in ('housing', 'loan', 'Subscribed'):
                single_data_point[col] = globals()[f'encode_{col}'](single_data_point[col])
            elif col == 'housing' or col == 'loan':
                single_data_point[col] = encode_binary(single_data_point[col])  # Use the defined function directly
            elif col == 'Subscribed':
                single_data_point[col] = encode_binary(single_data_point[col])  # Use the defined function directly

        # Convert to DataFrame with a single row
        single_df = pd.DataFrame([single_data_point])

        # Select features
        X_single = single_df[
            ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration',
             'campaign', 'pdays', 'poutcome', 'nr.employed']]

        prediction_single = loaded_model.predict(X_single)

        # Display prediction
        print("Prediction for the single data point using the loaded model:")

        decision = decode_binary(prediction_single[0])
        print(decision)

        if decision == "yes":
            return 1
        else:
            return 0
