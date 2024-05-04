import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
import seaborn as sns
import warnings


from sklearn.model_selection import learning_curve



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


from sklearn.metrics import confusion_matrix

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report



warnings.filterwarnings('ignore')


df1=pd.read_csv("trainset.csv")
print(df1)

df1.isnull().sum()
df2=df1.replace("unknown", pd.NA)
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


def encode_job(x):
  arr=['blue-collar', 'entrepreneur', 'technician', 'housemaid',
       'services', 'management', 'admin', 'student', 'unemployed',
       'self-employed', 'retired']
  x = x.strip().lower()
  if x in arr:
    return arr.index(x)
  else:
    return -1

def encode_marital(x):
  arr=['divorced', 'married', 'single']
  x = x.strip().lower()
  if x in arr:
    return arr.index(x)
  else:
    return -1

def encode_education(x):
  arr=['basic.4y', 'university.degree', 'basic.9y', 'professional.course',
       'high.school', 'basic.6y', 'illiterate']
  x = x.strip().lower()
  if x in arr:
    return arr.index(x)
  else:
    return -1

def encode_binary(x):
  arr=['yes', 'no']
  x = x.strip().lower()
  if x in arr:
    return arr.index(x)
  else:
    return -1

def encode_contact(x):
  arr=['telephone', 'cellular']
  x = x.strip().lower()
  if x in arr:
    return arr.index(x)
  else:
    return -1

def encode_month(x):
  arr=['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr',
       'sep','jan','feb']
  x = x.strip().lower()
  if x in arr:
    return arr.index(x)
  else:
    return -1

def encode_day_of_week(x):
  arr=['mon', 'tue', 'wed', 'thu', 'fri','sat','sun']
  x = x.strip().lower()
  if x in arr:
    return arr.index(x)
  else:
    return -1

def encode_poutcome(x):
  arr=['nonexistent', 'failure', 'success']
  x = x.strip().lower()
  if x in arr:
    return arr.index(x)
  else:
    return -1


df3=df2.copy()


columns_to_encode = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'Subscribed']

# Apply encoding functions using lambda function
for col in columns_to_encode:
    if col not in ('housing', 'loan', 'Subscribed') and col in df3:
        df3[col] = df3[col].apply(lambda x: globals()[f'encode_{col}'](x))
    elif col == 'housing' or col == 'loan':
        df3[col] = df3[col].apply(lambda x: globals()['encode_binary'](x))
    elif col == 'Subscribed':
        df3[col] = df3[col].apply(lambda x: globals()['encode_binary'](x))



# Histograms for numerical features
numerical_features = ['age', 'duration', 'campaign', 'pdays', 'nr.employed']
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df3, x=feature, bins=20, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    #plt.show()



# # Bar plots for categorical features
categorical_features = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df2, x=feature, hue='Subscribed')
    plt.title(f'Distribution of {feature} by Subscribed')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Subscribed', loc='upper right')
    #plt.show()



# # pie chart to observe distribution of categorical features
categorical_features = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']


for feature in categorical_features:
    plt.figure(figsize=(10, 10))
    feature_counts = df2[feature].value_counts()
    sizes = feature_counts.values
    labels = feature_counts.index
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(title='Distribution', loc='upper right', bbox_to_anchor=(1.5, 1))
    plt.title(f'Distribution of {feature}')
    #plt.show()

# # pie chart of feature divided by 'Subscribed' and 'Not Subscribed'
categorical_features = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# Create pie charts for each feature divided by 'Subscribed' and 'Not Subscribed'
for feature in categorical_features:
    plt.figure(figsize=(12, 6))
    feature_counts = df2.groupby([feature, 'Subscribed']).size().unstack()

    # Plot 'Subscribed' Pie Chart
    plt.subplot(1, 2, 1)
    feature_counts['yes'].plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.title(f'Subscribed: {feature}', fontweight='bold')

    # Plot 'Not Subscribed' Pie Chart
    plt.subplot(1, 2, 2)
    feature_counts['no'].plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.title(f'Not Subscribed: {feature}', fontweight='bold')

    plt.suptitle(f'Distribution of {feature} by Subscribed and Not Subscribed', y=1.02)
    plt.tight_layout()
    #plt.show()


# Box plots for numerical features
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df3, y=feature, x='Subscribed')
    plt.title(f'Boxplot of {feature} by Subscribed')
    plt.xlabel('Subscribed')
    plt.ylabel(feature)
    #plt.show()


# Correlation matrix and heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
#plt.show()


# Prepare the data
X = df3[['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'poutcome', 'nr.employed']]
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

# Plot feature importances or coefficients
for name, importance in feature_importances.items():
    plt.figure(figsize=(10, 6))
    if 'Logistic Regression' in name:
        sns.barplot(x=importance, y=X.columns)
        plt.title(f'Feature Importance for {name}')



df_test=pd.read_csv("testset.csv")


columns_to_encode = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'Subscribed']

# Apply encoding functions using lambda function
for col in columns_to_encode:
    if col not in ('housing', 'loan', 'Subscribed') and col in df3:
        df_test[col] = df_test[col].apply(lambda x: globals()[f'encode_{col}'](x))
    elif col == 'housing' or col == 'loan':
        df_test[col] = df_test[col].apply(lambda x: globals()['encode_binary'](x))
    elif col == 'Subscribed':
        df_test[col] = df_test[col].apply(lambda x: globals()['encode_binary'](x))


X_test = df_test[['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'poutcome', 'nr.employed']]
y_test = df_test['Subscribed']

X_train = df3[['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'poutcome', 'nr.employed']]
y_train = df3['Subscribed']




# Instantiate logistic regression model with maximum iteration 1000
log_reg_model = LogisticRegression(max_iter=1000)

# Fit the model to the training data
log_reg_model.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = log_reg_model.predict(X_test)
print(X_test)
print("####################")
print(y_pred)



train_sizes, train_scores, test_scores = learning_curve(log_reg_model, X_train, y_train, cv=5, scoring='accuracy')

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training accuracy', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, label='Validation accuracy', color='green')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')

plt.title('Learning Curve')
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid()
#plt.show()


cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['False', 'True'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
#plt.show()


# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Assuming you have already trained a logistic regression model named 'log_reg_model'
y_probs = log_reg_model.predict_proba(X_test)[:,1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc )

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
#plt.show()

# Print accuracy and classification report
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("---------------------------------------------------------")

# Visualize accuracy using box plot
plt.figure(figsize=(6, 4))
sns.boxplot(data=[accuracy], width=0.4)
plt.title('Accuracy')
plt.ylabel('Value')
#plt.show()

# Convert classification report to dictionary for tabular display
class_report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert dictionary to DataFrame for tabular display
import pandas as pd
class_report_df = pd.DataFrame(class_report_dict).transpose()

# Visualize classification report using heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(class_report_df.iloc[:-1, :].T, annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report')
#plt.show()




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

# Choose a single data point from the test set (the first one)
single_data_point = {
    'age': 62,
    'job': 'retired',
    'marital': 'married',
    'education': 'university.degree',
    'housing': 'no',
    'loan': 'no',
    'contact': 'cellular',
    'month': 'oct',
    'day_of_week': 'fri',
    'duration': 717,
    'campaign': 2,
    'pdays': 999,
    'poutcome': 'nonexistent',
    'Subscribed': 'yes',  # Adding the 'Subscribed' column
    'nr.employed': 5017.5,
}

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
X_single = single_df[['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'poutcome', 'nr.employed']]

# Perform prediction
# prediction_single = log_reg_model.predict(X_single)

# Display prediction
print("Prediction for the single data point:")
# print(decode_binary(prediction_single[0]))


from joblib import dump
#
# # Define the file path where you want to save the model
# model_file_path = 'logistic_regression_model.joblib'
#
# # Save the trained model
# dump(log_reg_model, model_file_path)
#
# print("Trained model saved successfully at:", model_file_path)


def train_model():
    # Instantiate logistic regression model with maximum iteration 1000
    log_reg_model = LogisticRegression(max_iter=1000)

    # Fit the model to the training data
    log_reg_model.fit(X_train, y_train)

    model_file_path = 'logistic_regression_model.joblib'

    # Save the trained model
    dump(log_reg_model, model_file_path)

    print("Model trained and saved successfully at:", model_file_path)




