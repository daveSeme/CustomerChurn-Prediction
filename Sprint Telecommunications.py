#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # Load the Dataset

# In[2]:


# Load the dataset
data = pd.read_csv('data/TelcoCustomerChurn.csv')

# Display the first few rows of the dataset to understand its structure
data.head()


# In[4]:


#To display basic information of the data set
data.info()


# # Data Preprocessing

# In[5]:


#I can see that the 'TotalCharges' column is of data type object (string) instead of numerical,
#so I am going to convert it to a numerical data type

# Convert 'TotalCharges' to numerical (float) type
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')


# In[6]:


#Also I am going to convert  the 'churn' to binary numerical values (0 for 'No' and 1 for 'Yes')

# Convert 'Churn' to binary numerical values (0 for 'No' and 1 for 'Yes')
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[7]:


data.info()


# # Mean Imputation

# In[8]:


# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing values if present (replace with mean for 'TotalCharges')
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)


# In[9]:


# Confirm that missing values have been handled
missing_values_after_handling = data.isnull().sum()
print("\nMissing Values after Handling:\n", missing_values_after_handling)


# # Feature Encoding

# In[10]:


# I need to encode categorical features into numerical values so that the machine learning model can use them.
#I will use one-hot encoding for this purpose.

# One-hot encode categorical columns
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Display the updated dataframe after encoding
data_encoded.head()


# # Feature Scaling

# In[11]:


#Scaling the numerical features so they have a similar influence on the machine learning model

from sklearn.preprocessing import StandardScaler

# Scale the numerical features
scaler = StandardScaler()
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])

# Display the updated dataframe after scaling
data_encoded.head()


# # Data Splitting

# In[12]:


# Split the dataset into features (X) and target variable (y)
X = data_encoded.drop('Churn', axis=1)
y = data_encoded['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the resulting sets
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[18]:


# Display the columns in the DataFrame
print("Columns in the DataFrame:")
print(data_encoded.columns)


# In[21]:


# Check the data types of columns in X_train
print(X_train.dtypes)


# In[22]:


# Drop the 'customerID' column since its not of a major impact in training
X_train = X_train.drop('customerID', axis=1)


# # Model Training and Evaluation

# In[23]:


# Training a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)


# In[24]:


# here are the predictions even when dropping the customerID to avoid errors
# Predictions
X_test = X_test.drop('customerID', axis=1)  # Drop 'customerID' from the test set as well
y_pred = rf_classifier.predict(X_test)


# In[25]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)


# In[26]:


# Print the evaluation metrics
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report)


# The model is approximately 79.28% accurate in predicting customer churn based on the test data.
# 
# The Classification Report shows that;
# 
# In Precision: The proportion of true positive predictions among all positive predictions. For predicting churn (1), it's about 65%, and for not predicting churn (0), it's about 83%.
# 
# In Recall: The proportion of true positive predictions among the actual positives. For predicting churn (1), it's about 47%, and for not predicting churn (0), it's about 91%.
# 
# F1-score: The weighted average of precision and recall. For predicting churn (1), it's about 54%, and for not predicting churn (0), it's about 87%.

# # Feature importances 
# I need to analyze the feature importances from the trained Random Forest model to understand which factors are most influential in predicting churn

# In[28]:


# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display feature importances
print("Feature Importances:\n", feature_importance_df)


# # Here are the impacts;
# 1.The total charges, tenure, and monthly charges would heavily impact customer churn predictions.
# 2.The type of internet service also plays a significant role, particularly fiber optic service.
# 3.The payment method, especially electronic checks, seems to have notable importance.
# 4.Gender, partner, tech support, online security, and other services also play a role, though to a lesser extent.
# 
# # Resolutions.
#  
# # Sprint can focus on improving the areas of high feature importance to reduce churn
# 1.Improving the quality or pricing of Fiber optic internet services or optimizing payment processes for electronic checks.
# 
# 2.Tailoring marketing strategies: Knowing that tenure and contract length matter, specific promotions or benefits could be offered to long-term customers.
# 
# 3.Addressing concerns: Understanding the impact of tech support and online security, Sprint can enhance these services to increase customer satisfaction and reduce churn.

# In[ ]:




