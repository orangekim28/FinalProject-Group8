#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pyreadstat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import os
#%%
# # Get the absolute path of the XPT file
# file_path = os.path.abspath('/Users/jiwoosuh/Downloads/LLCP2020.XPT ')
#
# # Load the XPT file
# df, meta = pyreadstat.read_xport(file_path)
#
# # Export to CSVff
# df.to_csv('/Users/jiwoosuh/Downloads/LLCP2020.csv', index=False)

#%%
df = pd.read_csv('/Users/jiwoosuh/Documents/FinalProject-Group8/Code/heart_2020_cleaned.csv')
# df = pd.read_csv("/Users/jiwoosuh/Downloads/LLCP2020.csv")
df.info()
print(f"Number of observations: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
col_df = list(df.columns.values)
for col in col_df:
    print(col, ':', str(df[col].unique()))

#%%
jw = ["BMI", "Sex", "AgeCategory", "Race", "GenHealth", "HeartDisease"]
df_jw = df[jw]
categorical_features = list(df_jw.select_dtypes(include=['object']).columns)
numerical_features = list(df_jw.select_dtypes(include=['int64', 'float64']).columns)
categorical_features.remove('HeartDisease')
print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")
#%%
# Create a figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first boxplot on the first subplot
sns.boxplot(x=df[df['HeartDisease']=='Yes']["BMI"], color="#ea4335", ax=ax[0])
ax[0].set_title("Boxplot of BMI for Heart Disease Yes")

# Plot the second boxplot on the second subplot
sns.boxplot(x=df[df['HeartDisease']=='No']["BMI"], color='#4285f4', ax=ax[1])
ax[1].set_title("Boxplot of BMI for Heart Disease No")

# Show the plot
plt.show()

#%%
# Create a figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first histogram on the first subplot
sns.histplot(df[df['HeartDisease']=='Yes'], x="BMI", kde=True, color="#ea4335", ax=ax[0])
ax[0].set_title("Histogram of BMI for Heart Disease Yes")

# Plot the second histogram on the second subplot
sns.histplot(df[df['HeartDisease']=='No'], x="BMI", kde=True, color='#4285f4', ax=ax[1])
ax[1].set_title("Histogram of BMI for Heart Disease No")

# Show the plot
plt.show()

# for col in jw:
#     plt.figure(figsize=(13, 6))
#     sns.countplot(x=df[col], hue=col, data=df_jw, palette='YlOrBr')
#     plt.xlabel(f'{col}')
#     plt.legend()
#     plt.ylabel('Frequency')
#     plt.show()
#%%
# sns.histplot(df, x = 'AgeCategory', hue='HeartDisease', bins=15,  multiple ='stack')
# plt.title('Histogram ')
# plt.show()
#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first histogram on the first subplot
sns.histplot(df[df['HeartDisease']=='Yes'], x="AgeCategory", kde=True, color="#ea4335", ax=ax[0])
ax[0].set_title("Histogram of Age Cateogry for Heart Disease Yes")

# Plot the second histogram on the second subplot
sns.histplot(df[df['HeartDisease']=='No'], x="AgeCategory", kde=True, color='#4285f4', ax=ax[1])
ax[1].set_title("Histogram of Age Cateogry for Heart Disease No")

# Show the plot
plt.show()

#%%
# for Categorical columns:
# - distribution of itself (by histogram)
# - compare by heart disease (by pie, bar chart)
# for Numerical columns:
# - distribution of itself (by historgram, scatterplot, boxplot)
# - compare by heart disease (by stacked histogram)
# conclusion for each EDA part
# after EDA
# - heatmap
# - pre-processing
# - Balancing data when? * ask Amir
# - Sampling method * Use random sampler function. with ratio 10. and then do feature engieering

# jw = ["BMI", "Sex", "AgeCategory", "Race", "GenHealth", "HeartDisease"]
# df_jw = df[jw]
# categorical_features = list(df_jw.select_dtypes(include=['object']).columns)
# numerical_features = list(df_jw.select_dtypes(include=['int64', 'float64']).columns)
# #%%
# # Loop over each numerical feature
# for feature in numerical_features:
#     # Create a figure with two subplots
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#
#     # Plot the first boxplot on the first subplot
#     sns.boxplot(x=df[df['HeartDisease']=='Yes'][feature], color="#ea4335", ax=ax[0])
#     ax[0].set_title(f"Boxplot of {feature} for Heart Disease Yes")
#
#     # Plot the second boxplot on the second subplot
#     sns.boxplot(x=df[df['HeartDisease']=='No'][feature], color='#4285f4', ax=ax[1])
#     ax[1].set_title(f"Boxplot of {feature} for Heart Disease No")
#
#     # Show the plot
#     plt.show()
#
#     # Create a figure with two subplots
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#
#     # Plot the first histogram on the first subplot
#     sns.histplot(df[df['HeartDisease']=='Yes'], x=feature, kde=True, color="#ea4335", ax=ax[0])
#     ax[0].set_title(f"Histogram of {feature} for Heart Disease Yes")
#
#     # Plot the second histogram on the second subplot
#     sns.histplot(df[df['HeartDisease']=='No'], x=feature, kde=True, color='#4285f4', ax=ax[1])
#     ax[1].set_title(f"Histogram of {feature} for Heart Disease No")
#
#     # Show the plot
#     plt.show()
# jw = ["BMI", "Sex", "AgeCategory", "Race", "GenHealth", "HeartDisease"]
# df_jw = df[jw]
# categorical_features = list(df_jw.select_dtypes(include=['object']).columns)
# numerical_features = list(df_jw.select_dtypes(include=['int64', 'float64']).columns)

# Loop over each numerical feature


#%%
# Loop for histogram for categorical features
for feature in categorical_features:
    # Plot the histogram as a whole
    # sns.histplot(df, x=feature)
    # plt.title(f"Histogram of {feature}")
    # # Show the plot
    # plt.tight_layout()
    # plt.show()

    plt.pie(df[feature].value_counts(), labels=df[feature].value_counts().index, autopct='%.0f%%')
    plt.legend(title='Segment')
    plt.title(f"Pie chart of {feature}")
    plt.show()

    sns.countplot(x=df[feature].sort_values(), hue=df.HeartDisease)
    plt.title(f"Histogram of {feature}")
    plt.show()

    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first histogram on the first subplot
    sns.histplot(df[df['HeartDisease']=='Yes'], x=feature, kde=True, color="#ea4335", ax=ax[0])
    ax[0].set_title(f"Histogram of {feature} for Heart Disease Yes")

    # Plot the second histogram on the second subplot
    sns.histplot(df[df['HeartDisease']=='No'], x=feature, kde=True, color='#4285f4', ax=ax[1])
    ax[1].set_title(f"Histogram of {feature} for Heart Disease No")

    # Show the plot
    plt.tight_layout()
    plt.show()

# plt.figure(figsize=(10,4))
# sns.countplot(x = df.AgeCategory.sort_values(),hue=df.HeartDisease)
# plt.title("Age Category for Heart Disease Yes")
# plt.show()

df[df['HeartDisease']=='Yes'].describe(include='object')
hdyes = df[df['HeartDisease']=='Yes']

plt.figure(figsize=(10,4))
sns.countplot(x = hdyes.AgeCategory.sort_values(),hue=hdyes.Sex)
plt.title("Age Category for Heart Disease Yes")
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(x = hdyes.Race.sort_values(),hue=hdyes.Sex)
plt.title("Race for Heart Disease Yes")
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(x = hdyes.GenHealth.sort_values(),hue=hdyes.Sex)
plt.title("Gen Health for Heart Disease Yes")
plt.show()


#%%
# Numerical variables -- MentalHealth
sns.distplot(df['MentalHealth'], kde=False, bins=20, hist=True)
plt.show()

sns.boxplot(x='MentalHealth', data=df)
plt.show()

sns.histplot(df, x = 'MentalHealth', hue='HeartDisease', bins=15,  multiple ='stack', kde=True)
plt.show()

# %%
# Numerical variables -- PhysicalHealth 
sns.distplot(df['PhysicalHealth'], kde=False, bins=20, hist=True)
plt.show()


sns.boxplot(x='PhysicalHealth', data=df)
plt.show()

sns.histplot(df, x = 'PhysicalHealth', hue='HeartDisease', bins=15,  multiple ='stack', kde=True)
plt.show()
# %%
# Numerical variables -- SleepTime 
sns.distplot(df['SleepTime'], kde=False, bins=20, hist=True)
plt.show()

sns.boxplot(x='SleepTime', data=df)
plt.show()

sns.histplot(df, x = 'SleepTime', hue='HeartDisease', bins=15,  multiple ='stack', kde=True)
plt.show()

# %%
#Categorical Variables -- Smoking
sns.countplot(x=df["Smoking"])
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.show()

plt.pie(df.Smoking.value_counts(), labels=df.Smoking.value_counts().index,autopct='%.0f%%')
plt.legend(title='Smoking')
plt.show()
sns.countplot(x = df.HeartDisease ,hue=df.Smoking)
plt.show()

# %%
#Categorical Variables -- Alcohol Drinking
sns.countplot(x=df["AlcoholDrinking"])
plt.xlabel('AlcoholDrinking')
plt.ylabel('Count')
plt.show()

plt.pie(df.AlcoholDrinking.value_counts(), labels=df.AlcoholDrinking.value_counts().index,autopct='%.0f%%')
plt.legend(title='AlcoholDrinking')
plt.show()

sns.countplot(x = df.HeartDisease ,hue=df.AlcoholDrinking)
plt.show()

# %%
#Categorical Variables -- Difficult Walking
sns.countplot(x=df["DiffWalking"])
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.show()

plt.pie(df.DiffWalking.value_counts(), labels=df.DiffWalking.value_counts().index,autopct='%.0f%%')
plt.legend(title='DiffWalking')
plt.show()

sns.countplot(x = df.HeartDisease ,hue=df.DiffWalking)
plt.show()

# %%
#Categorical Variables -- Physical Activity
sns.countplot(x=df["PhysicalActivity"])
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.show()

plt.pie(df.PhysicalActivity.value_counts(), labels=df.PhysicalActivity.value_counts().index,autopct='%.0f%%')
plt.legend(title='PhysicalActivity')
plt.show()

sns.countplot(x = df.HeartDisease ,hue=df.PhysicalActivity)
plt.show()

#%%
###### DATA IMBALANCE

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('/Users/jiwoosuh/Documents/FinalProject-Group8/Code/heart_2020_cleaned.csv')
categorical_features = list(df.select_dtypes(include=['object']).columns)
numerical_features = list(df.select_dtypes(include=['int64', 'float64']).columns)
categorical_features.remove('HeartDisease')
print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")
# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=categorical_features)
# Split into features (X) and target (y)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

#%%
# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    # 'Decision Tree': DecisionTreeClassifier(),
    # 'Random Forest': RandomForestClassifier()
}

#%%
# Set up K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# set up the oversampler
from imblearn.over_sampling import RandomOverSampler
sampler = RandomOverSampler()

# Loop through models and evaluate performance
for name, model in models.items():
    scores = []
    for i, (train_idx, val_idx) in enumerate(kfold.split(X)):
        # Split data into training and validation sets for this fold
        print(f"Fold{i}")
        # print(f"Train: index = {train_idx}")
        # print(f"Val: index = {val_idx}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train,y_train)

        # Fit the model on the training data
        # model.fit(X_train, y_train)
        model.fit(X_train_resampled, y_train_resampled)

        # Predict on the validation data and calculate accuracy
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"accuracy for {i}: {acc}")
        scores.append(acc)

    # Print the average accuracy score for this model
    print(f'{name}: Average accuracy score = {sum(scores) / len(scores):.2f}')

