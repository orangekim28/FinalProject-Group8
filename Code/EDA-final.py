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
# import pyreadstat
import os
from dython.nominal import associations
from dython.nominal import identify_nominal_columns
#%%
# Data Infomation
df = pd.read_csv('heart_2020_cleaned.csv')
print(df.head())
print(df.columns)
print(df.shape)
categorical_features=identify_nominal_columns(df)
numerical_features = ['BMI','PhysicalHealth','MentalHealth','SleepTime']
print(df.columns)
print(df.info())
print(df.describe())
print(df.isna().sum())

col_df = list(df.columns.values)
for col in col_df:
    print(col, ':', str(df[col].unique()))
    
# count the target variable with yes and no
print(df['HeartDisease'].value_counts())

#%%#################################EDA analysis#########################
# Self-charateristic features - jiwoo
jw = ["BMI", "Sex", "AgeCategory", "Race", "GenHealth", "HeartDisease"]
df_jw = df[jw]
categorical_features = list(df_jw.select_dtypes(include=['object']).columns)
numerical_features = list(df_jw.select_dtypes(include=['int64', 'float64']).columns)
categorical_features.remove('HeartDisease')
print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")
#%% numerical features
# Create a figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first boxplot on the first subplot
sns.boxplot(x=df[df['HeartDisease']=='Yes']["BMI"], color="#ea4335", ax=ax[0])
ax[0].set_title("Boxplot of BMI for Heart Disease Yes")

# Plot the second boxplot on the second subplot
sns.boxplot(x=df[df['HeartDisease']=='No']["BMI"], color='#4285f4', ax=ax[1])
ax[1].set_title("Boxplot of BMI for Heart Disease No")
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
plt.show()

#%%
# Loop for histogram for categorical features
for feature in categorical_features:

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

#%%
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
# %% EDA analysis - categorical features - Diesease- guoshan
## univariate analysis
Disease_feature= ['Stroke','Diabetic','Asthma','KidneyDisease','SkinCancer']
def univaiate(feature):
    freq_table = pd.crosstab(index=df[feature],columns='count')
    print(freq_table)
    plt.figure(figsize=(15,10))
    plt.subplot(1,2,1)
    sns.histplot(x=feature,data=df).set(title = feature+' vs Heart Disease')
    plt.subplot(1,2,2)
    plt.pie(df[feature].value_counts(),labels=df[feature].unique().tolist(),autopct='%1.1f%%')
    plt.legend(loc = (1.3,1))
    plt.title(feature)
    plt.tight_layout()
    plt.show()
    
for feature in Disease_feature:
    univaiate(feature)

# %%
## multivariate analysis
def multivariate(feature):
    plt.figure(figsize=(10,10))
    sns.countplot(x='HeartDisease',hue = feature,data=df).set(title = feature+' vs Heart Disease')
    plt.show()
    
    plt.figure(figsize=(15,10))
    plt.subplot(1,2,1)
    plt.pie(df[df['HeartDisease']=='Yes'][[feature]].value_counts(),labels=df[df['HeartDisease']=='Yes'][feature].unique().tolist(),autopct='%1.1f%%')
    plt.legend(loc = (1.3,1))
    plt.title(feature+' vs Heart Disease-yes')
    
    plt.subplot(1,2,2)
    plt.pie(df[df['HeartDisease']=='No'][feature].value_counts(),labels=df[df['HeartDisease']=='No'][feature].unique().tolist(),autopct='%1.1f%%')
    plt.legend(loc = (1.3,1))
    plt.title(feature+' vs Heart Disease-no')

    plt.tight_layout()
    plt.show()
 
for feature in Disease_feature:
    multivariate(feature)
    
#%%
# MentalHealth variable
sns.distplot(df['MentalHealth'], kde=False, bins=20, hist=True)
plt.show()

sns.boxplot(x='MentalHealth', data=df)
plt.show()

#plt.scatter()

sns.histplot(df, x = 'MentalHealth', hue='HeartDisease', bins=15,  multiple ='stack', kde=True)
plt.show()

# %%
# PhysicalHealth Variable
sns.distplot(df['PhysicalHealth'], kde=False, bins=20, hist=True)
plt.show()


sns.boxplot(x='PhysicalHealth', data=df)
plt.show()

sns.histplot(df, x = 'PhysicalHealth', hue='HeartDisease', bins=15,  multiple ='stack', kde=True)
plt.show()

# %%
# SleepTime variable
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

# %%
#correlation matrix
complete_correlation= associations(df,figsize=(15,16))
# sort the correlation for each feature
corr = complete_correlation['corr']
for i in df.columns:
    print(corr[i].sort_values(ascending=False))
    print(' ')
'''
HeartDisease
HeartDisease        1.000000
GenHealth           0.259519
AgeCategory         0.245588
DiffWalking         0.201234
Stroke              0.196798
Diabetic            0.185101
PhysicalHealth      0.170721
KidneyDisease       0.145157
Smoking             0.107738
PhysicalActivity    0.100001
SkinCancer          0.093281
Sex                 0.070007
BMI                 0.051803
Race                0.051230
Asthma              0.041390
AlcoholDrinking     0.032009
MentalHealth        0.028591
SleepTime           0.008327
Name: HeartDisease, dtype: float64

GenHealth
PhysicalHealth      0.588780

DiffWalking
GenHealth           0.457933

'''


