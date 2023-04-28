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
categorical_features = identify_nominal_columns(df)
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
# Categorical Features

def univaiate(feature):
    freq_table = pd.crosstab(index=df[feature],columns='count')
    print(freq_table)
    plt.figure(figsize=(15,10))
    plt.subplot(1,2,1)
    sns.histplot(x=feature,data=df).set(title = 'Histogram of '+feature)
    plt.subplot(1,2,2)
    plt.pie(df[feature].value_counts(),labels=df[feature].unique().tolist(),autopct='%1.1f%%')
    plt.legend(loc = (1.3,1))
    plt.title(feature)
    plt.tight_layout()
    plt.show()
        
for feature in categorical_features:
    univaiate(feature)

#%%
## multivariate analysis
if 'HeartDisease' in categorical_features:
    categorical_features.remove('HeartDisease')

Disease_feature= ['Stroke','Diabetic','Asthma','KidneyDisease','SkinCancer']

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
    
for feature in categorical_features:
    multivariate(feature)

# def multivariate(feature):
#     # Create a figure with two subplots
#     fig, ax = plt.subplots(1, 2, figsize=(12, 8))
#     # Plot the first histogram on the first subplot
#     sns.histplot(df[df['HeartDisease']=='Yes'], x=feature, ax=ax[0])
#     ax[0].set_title(f"Histogram of {feature} for Heart Disease Yes")
#     # Plot the second histogram on the second subplot
#     sns.histplot(df[df['HeartDisease']=='No'], x=feature, ax=ax[1])
#     ax[1].set_title(f"Histogram of {feature} for Heart Disease No")

#     plt.figure(figsize=(15,10))
#     plt.subplot(1,2,1)
#     plt.pie(df[df['HeartDisease']=='Yes'][[feature]].value_counts(),labels=df[df['HeartDisease']=='Yes'][feature].unique().tolist(),autopct='%1.1f%%')
#     plt.legend(loc = (1.3,1))
#     plt.title(feature+' vs Heart Disease-Yes')
#     plt.subplot(1,2,2)
#     plt.pie(df[df['HeartDisease']=='No'][feature].value_counts(), labels=df[df['HeartDisease']=='No'][feature].unique().tolist(),autopct='%1.1f%%')
#     plt.legend(loc = (1.3,1))
#     plt.title(feature+' vs Heart Disease-No')

#     # Show the plot
#     plt.tight_layout()
#     plt.show()
 
# for feature in categorical_features:
#     multivariate(feature)
    
    
#%% Numerical Features

for feature in numerical_features:

    # Plot the first boxplot on the first subplot
    sns.boxplot(x=df[feature]).set_title("Boxplot of " +feature)
    plt.show()
    sns.distplot(df[feature], kde=False, bins=20, hist=True)
    plt.title("Histogram of " + feature )
    plt.show()
    
    sns.histplot(df, x =df[feature], hue=df['HeartDisease'], bins=15,  multiple ='stack', kde=True)
    plt.title("Histogram plot of " + feature + " with Heart Disease")
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


