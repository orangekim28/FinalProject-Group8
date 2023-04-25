#%%
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

#%%
df = pd.read_csv("heart_2020_cleaned.csv")
df.info()
# %%
import matplotlib.pyplot as plt
import seaborn as sns

# for Categorical columns:
# - distribution of itself (by histogram)
# - compare by heart disease (by pie, bar chart)
# for Numerical columns:
# - distribution of itself (by historgram, scatterplot, boxplot)
# - compare by heart disease (by stacked histogram)

#%%
# MentalHealth variable
sns.distplot(df['MentalHealth'], kde=False, bins=20, hist=True)
plt.show()

sns.boxplot(x='MentalHealth', data=df)
plt.show()

plt.scatter(x=)

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
