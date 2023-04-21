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
# Plot for Mental Health rating for those who have HeartDisease vs. No HeartDisease

plt.bar(x='MentalHealth', bins ='auto', alpha=0.5 , edgecolor='black', linewidth=1)
plt.show()
sns.boxplot(x='MentalHealth', data=df)
plt.show()

sns.histplot(df, x = 'MentalHealth', hue='HeartDisease', bins=15,  multiple ='stack', kde=True)
plt.show()

# %%
# Plot for days a patient had physical injury during past 30 days for those who have HeartDisease vs. No HeartDisease
sns.boxplot(x='PhysicalHealth', data=df)
plt.show()

sns.histplot(df, x = 'PhysicalHealth', hue='HeartDisease', bins=15,  multiple ='stack', kde=True)
plt.show()

# %%
# Plot for days a patient had physical injury during past 30 days for those who have HeartDisease vs. No HeartDisease
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
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.show()

sns.countplot(x = df.HeartDisease ,hue=df.AlcoholDrinking)
# %%
#Categorical Variables -- Difficult Walking
sns.countplot(x=df["DiffWalking"])
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.show()

sns.countplot(x = df.HeartDisease ,hue=df.DiffWalking)
# %%
#Categorical Variables -- Physical Activity
sns.countplot(x=df["PhysicalActivity"])
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.show()

sns.countplot(x = df.HeartDisease ,hue=df.PhysicalActivity)

