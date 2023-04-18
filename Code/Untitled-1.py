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

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]=='No']["MentalHealth"], alpha=0.5,shade = True, color="#4285f4", label="No HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]=='Yes']["MentalHealth"], alpha=0.5,shade = True, color="#ea4335", label="HeartDisease", ax = ax)
ax.set_xlabel("MentalHealth")
ax.set_ylabel("Frequency")
ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.show()
# %%
fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]=='No']["PhysicalHealth"], alpha=0.5,shade = True, color="#4285f4", label="No HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]=='Yes']["PhysicalHealth"], alpha=0.5,shade = True, color="#ea4335", label="HeartDisease", ax = ax)
ax.set_xlabel("PhysicalHealth")
ax.set_ylabel("Frequency")
ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.show()
# %%
#How many patients that have heart disease are smoking?
have_heart_dis=df[df.HeartDisease =='Yes']
sns.countplot(x = have_heart_dis.Smoking ,hue=have_heart_dis.Sex)
# %%
#How many patients that have heart disease are drink alcohol?
have_heart_dis=df[df.HeartDisease =='Yes']
sns.countplot(x = have_heart_dis.AlcoholDrinking ,hue=have_heart_dis.Sex)
# %%
#How many patients that have heart disease have difficult walking?
have_heart_dis=df[df.HeartDisease =='Yes']
sns.countplot(x = have_heart_dis.DiffWalking ,hue=have_heart_dis.Sex)
# %%
#How many patients that have heart disease do physical exercise?
have_heart_dis=df[df.HeartDisease =='Yes']
sns.countplot(x = have_heart_dis.PhysicalActivity ,hue=have_heart_dis.Sex)
# %%
fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(df[df["HeartDisease"]=='No']["SleepTime"], alpha=0.5,shade = True, color="#4285f4", label="No HeartDisease", ax = ax)
sns.kdeplot(df[df["HeartDisease"]=='Yes']["SleepTime"], alpha=0.5,shade = True, color="#ea4335", label="HeartDisease", ax = ax)
ax.set_xlabel("SleepTime")
ax.set_ylabel("Frequency")
ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.show()
# %%
