import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
#%%
df = pd.read_csv("heart_2020_cleaned.csv")
df.info()
print(f"Number of observations: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
col_df = list(df.columns.values)
for col in col_df:
    print(col, ':', str(df[col].unique()))

#%%
jw = ["BMI", "Sex", "AgeCategory", "Race", "GenHealth", "HeartDisease"]
df_jw = df[jw]
sns.boxplot(x=df[df['HeartDisease']=='Yes']["BMI"], color="#ea4335")
plt.title("Boxplot of BMI for Heart Disease Yes")
plt.show()
sns.histplot(df[df['HeartDisease']=='Yes'], x="BMI", kde=True, color="#ea4335")
plt.title("Histogram of BMI for Heart Disease Yes")
plt.show()
sns.boxplot(x=df[df['HeartDisease']=='No']["BMI"], color='#4285f4')
plt.title("Boxplot of BMI for Heart Disease No")
plt.show()
sns.histplot(df[df['HeartDisease']=='No'], x="BMI", kde=True, color='#4285f4')
plt.title("Histogram of BMI for Heart Disease Yes")
plt.show()
# for col in jw:
#     plt.figure(figsize=(13, 6))
#     sns.countplot(x=df[col], hue=col, data=df_jw, palette='YlOrBr')
#     plt.xlabel(f'{col}')
#     plt.legend()
#     plt.ylabel('Frequency')
#     plt.show()