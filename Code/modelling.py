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

##################################EDA analysis#########################
#%% Self-charateristic features - jiwoo
jw = ["BMI", "Sex", "AgeCategory", "Race", "GenHealth", "HeartDisease"]
df_jw = df[jw]
cf = list(df_jw.select_dtypes(include=['object']).columns)
nf = list(df_jw.select_dtypes(include=['int64', 'float64']).columns)
cf.remove('HeartDisease')
print(f"Categorical features: {cf}")
print(f"Numerical features: {nf}")
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
# MentalHealth variable - kyuri
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
#%%
def outliers(feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    print('feature:',feature)
    print('Q1:',Q1)
    print('Q3:',Q3)
    print('IQR:',IQR)
    print(f'number of outliners in upper bound:{df[df[feature] > Q3 + 1.5 * IQR].shape[0]}')
    print(f'number of outliners in lower bound:{df[df[feature] < Q1 - 1.5 * IQR].shape[0]}')
    print('---------------------------------')
for i in numerical_features:
    outliers(i)

# %%
df_clean = df.copy()
# remove the outliers
def remove_outliers(feature):
    Q1 = df_clean[feature].quantile(0.25)
    Q3 = df_clean[feature].quantile(0.75)
    IQR = Q3 - Q1
    df_clean.drop(df_clean[df_clean[feature] > Q3 + 1.5 * IQR].index, inplace = True)
    df_clean.drop(df_clean[df_clean[feature] < Q1 - 1.5 * IQR].index, inplace = True)
for i in numerical_features:
    remove_outliers(i)

print(df_clean.shape)

y_clean = df_clean[['HeartDisease']]
x_clean = df_clean.drop(['HeartDisease'],axis=1)

# %% data preprocessing
# encode the target with yes:1 and no:0
y_encoded = y_clean.replace({'HeartDisease':{'Yes':1,'No':0}})
## one hot encoding for categorical features
x_encoded = pd.get_dummies(x_clean)
print(x_encoded.columns)
print(x_encoded.shape) 
df_encoded = pd.concat([x_encoded,y_encoded],axis=1)

# %%
## normalize the numercial features from the balance dataset and concat the rest of the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_encoded)
x_scaled = pd.DataFrame(x_scaled,columns=x_encoded.columns)
print(x_scaled.shape)
print(x_scaled.columns)


#%% split the data into train and test before feature selection(unbiased)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y_encoded,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#%%
# cross validation on the train set for feature selection and model selection
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
#%%
#%%
# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    # 'Random Forest': RandomForestClassifier(),
    # 'SVC': SVC(),
    'MLP': MLPClassifier()
}

# Set up K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# set up the oversampler
sampler = RandomOverSampler()
# Loop through models and evaluate performance
for name, model in models.items():
    scores = []
    for i, (train_idx, val_idx) in enumerate(kfold.split(x_train)):
        # Split data into training and validation sets for this fold
        # print(f"Fold{i}")
  
        X_train_fold, X_val_fold = x_scaled.iloc[train_idx], x_scaled.iloc[val_idx]
        y_train_fold, y_val_fold = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_fold,y_train_fold)
        # check the balance of the resampled data
        # print(y_train_resampled.value_counts())

        bestfeatures = SelectKBest(score_func=chi2, k=30)
        fit = bestfeatures.fit(X_train_resampled,y_train_resampled)
        # transform the data
        X_train_featured = fit.transform(X_train_resampled)
        X_val_featured = fit.transform(X_val_fold)
        # model.fit(X_train_featured, y_train_resampled)
        model.fit(X_train_resampled, y_train_resampled)

        # Predict on the validation data and calculate accuracy
        # y_pred = model.predict(X_val_featured)
        y_pred = model.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred)
        # print(f"accuracy for {i}: {acc}")
        scores.append(acc)
     
    # Print the average accuracy score for this model
    print(f'{name}: Average accuracy score = {sum(scores) / len(scores):.2f}')
    # resample the train data
#%%
    X_train_resampled_all, y_train_resampled_all = sampler.fit_resample(x_train,y_train)
    X_train_resampled_all = bestfeatures.fit_transform(X_train_resampled_all,y_train_resampled_all)
    model.fit(X_train_resampled_all, y_train_resampled_all)
    # transform the x test data
    x_test_featured = bestfeatures.transform(x_test)
    y_pred = model.predict(x_test_featured)
    acc = accuracy_score(y_test, y_pred)
    print(f'{name}: Test accuracy score = {acc:.2f}')
    print('---------------------------------')
    # evaluate the model with confusion matrix with column names
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                        index = ['Actual No Disease','Actual Disease'],
                        columns = ['Predicted No Disease','Predicted Disease'])
    print(f'Confusion Matrix for {name}')
    print(cm_df)
    print('---------------------------------')
    # evaluate the model with ROC curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y_test, model.predict(x_test_featured))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test_featured)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {name}')
    plt.legend(loc="lower right")
    plt.show()
    print('---------------------------------')
    # evaluate the model with precision-recall curve
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import auc
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(x_test_featured)[:,1]) 
    average_precision = average_precision_score(y_test, model.predict_proba(x_test_featured)[:,1])
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1])
    plt.title(f'Precision-Recall curve for {name}: AP={average_precision:0.2f}')
    plt.show()
    print('---------------------------------')
    # evaluate the model with f1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred)
    print(f'f1 score for {name}: {f1:.2f}')
    print('---------------------------------')
    # evaluate the model with classification report
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    print('---------------------------------')




#%%

#%% feature selection
# ## use SelectKBest to select the best features from the categorical features
# from sklearn.feature_selection import SelectKBest, chi2
# bestfeatures = SelectKBest(score_func=chi2, k=50) 
# fit = bestfeatures.fit(x_scaled,y_encoded)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(x_scaled.columns)
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['features','Score']
# # sort the features by the score
# print(featureScores.nlargest(50,'Score'))
#%%
# bestfeatures = SelectKBest(score_func=chi2, k=45)
# fit = bestfeatures.fit(x_scaled,y_encoded)
# x_selected = fit.transform(x_scaled)
# # x_selected = pd.DataFrame(x_selected,columns=x_scaled.columns[fit.get_support()])
# print(x_selected.shape)

# %%
