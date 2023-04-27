#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dython.nominal import associations
from dython.nominal import identify_nominal_columns
#%%
# Read the data
df = pd.read_csv('heart_2020_cleaned.csv')
# %%
# feature summary
print(df.head())
print(df.columns)
print(df.shape)
categorical_features=identify_nominal_columns(df)
numerical_features = ['BMI','PhysicalHealth','MentalHealth','SleepTime']
print(df.columns)
print(df.info())
print(df.describe())
print(df.isna().sum())

# %% EDA analysis - categorical features - Diesease
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

# %%
# plot the boxplot of the numerical features
def boxplot(feature):
    plt.figure(figsize=(10,10))
    sns.boxplot(y=feature,data=df).set(title = feature)

    plt.tight_layout()
    plt.show()

for i in numerical_features:
    boxplot(i)
  
# %%  
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

y = df_clean['HeartDisease']
x = df_clean.drop(['HeartDisease'],axis=1)

# %% data preprocessing
## one hot encoding
df_encoded = pd.get_dummies(df_clean,columns=categorical_features)
y_encoded = df_encoded[['HeartDisease_Yes','HeartDisease_No']]
x_encoded = df_encoded.drop(['HeartDisease_Yes','HeartDisease_No'],axis=1)
print(df_encoded.columns)
print(df_encoded.shape) 
# %% 
## balance the data
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(x_encoded.to_numpy(), y_encoded.to_numpy())
print(x_smote.shape)
print(y_smote.shape)

# %%
## normalize the numercial features and concat the rest of the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_smote)
x_scaled = pd.DataFrame(x_scaled,columns=x_encoded.columns)
print(x_scaled.shape)
print(x_scaled.columns)


#%% feature selection
## use SelectKBest to select the best features from the categorical features
from sklearn.feature_selection import SelectKBest, chi2
bestfeatures = SelectKBest(score_func=chi2, k='all') 
fit = bestfeatures.fit(x_scaled,y_smote)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x_scaled.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['features','Score']
# sort the features by the score
print(featureScores.nlargest(50,'Score'))
#%%
# transform the data 
bestfeatures = SelectKBest(score_func=chi2, k=45)
fit = bestfeatures.fit(x_scaled,y_smote)
x_selected = fit.transform(x_scaled)
# transform x_selected to dataframe
x_selected = pd.DataFrame(x_selected,columns=x_scaled.columns[fit.get_support()])
print(x_selected.shape)

# %% feature reduction
## use FAMD to see the variance of the features
from prince import FAMD
famd = FAMD(n_components=45, n_iter=3, copy=True, check_input=True, engine='auto', random_state=42)
famd = famd.fit(x_selected)
print(famd.explained_inertia_) 
## also draw a diagram to see the variance of the features
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(famd.explained_inertia_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# %%
