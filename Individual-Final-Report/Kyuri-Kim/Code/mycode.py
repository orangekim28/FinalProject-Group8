#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

#%%
# distribution of target variable
sns.countplot(x=df['HeartDisease'])
plt.title("Distribution of Heart Disease")
plt.show()

##################################EDA analysis#########################
#%%
# Categorical Features
# def univaiate(feature):
#     freq_table = pd.crosstab(index=df[feature],columns='count')
#     print(freq_table)
#     plt.figure(figsize=(18,10))
#     plt.subplot(1,2,1)
#     sns.histplot(x=feature,data=df).set(title = 'Histogram of '+feature)
#     plt.subplot(1,2,2)
#     plt.pie(df[feature].value_counts(),labels=df[feature].unique().tolist(),autopct='%1.1f%%')
#     plt.legend(loc = (1.3,1))
#     plt.title(feature)
#     plt.tight_layout()
#     plt.show()

#%%
# Categorical Features
def univaiate(feature):
    freq_table = pd.crosstab(index=df[feature],columns='count').sort_values('count', ascending=False)
    print(freq_table)
    plt.figure(figsize=(6,6))
    ax1 = plt.subplot(1,2,1)
    freq_table.plot.bar(y='count', ax = ax1) #rot
    plt.title(f'Histogram of {feature}')
    ax2 = plt.subplot(1,2,2)
    freq_table.plot.pie(y='count', ax=ax2)
    plt.title(f'Pie Chart of {feature}')
    plt.legend(loc = (1.3,1))
    plt.suptitle(feature)
    # plt.tight_layout()
    plt.show()

for feature in categorical_features:
    univaiate(feature)

#%% Subplotting
fig, axes = plt.subplots(nrows=len(categorical_features), ncols=2, figsize=(16, 8*len(categorical_features)))

for i, feature in enumerate(categorical_features):
    freq_table = pd.crosstab(index=df[feature], columns='count').sort_values('count', ascending=False)
    
    # Bar chart
    freq_table.plot.bar(y='count', ax=axes[i][0])
    axes[i][0].set_title(f'Histogram of {feature}')
    
    # Pie chart
    freq_table.plot.pie(y='count', ax=axes[i][1])
    axes[i][1].set_title(f'Pie Chart of {feature}')
    
    axes[i][1].legend(loc=(1.3, 0.5))
    
plt.tight_layout()
plt.show()

#%%
## multivariate analysis
if 'HeartDisease' in categorical_features:
    categorical_features.remove('HeartDisease')

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
#%% Subplotting
if 'HeartDisease' in categorical_features:
    categorical_features.remove('HeartDisease')

def multivariate(feature):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    
    sns.countplot(x='HeartDisease', hue=feature, data=df, ax=axes[0])
    axes[0].set_title(feature+' vs Heart Disease')
    
    hd_yes_counts = df[df['HeartDisease']=='Yes'][feature].value_counts()
    hd_yes_labels = hd_yes_counts.index.tolist()
    hd_yes_sizes = hd_yes_counts.tolist()
    axes[1].pie(hd_yes_sizes, labels=hd_yes_labels, autopct='%1.1f%%')
    axes[1].set_title(feature+' vs Heart Disease-yes')
    
    plt.tight_layout()
    plt.show()
    
for feature in categorical_features:
    multivariate(feature)


##%
hdyes = df[df['HeartDisease']=='Yes']

plt.figure(figsize=(10,4))
sns.countplot(x = hdyes.AgeCategory.sort_values(),hue=hdyes.Sex)
plt.title("Age Category for Heart Disease Yes")
# plt.savefig("/Users/jiwoosuh/Documents/FinalProject-Group8/plots/graph.png")
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(x = hdyes.Race.sort_values(),hue=hdyes.Sex)
plt.title("Race for Heart Disease Yes")
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(x = hdyes.GenHealth.sort_values(),hue=hdyes.Sex)
plt.title("Gen Health for Heart Disease Yes")
plt.show()

#%% Subplotting
hdyes = df[df['HeartDisease']=='Yes']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))

sns.countplot(x=hdyes.AgeCategory.sort_values(), hue=hdyes.Sex, ax=axes[0])
axes[0].set_title("Age Category for Heart Disease Yes")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', labelrotation=90)

sns.countplot(x=hdyes.Race.sort_values(), hue=hdyes.Sex, ax=axes[1])
axes[1].set_title("Race for Heart Disease Yes")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis='x', labelrotation=90)

sns.countplot(x=hdyes.GenHealth.sort_values(), hue=hdyes.Sex, ax=axes[2])
axes[2].set_title("Gen Health for Heart Disease Yes")
axes[2].set_ylabel("Count")
axes[2].tick_params(axis='x', labelrotation=90)

plt.tight_layout()
plt.show()

    
#%% Numerical Features - Univariate
for feature in numerical_features:
    # Plot the first boxplot on the first subplot
    sns.boxplot(x=df[feature]).set_title("Boxplot of " +feature)
    plt.show()
    sns.distplot(df[feature], kde=False, bins=20, hist=True)
    plt.title("Histogram of " + feature )
    plt.show()

#%% Subplotting

fig, axes = plt.subplots(nrows=len(numerical_features), ncols=2, figsize=(12, 6*len(numerical_features)))

for i, feature in enumerate(numerical_features):
    # Boxplot
    sns.boxplot(x=df[feature], ax=axes[i][0])
    axes[i][0].set_title("Boxplot of " + feature)
    
    # Histogram
    sns.distplot(df[feature], kde=False, bins=20, hist=True, ax=axes[i][1])
    axes[i][1].set_title("Histogram of " + feature)
    
plt.tight_layout()
plt.show()


#%% boxplot for numerical features - multivariate
fig, axes = plt.subplots(nrows=len(numerical_features), ncols=2, figsize=(12, 6*len(numerical_features)))

for i, feature in enumerate(numerical_features):
    sns.boxplot(x=df['HeartDisease'], y=df[feature], ax=axes[i][0])
    axes[i][0].set_title("Boxplot of " +feature)

    sns.histplot(df, x=df[feature], hue=df['HeartDisease'], bins=15, multiple='stack', kde=True, ax=axes[i][1])
    axes[i][1].set_title("Histogram plot of " + feature + " by Heart Disease")

plt.tight_layout()
plt.show()
