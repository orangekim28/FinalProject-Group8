#%%
import pandas as pd
import numpy as np
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
# drop the target variable
categorical_features.remove('HeartDisease')
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
# %%
# encode the numercial features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_clean[numerical_features] = scaler.fit_transform(x_clean[numerical_features])
print(x_clean.head())
# %% 
#one hot encoding the categorical features
x_encode = pd.get_dummies(x_clean, columns=categorical_features)
print(x_encode.head())
# encode the y variable with yes:1 and no:0
y_encode = y_clean.replace({'HeartDisease': {'Yes': 1, 'No': 0}})

# %%
# Divide each categorical column by the square root of its probability(sqrt(μₘ))
#categorical_features exclude the numerical features
categorical_features = list(set(x_encode.columns) - set(numerical_features))
for i in categorical_features:
    x_encode[i] = x_encode[i]/np.sqrt(x_encode[i].mean())
print(x_encode.head())

# %% split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x_encode, y_encode, test_size=0.2, random_state=42)

# %%
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,roc_curve,f1_score,precision_recall_curve
from prince import FAMD
from imblearn.under_sampling import RandomUnderSampler
# import the PCA
from sklearn.decomposition import PCA
# %% ramdomundersampling the training data
rd = RandomUnderSampler()
x_train, y_train = rd.fit_resample(x_train, y_train)

# %%
# apply pca to the data
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
# plot the pca graph
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

# %%
# train the dataset with the logistic regression
# ravel the y_train
y_train = y_train.values.ravel()
dt = LogisticRegression()
dt.fit(x_pca, y_train)
# predict the test data
y_pred = dt.predict(x_test)

# %%
# show the metrics of the train model not the test model
from sklearn.metrics import balanced_accuracy_score
print(classification_report(y_train, dt.predict(x_pca)))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_train, dt.predict(x_pca))}")
print('---------------------------------')

print(classification_report(y_test, y_pred))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
# from sklearn.metrics import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)}")

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
rfpca = RandomForestClassifier()
rfpca.fit(x_pca, y_train)
# predict the test data
y_pred = rfpca.predict(x_test)

# %%
# show the metrics of the train model not the test model
from sklearn.metrics import balanced_accuracy_score
print(classification_report(y_train, rfpca.predict(x_pca)))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_train, rfpca.predict(x_pca))}")
print('---------------------------------')

print(classification_report(y_test, y_pred))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
# from sklearn.metrics import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)}")

#%%
mlppca = MLPClassifier()
mlppca.fit(x_pca, y_train)
# predict the test data
y_pred = mlppca.predict(x_test)

# %%
# show the metrics of the train model not the test model
from sklearn.metrics import balanced_accuracy_score
print(classification_report(y_train, mlppca.predict(x_pca)))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_train, mlppca.predict(x_pca))}")
print('---------------------------------')

print(classification_report(y_test, y_pred))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
# from sklearn.metrics import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)}")

#%%
mlppca = MLPClassifier()
mlppca.fit(x_pca, y_train)
# predict the test data
y_pred = mlppca.predict(x_test)

# %%
# show the metrics of the train model not the test model
from sklearn.metrics import balanced_accuracy_score
print(classification_report(y_train, mlppca.predict(x_pca)))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_train, mlppca.predict(x_pca))}")
print('---------------------------------')

print(classification_report(y_test, y_pred))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
# from sklearn.metrics import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)}")

#%%
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
knnpca = KNeighborsClassifier()
knnpca.fit(x_pca, y_train)
# predict the test data
y_pred = knnpca.predict(x_test)

# %%
# show the metrics of the train model not the test model
from sklearn.metrics import balanced_accuracy_score
print(classification_report(y_train, knnpca.predict(x_pca)))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_train, knnpca.predict(x_pca))}")
print('---------------------------------')

print(classification_report(y_test, y_pred))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
# from sklearn.metrics import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)}")

#%%
xgbpca = xgb.XGBClassifier()
xgbpca.fit(x_pca, y_train)
# predict the test data
y_pred = xgbpca.predict(x_test)

# %%
# show the metrics of the train model not the test model
from sklearn.metrics import balanced_accuracy_score
print(classification_report(y_train, xgbpca.predict(x_pca)))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_train, xgbpca.predict(x_pca))}")
print('---------------------------------')

print(classification_report(y_test, y_pred))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
# from sklearn.metrics import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)}")


#%%
svmpca = SVC()
svmpca.fit(x_pca, y_train)
# predict the test data
y_pred = svmpca.predict(x_test)

# %%
# show the metrics of the train model not the test model
from sklearn.metrics import balanced_accuracy_score
print(classification_report(y_train, svmpca.predict(x_pca)))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_train, svmpca.predict(x_pca))}")
print('---------------------------------')

print(classification_report(y_test, y_pred))
print('---------------------------------')
# show the balanced accuracy score
# import balanced_accuracy_score
# from sklearn.metrics import balanced_accuracy_score
print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)}")

# %% 
# classifiers = {
#       'Decision Tree': DecisionTreeClassifier(),
#     # 'Random Forest': RandomForestClassifier(),
#     #   'Logistic Regression': LogisticRegression(),
#     #    'SVC': SVC(kernel='linear')
# } 

# for clf_name, clf in classifiers.items():
#     # train pipeline with oversampling
#     train_pipeline = Pipeline([
#         ('sampler', RandomUnderSampler()),
#         # apply pca to the data
#         ('dimension_reducer', PCA(n_components=0.95)),
#         ('classifier', clf)
#     ])
#     train_pipeline.fit(x_train, y_train)
    
#     print(f"Classifier: {clf_name}")
#     # print(f"Selected features: {list(x_train.columns[train_pipeline.named_steps['feature_selector'].get_support()])}")
#     # print(f"Number of features selected: {train_pipeline.named_steps['feature_selector'].n_features_}")
#     # print(f"Cross-validation score: {train_pipeline.named_steps['feature_selector'].grid_scores_[train_pipeline.named_steps['feature_selector'].n_features_ - 1]}")
#     # show the metrics of the train model
#     y_pred = train_pipeline.predict(x_train)
#     print(classification_report(y_train, y_pred))
#     print('---------------------------------')
 
#     y_pred_proba_train = train_pipeline.predict_proba(x_train)
#     print(f"ROC AUC score: {roc_auc_score(y_train, y_pred_proba_train[:, 1])}")
#     print('---------------------------------')
  
#     # plt.figure()
#     # plt.xlabel("Number of features selected")
#     # plt.ylabel("Cross validation score")
#     # plt.plot(range(1, len(train_pipeline.named_steps['feature_selector'].grid_scores_) + 1), train_pipeline.named_steps['feature_selector'].grid_scores_)
#     # plt.show()
    
#     # apply trained pipeline to the test data without oversampling
#     test_pipeline = Pipeline([
#         # ('feature_selector', train_pipeline.named_steps['feature_selector']),
#         ('dimension_reducer', train_pipeline.named_steps['dimension_reducer']),
#         ('classifier', clf)
#         # ('dimension_reducer', train_pipeline.named_steps['dimension_reducer']),
#     ])
#     y_pred = test_pipeline.predict(x_test)

#     print(classification_report(y_test, y_pred))
#     print('---------------------------------')
#     # print the confusion matrix in to a tabel form
#     print(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Predicted No Disease', 'Predicted Disease'], index=['Actual No Disease', 'Actual Disease']))
#     print('---------------------------------')
#     y_pred_proba = test_pipeline.predict_proba(x_test)[:, 1]
#     print(f"ROC-AUC score: {roc_auc_score(y_test, y_pred_proba)}")
#     print('---------------------------------')
#     # plot the precision_recall_curve
#     precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
#     plt.figure()    
#     plt.plot(recall, precision)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title(f'Precision-Recall curve of {clf_name}')
#     plt.show()
#     print('---------------------------------')
#     fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_proba_train[:, 1])
#     fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
#     plt.figure()
#     plt.plot(fpr_train, tpr_train, label='Train')
#     plt.plot(fpr, tpr, label='Test')
#     plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC curve of {clf_name}')
#     plt.legend()
#     plt.show()
    
    
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() 
#     balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2
#     print(f"Balanced accuracy: {balanced_accuracy}")
#     print('---------------------------------')

# %%
