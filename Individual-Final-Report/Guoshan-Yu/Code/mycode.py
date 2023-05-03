import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from dython.nominal import associations
from dython.nominal import identify_nominal_columns

# Data Infomation
df = pd.read_csv('heart_2020_cleaned.csv')

categorical_features=identify_nominal_columns(df)
numerical_features = ['BMI','PhysicalHealth','MentalHealth','SleepTime']

############################### Data preprocessing ##############################
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


################################# modelling ########################################
#%%
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,roc_curve,f1_score,precision_recall_curve
#%%
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression()
    # "XGBoost": xgb.XGBClassifier(),
    # "AdaBoost": AdaBoostClassifier()
}

for clf_name, clf in classifiers.items():
    # train pipeline with oversampling
    train_pipeline = Pipeline([
        ('sampler', RandomUnderSampler()),
        ('feature_selector', RFECV(estimator=clf, step=1, cv=StratifiedKFold(n_splits=10), scoring='f1')),
        ('classifier', clf)
    ])
    train_pipeline.fit(x_train, y_train)

    print(f"Classifier: {clf_name}")
    print(f"Selected features: {list(x_train.columns[train_pipeline.named_steps['feature_selector'].get_support()])}")
    print(f"Number of features selected: {train_pipeline.named_steps['feature_selector'].n_features_}")
    print(
         f"Cross-validation score: {train_pipeline.named_steps['feature_selector'].grid_scores_[train_pipeline.named_steps['feature_selector'].n_features_ - 1]}")
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score - F1 (macro)")
    plt.plot(range(1, len(train_pipeline.named_steps['feature_selector'].grid_scores_) + 1),
              train_pipeline.named_steps['feature_selector'].grid_scores_)
    plt.show()


    # show the metrics of the train model
    y_pred = train_pipeline.predict(x_train)
    print(classification_report(y_train, y_pred))
    print('---------------------------------')

    y_pred_proba_train = train_pipeline.predict_proba(x_train)
    print(f"ROC AUC score: {roc_auc_score(y_train, y_pred_proba_train[:, 1])}")
    print('---------------------------------')


    # apply trained pipeline to the test data without oversampling
    test_pipeline = Pipeline([
        ('feature_selector', train_pipeline.named_steps['feature_selector']),
        ('classifier', train_pipeline.named_steps['classifier']),

    ])
    y_pred = test_pipeline.predict(x_test)

    print(classification_report(y_test, y_pred))
    print('---------------------------------')
    # print the confusion matrix in to a tabel form
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Predicted No Disease', 'Predicted Disease'],
                       index=['Actual No Disease', 'Actual Disease']))
    print('---------------------------------')
    y_pred_proba = test_pipeline.predict_proba(x_test)[:, 1]
    print(f"ROC-AUC score: {roc_auc_score(y_test, y_pred_proba)}")
    print('---------------------------------')
    # plot the precision_recall_curve
   # draw the no skill line of the precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.figure()    
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    baseline = len(y_test[(y_test['HeartDisease']==1)])/len(y_test)
    plt.plot([0, 1],[baseline,baseline], linestyle='--')
    plt.title(f'Precision-Recall curve of {clf_name}')
    plt.show()
    print('---------------------------------')
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_proba_train[:, 1])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr_train, tpr_train,
             label='Train ROC curve (area = %0.2f)' % roc_auc_score(y_train, y_pred_proba_train[:, 1]))
    plt.plot(fpr, tpr, label='Test ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve of {clf_name}')
    plt.legend()
    plt.show()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2
    print(f"Balanced accuracy: {balanced_accuracy}")
    print('---------------------------------')

