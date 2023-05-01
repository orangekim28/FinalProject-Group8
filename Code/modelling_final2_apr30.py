#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from dython.nominal import associations
from dython.nominal import identify_nominal_columns
#%%
# Data Infomation
df = pd.read_csv('/Users/jiwoosuh/Documents/FinalProject-Group8/Code/heart_2020_cleaned.csv')
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

hdyes = df[df['HeartDisease']=='Yes']

plt.figure(figsize=(10,4))
sns.countplot(x = hdyes.AgeCategory.sort_values(),hue=hdyes.Sex)
plt.title("Age Category for Heart Disease Yes")
plt.savefig("/Users/jiwoosuh/Documents/FinalProject-Group8/plots/graph.png")
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(x = hdyes.Race.sort_values(),hue=hdyes.Sex)
plt.title("Race for Heart Disease Yes")
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(x = hdyes.GenHealth.sort_values(),hue=hdyes.Sex)
plt.title("Gen Health for Heart Disease Yes")
plt.show()
    
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

#%% boxplot for numerical features
for feature in numerical_features:
    sns.boxplot(x=df['HeartDisease'], y=df[feature]).set_title("Boxplot of " +feature)
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

#%%
# imbalance info
from imblearn.under_sampling import RandomUnderSampler
print("Before Undersampling, counts of label '1': {}".format(sum(y_train.values == 1)))
print("Before Undersampling, counts of label '0': {} \n".format(sum(y_train.values == 0)))

rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(x_train, y_train)
# X_train_res2, y_train_res2 = rus.fit_resample(x_train, y_train)

print('After Undersampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After Undersampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After Undersampling, counts of label '1': {}".format(sum(y_train_res.values == 1)))
print("After Undersampling, counts of label '0': {}".format(sum(y_train_res.values == 0)))
#%%
# distribution of target variable after oversampling
sns.countplot(x=y_train_res['HeartDisease'])
plt.title("Distribution of Heart Disease after Oversampling")
plt.show()

################################# modelling ########################################
#%%
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,roc_curve,f1_score,precision_recall_curve
#%%
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression()
    # 'SVC': SVC(kernel='linear')
}

for clf_name, clf in classifiers.items():
    # train pipeline with oversampling
    train_pipeline = Pipeline([
        ('sampler', RandomUnderSampler()),
        ('feature_selector', RFECV(estimator=clf, step=1, cv=StratifiedKFold(n_splits=5), scoring='f1')),
        ('classifier', clf)
    ])
    train_pipeline.fit(x_train, y_train)

    print(f"Classifier: {clf_name}")
    print(f"Selected features: {list(x_train.columns[train_pipeline.named_steps['feature_selector'].get_support()])}")
    print(f"Number of features selected: {train_pipeline.named_steps['feature_selector'].n_features_}")
    # print(
    #     f"Cross-validation score: {train_pipeline.named_steps['feature_selector'].grid_scores_[train_pipeline.named_steps['feature_selector'].n_features_ - 1]}")
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score - F1 (macro)")
    # plt.plot(range(1, len(train_pipeline.named_steps['feature_selector'].grid_scores_) + 1),
    #          train_pipeline.named_steps['feature_selector'].grid_scores_)
    plt.show()

    # apply trained pipeline to the test data without oversampling
    test_pipeline = Pipeline([
        ('feature_selector', train_pipeline.named_steps['feature_selector']),
        ('classifier', train_pipeline.named_steps['classifier'])
    ])
    y_pred = test_pipeline.predict(x_test)

    print(f"Classifier: {clf_name}")
    print(f"Selected features: {list(x_train.columns[train_pipeline.named_steps['feature_selector'].get_support()])}")
    print(f"Number of features selected: {train_pipeline.named_steps['feature_selector'].n_features_}")
    # print(
    #     f"Cross-validation score: {train_pipeline.named_steps['feature_selector'].grid_scores_[train_pipeline.named_steps['feature_selector'].n_features_ - 1]}")
    # # show the metrics of the train model
    y_pred = train_pipeline.predict(x_train)
    print(classification_report(y_train, y_pred))
    print('---------------------------------')

    y_pred_proba_train = train_pipeline.predict_proba(x_train)
    print(f"ROC AUC score: {roc_auc_score(y_train, y_pred_proba_train[:, 1])}")
    print('---------------------------------')

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    # plt.plot(range(1, len(train_pipeline.named_steps['feature_selector'].grid_scores_) + 1),
    #          train_pipeline.named_steps['feature_selector'].grid_scores_)
    plt.show()

    # apply trained pipeline to the test data without oversampling
    test_pipeline = Pipeline([
        ('feature_selector', train_pipeline.named_steps['feature_selector']),
        ('classifier', train_pipeline.named_steps['classifier']),

        # ('dimension_reducer', train_pipeline.named_steps['dimension_reducer'])

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
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
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

#%%
# # MLP classifier
# from sklearn.model_selection import GridSearchCV
#
# # Define your MLPClassifier with the default hyperparameters
# clf = MLPClassifier(random_state=42)
#
# # Define the parameter grid to search over
# param_grid = {
#     'classifier__hidden_layer_sizes': [(100,), (100, 50), (200, 100, 50)],
#     'classifier__activation': ['logistic', 'tanh', 'relu'],
#     'classifier__solver': ['sgd', 'adam'],
#     'classifier__alpha': [0.0001, 0.001, 0.01]
# }
#
# # Define the pipeline that includes oversampling using RandomOverSampler
# pipeline = Pipeline([
#     ('sampler', RandomOverSampler()),
#     ('classifier', clf)
# ])
#
# # Use GridSearchCV to search over the parameter grid
# cv = StratifiedKFold(n_splits=10)
# grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', verbose=1)
# grid.fit(x_train,y_train.to_numpy().flatten())
#
# # Print the best parameters and the corresponding score
# print("Best parameters: ", grid.best_params_)
# print("Best score: ", grid.best_score_)
#
# # train the model with the best parameters
# best_pipeline = Pipeline([
#     ('sampler', RandomOverSampler()),
#     ('classifier', MLPClassifier(random_state=42, **grid.best_params_))
# ])
# best_pipeline.fit(x_train, y_train)
#
# # apply trained pipeline to the test data
# # without oversampling
# test_pipeline = Pipeline([
#     ('classifier', best_pipeline.named_steps['classifier'])
# ])
# y_pred = test_pipeline.predict(x_test)
#
# print(classification_report(y_test, y_pred))
# print('---------------------------------')
# print(confusion_matrix(y_test, y_pred))
# y_pred_proba = test_pipeline.predict_proba(x_test)[:, 1]
# print(f"ROC-AUC score: {roc_auc_score(y_test, y_pred_proba)}")
#     #plot roc curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# plt.figure()
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# plt.show()
#
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2
# print(f"Balanced accuracy: {balanced_accuracy}")
# print('---------------------------------')
# # get the marco f1 score
# print(f"Macro F1 score: {f1_score(y_test, y_pred, average='macro')}")
# %%
# Trying MLP by keras
x_test,x_val,y_test,y_val = train_test_split(x_test,y_test,test_size=0.5,random_state=42)
print(x_test.shape)
print(x_val.shape)
print(y_test.shape)
print(y_val.shape)


# balancing it by SMOTE
from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(y_train.values == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train.values == 0)))

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(x_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res.values ==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res.values ==0)))

#%%
# model building
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# f1 score
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

input_shape = [X_train_res.shape[1]]

# A sequential dnn
model = keras.models.Sequential()

# Add the input layer
model.add(keras.layers.Flatten(input_shape=input_shape))

# Add the first hidden layer
model.add(keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.BatchNormalization())

# Add the output layer
model.add(keras.layers.Dense(1, activation='sigmoid'))

# The model summary
model.summary()
# Compile the model
model.compile(optimizer=keras.optimizers.legacy.Adam(),
              loss='binary_crossentropy',
              metrics=[f1])

# EarlyStopping callback
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

# ReduceLROnPlateau callback
reduce_lr_on_plateau_cb = keras.callbacks.ReduceLROnPlateau(
    factor=0.1,
    patience=2)

# Train, evaluate and save the best model
history = model.fit(X_train_res,y_train_res,
                    epochs=25,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping_cb,
                               reduce_lr_on_plateau_cb])

# plotting figure
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.tight_layout()
# plt.savefig(abspath_curr + '/result/figure/learning_curve.pdf')
plt.show()

loss, accuracy = model.evaluate(x_test,y_test)