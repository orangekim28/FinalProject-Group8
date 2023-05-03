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
# tryinig K-fold 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
# Initialize k-fold
kfold = KFold(n_splits=10)

# Train and evaluate logistic regression model using k-fold cross validation
for train_idx, test_idx in kfold.split(X):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Initialize logistic regression model
    model = LogisticRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the test data
    score = model.score(X_test, y_test)
    print(f"Accuracy: {score:.4f}")


#%%
from sklearn.feature_selection import SelectPercentile, chi2

FeatureSelection = SelectPercentile(score_func=chi2, percentile=40)
X_new = FeatureSelection.fit_transform(x_train, y_train)
print('X_new shape is', X_new.shape)
print('selected features are:', FeatureSelection.get_support())
print('number of selected features:', FeatureSelection.get_support().sum())
print('features names are', FeatureSelection.get_feature_names_out())

selectedf40 = ['Smoking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic',
       'PhysicalActivity', 'GenHealth', 'KidneyDisease',
       'SkinCancer', 'BMI']
x_clean2 = x_clean[selectedf40]
x_encoded2 = pd.get_dummies(x_clean2)
print(x_encoded2.columns)
print(x_encoded2.shape)
scaler = MinMaxScaler()
x_scaled2 = scaler.fit_transform(x_encoded2)
x_scaled2 = pd.DataFrame(x_scaled2,columns=x_encoded2.columns)
print(x_scaled2.shape)
print(x_scaled2.columns)
x_train2,x_test2,y_train2,y_test2 = train_test_split(x_scaled2,y_encoded,test_size=0.2,random_state=42)
print(x_train2.shape)
print(x_test2.shape)
print(y_train2.shape)
print(y_test2.shape)
x_test2,x_val,y_test2,y_val = train_test_split(x_test2,y_test2,test_size=0.5,random_state=42)
print(x_test2.shape)
print(x_val.shape)
print(y_test2.shape)
print(y_val.shape)

#%%
from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(y_train2.values == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train2.values == 0)))

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(x_train2, y_train2)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res.values ==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res.values ==0)))

#%%
simple eda for balanced df
balanced_df = pd.concat([X_train_res2, y_train_res2], axis=1)
trainres_cols = X_train_res2.columns
for col in trainres_cols:
    sns.countplot(x=balanced_df[col], hue=balanced_df.HeartDisease)
    plt.show()


#%%

from imblearn.under_sampling import RandomUnderSampler

print("Before Undersampling, counts of label '1': {}".format(sum(y_train.values == 1)))
print("Before Undersampling, counts of label '0': {} \n".format(sum(y_train.values == 0)))

rus = RandomUnderSampler(random_state=42)
X_train_res2, y_train_res2 = rus.fit_resample(x_train2, y_train2)
# X_train_res2, y_train_res2 = rus.fit_resample(x_train, y_train)

print('After Undersampling, the shape of train_X: {}'.format(X_train_res2.shape))
print('After Undersampling, the shape of train_y: {} \n'.format(y_train_res2.shape))

print("After Undersampling, counts of label '1': {}".format(sum(y_train_res2.values == 1)))
print("After Undersampling, counts of label '0': {}".format(sum(y_train_res2.values == 0)))


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
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,roc_curve,f1_score,precision_recall_curve
#LR threshold test
lr = LogisticRegression()
lr.fit(X_train_res2, y_train_res2)
lr_predictions = lr.predict(x_test2)
f1_score_test_lr = f1_score(y_test2, lr_predictions)
# print('F1 score for LR on the training set:', f1_score_train)
print('F1 score for LR on the test set:', f1_score_test_lr)
print(classification_report(y_test2, lr_predictions))
print('---------------------------------')
print(confusion_matrix(y_test2, lr_predictions))
# Make predictions on test set with different thresholds
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = (lr.predict_proba(x_test2[:, 1] > thresh)).astype(int)
    precision = precision_score(y_test2, y_pred, pos_label=1)
    recall = recall_score(y_test2, y_pred, pos_label=1)
    print(f'Threshold: {thresh:.1f} | Precision: {precision:.3f} | Recall: {recall:.3f}')

#%%
#LR threshold test

lr = LogisticRegression()
lr.fit(X_train_res2, y_train_res2)
lr_predictions = lr.predict(x_test2)
f1_score_test_lr = f1_score(y_test2, lr_predictions)
# print('F1 score for LR on the training set:', f1_score_train)
print('F1 score for LR on the test set:', f1_score_test_lr)
print(classification_report(y_test2, lr_predictions))
print('---------------------------------')
print(confusion_matrix(y_test2, lr_predictions))
# Make predictions on test set with different thresholds
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = (lr.predict_proba(x_test2)[:, 1] > thresh).astype(int)
    precision = precision_score(y_test2, y_pred, pos_label=1)
    recall = recall_score(y_test2, y_pred, pos_label=1)
    print(f'Threshold: {thresh:.1f} | Precision: {precision:.3f} | Recall: {recall:.3f}')

#%%
from sklearn.ensemble import RandomForestClassifier

# Create the classifier object
rf = RandomForestClassifier()
# Train the model on the resampled data
rf.fit(X_train_res2, y_train_res2)
# Make predictions on the test data
rf_predictions = rf.predict(x_test2)
# Calculate F1 score on the test set
f1_score_test_rf = f1_score(y_test2, rf_predictions)
# Print the results
print('F1 score for Random Forest on the test set:', f1_score_test_rf)
print(classification_report(y_test2, rf_predictions))
print('---------------------------------')
print(confusion_matrix(y_test2, rf_predictions))


#%%
mlp = MLPClassifier(hidden_layer_sizes=(36, 256, 256), max_iter=1000)
mlp.fit(X_train_res2, y_train_res2.values.ravel())
mlppredictions = mlp.predict(x_test2)
f1_score_test_mlp = f1_score(y_test2, mlppredictions)
print('F1 score for MLP on the test set:', f1_score_test_mlp)
print(classification_report(y_test2, mlppredictions))
print('---------------------------------')
print(confusion_matrix(y_test2, mlppredictions))

classifiers = {
    "XGBoost" : xgb.XGBClassifier(),
    "AdaBoost" : AdaBoostClassifier()
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

loss, accuracy = model.evaluate(x_test2,y_test2)



# Load data
df = pd.read_csv("heart.csv")

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"])

# Split features and target variable
X = df.drop(columns=["target"])
y = df["target"]

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
# Initialize k-fold
kfold = KFold(n_splits=10)

# Train and evaluate logistic regression model using k-fold cross validation
for train_idx, test_idx in kfold.split(X):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Initialize logistic regression model
    model = LogisticRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the test data
    score = model.score(X_test, y_test)
    print(f"Accuracy: {score:.4f}")
