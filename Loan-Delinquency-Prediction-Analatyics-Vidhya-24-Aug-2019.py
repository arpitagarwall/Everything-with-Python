# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 01:05:11 2019

@author: erragpa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/erragpa/Desktop/Loan Delinquency Prediction/train.csv')
df.columns
df.drop(['loan_id', 'origination_date', 'first_payment_date', 'financial_institution'], axis=1, inplace=True)

df.dtypes

df.isnull().sum()


X = df.iloc[:, :-1].values
y = df.iloc[:, 24].values

df.dtypes

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
onehotencoder = OneHotEncoder(categorical_features = [8])
X = onehotencoder.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


print('accuracy %s' % accuracy_score(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
print('confusion matrix \n %s' % cm)
print(classification_report(y_test, predictions))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


import pickle

filename = 'final_model.sav'
pickle.dump(model, open(filename, 'wb'))

load_model = pickle.load(open(filename, 'rb'))
result = load_model.score(X_test, y_test)
print(result)


final_predict = pd.read_csv('C:/Users/erragpa/Desktop/Loan Delinquency Prediction/test.csv')
final_predict.columns

final_predict.drop(['loan_id', 'origination_date', 'first_payment_date', 'financial_institution'], axis=1, inplace=True)

final_predict.dtypes

final_predict = final_predict.values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_final_predict = LabelEncoder()
final_predict[:, 0] = labelencoder_final_predict.fit_transform(final_predict[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
final_predict[:, 8] = labelencoder_final_predict.fit_transform(final_predict[:, 8])
onehotencoder = OneHotEncoder(categorical_features = [8])
final_predict = onehotencoder.fit_transform(final_predict).toarray()


X_train = final_predict
pred = load_model.predict(X_train)
predicted = [round(value) for value in pred]
print(predicted)


solution = pd.DataFrame(predicted)
solution.columns


last = pd.read_csv('C:/Users/erragpa/Desktop/Loan Delinquency Prediction/test.csv')

final_subv1 = last.join(solution)

final_subv1.columns
final_subv1.columns = final_subv1.columns.astype(str)
final_subv1.rename(columns={'0':'m13'}, inplace=True)
final_subv1.columns

final_subv1 = final_subv1[['loan_id', 'm13']]
final_subv1.columns
final_subv1.to_csv('C:/Users/erragpa/Desktop/Loan Delinquency Prediction/ResultV7.csv', index=False)