import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

file = "Downloads\hospital\hospital_readmissions.csv"

df = pd.read_csv(file)

num_target = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency', 'change', 'diabetes_med', 'readmitted']

#df['change'] = df['change'].replace({'yes': 1, 'no': 0})
with pd.option_context("future.no_silent_downcasting", True):
    df['change'] = df['change'].replace('no', 0).infer_objects(copy=False)

with pd.option_context("future.no_silent_downcasting", True):
    df['change'] = df['change'].replace('yes', 1).infer_objects(copy=False)

#df['diabetes_med'] = df['diabetes_med'].replace({'yes': 1, 'no': 0})
with pd.option_context("future.no_silent_downcasting", True):
    df['diabetes_med'] = df['diabetes_med'].replace('no', 0).infer_objects(copy=False)

with pd.option_context("future.no_silent_downcasting", True):
    df['diabetes_med'] = df['diabetes_med'].replace('yes', 1).infer_objects(copy=False)   

#df['readmitted'] = df['readmitted'].replace({'yes': 1, 'no': 0})
with pd.option_context("future.no_silent_downcasting", True):
    df['readmitted'] = df['readmitted'].replace('no', 0).infer_objects(copy=False)

with pd.option_context("future.no_silent_downcasting", True):
    df['readmitted'] = df['readmitted'].replace('yes', 1).infer_objects(copy=False)


le = dict()
content = pd.DataFrame()

for feat in ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'glucose_test','A1Ctest']:
  le[feat] = LabelEncoder()
  content[feat] = le[feat].fit_transform(df[feat])

for feat in num_target:
  content[feat] = df[feat]


features = content.columns[:-1]
X = content[features]
y = content['readmitted']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
#save Xtest in a csv file

model = RandomForestClassifier(n_estimators = 100, min_samples_split = 5, max_depth= 11,  criterion = 'gini', random_state = 0)

model.fit(X_train, y_train)

#joblib.dump(model, 'model.pkl')

print(X_test)
print(X_test.dtypes)
'''
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Accuracy is {0:.2f}".format(accuracy_score(y_test, y_pred)))
print("Precision is {0:.2f}".format(precision_score(y_test, y_pred)))
print("Recall is {0:.2f}".format(recall_score(y_test, y_pred)))
'''