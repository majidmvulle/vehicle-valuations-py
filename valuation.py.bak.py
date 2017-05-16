__author__ = 'Majid Mvulle'

from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import json
import sys
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", type=str, required=True)
args = parser.parse_args()

with open(args.file) as json_file:
    full_dataset = json.dumps(json.load(json_file))

# clf1 = LogisticRegression(random_state=1)
# clf2 = RandomForestClassifier(random_state=1)
# clf3 = GaussianNB()

df = pd.read_json(full_dataset)
# df.drop('z_price', axis=1)
car = df.values
X, y = car[:, :-1], car[:, -1]
X, y = X.astype(int), y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

#
scaler = preprocessing.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#eclf5 = BaggingClassifier()
#eclf5.fit(X_train, y_train)
#prediction = eclf5.predict(X_test)
#print('{"price": '+str(prediction[0])+"}")


eclf6 = DecisionTreeClassifier()
eclf6.fit(X_train_scaled, y_train)
prediction = eclf6.predict(X_test_scaled)
print('{"price": '+str(prediction[0])+"}")

#lm = LinearRegression()
#lm.fit(X_train, y_train)
#prediction = lm.predict(X_test)
#print('{"price": '+str(prediction[0])+"}")

#rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_estimators=500, n_jobs=1, oob_score=False, random_state=None, verbose=0)
#rf.fit(X_train, y_train)
#prediction = rf.predict(X_test)
#print('{"price": '+str(prediction[0])+"}")
# print(clf.score(X_test_scaled, y_test))


#lr = LogisticRegression()
#lr.fit(X_train, y_train)
#prediction = lr.predict(X_test)
#print('{"price": '+str(prediction[0])+"}")