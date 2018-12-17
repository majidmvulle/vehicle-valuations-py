__author__ = 'Majid Mvulle'

from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import pandas as pd
import json
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeRegressor
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", type=str, required=True)
args = parser.parse_args()

with open(args.file) as json_file:
    full_dataset = json.dumps(json.load(json_file))

# clf1 = LogisticRegression(random_state=1)
# clf2 = RandomForestClassifier(random_state=1)
# clf3 = GaussianNB()


def mapper(f, c):
    return "(%0.2f %s)" % (c, f)

df = pd.read_json(full_dataset)

# training_no_price = training.drop(['z_price'], 1)

# dv = DictVectorizer()
# dv.fit(training.T.to_dict().values())


# lr = LinearRegression().fit(dv.transform(training_no_price.T.to_dict().values()), training.z_price)
# print(' + '.join([format(lr.intercept_, '0.2f')] + mapper(dv.feature_names_, lr.coef_)))

# print(format(lr.intercept_, '0.2f'))
# print('{"price": '+str(format(lr.intercept_, '0.2f'))+"}")



# df.drop('z_price', axis=1)
car = df.values
X, y = car[:, :], car[:, -1]
X, y = X.astype(int), y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

#
# scaler = preprocessing.MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)

# eclf5 = BaggingRegressor()
# eclf5.fit(X_train, y_train)
# prediction = eclf5.predict(X_test)
# print('{"price": '+str(prediction[0])+"}")


# eclf6 = DecisionTreeRegressor(criterion="mae")
# eclf6.fit(X_train, y_train)
# prediction = eclf6.predict(X_test)
# print('{"price": '+str(prediction[0])+"}")


# lm = LinearRegression(normalize=True)
# lm.fit(X_train, y_train)
# prediction = lm.predict(X_test)
# print('{"price": '+str(prediction[0])+"}")


# rf = RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0)
# rf.fit(X_train, y_train)
# prediction = rf.predict(X_test)
# print('{"price": '+str(prediction[0])+"}")
# print(clf.score(X_test_scaled, y_test))


lr = LogisticRegression(solver='lbfgs', multi_class='auto')
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)
print('{"price": '+str(prediction[0])+"}")