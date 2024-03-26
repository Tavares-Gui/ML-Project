import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

df = pd.read_csv('./datasets/titanic.csv')

le = LabelEncoder()

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

df['Embarked'] = le.fit_transform(df['Embarked'])
df['Sex'] = le.fit_transform(df['Sex'])

df['Age'] = df['Age'].astype(int)
df['Fare'] = df['Fare'].astype(int)

data = df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis =1, inplace=True)

Y = df['Survived']
X = df.drop('Survived', axis = 1)

Train = df.drop(['Survived'], axis=1)
Test = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(Train, Test, test_size = 0.2, random_state = 1)


# LRmodel = LogisticRegression(solver='liblinear', max_iter=200)
# LRmodel.fit(x_train, y_train)
# y_pred = LRmodel.predict(x_test)
# accuracy = accuracy_score(y_pred,y_test)
# precision = precision_score(y_pred,y_test)

# print('Logistic regression accuracy: {:.2f}%'.format(accuracy * 100))
# print('Logistic regression precision: {:.2f}%'.format(precision * 100))


# SVRmodel = GridSearchCV(SVR(),
#     {
#         'kernel': ['poly', 'rbf', 'sigmoid'],
#         'C' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 7, 8, 9, 10],
#         'tol': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5],
#         'epsilon': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     },
#     n_jobs = 8
# )

# SVRmodel.fit(x_train, y_train)
# print(SVRmodel.best_params_)

# model = SVRmodel.best_estimator_


scores = cross_val_score(ElasticNet(fit_intercept = True), X, Y, cv = 8)

ELmodel = GridSearchCV(ElasticNet(fit_intercept = True),
    {
        'alpha': list(map(lambda x: x / 10, range(1, 10))),
        'l1_ratio': list(map(lambda x: x / 10, range(1, 10))),
    },
    n_jobs = 4
)

model.fit(X_train, Y_train)

model = ELmodel.best_estimator_

dump(model, 'model.pkl')

print(mean_absolute_error(Y, model.predict(X)))

Ypred = model.predict(X)

plt.plot(Y)
plt.plot(Ypred)
plt.show()

wR = []
wP = []

Ymm = []
Ypmm = []

for i in range(len(Y)):
    wR.append(Y[i])
    wP.append(Ypred[i])
    
    if len(wR) > 15:
        Ymm.append(sum(wR) / 15)
        Ypmm.append(sum(wP) / 15)
        
        wR.pop(0)
        wP.pop(0)
            
plt.plot(Ymm)
plt.plot(Ypmm)
plt.show()
