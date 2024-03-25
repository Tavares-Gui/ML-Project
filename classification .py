import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

df = pd.read_csv('/kaggle/input/titanic-datasets/titanic.csv')

le = LabelEncoder()

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

df['Embarked'] = le.fit_transform(df['Embarked'])
df['Sex'] = le.fit_transform(df['Sex'])

df['Age'] = df['Age'].astype(int)
df['Fare'] = df['Fare'].astype(int)

data = df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis =1, inplace=True)

Train = df.drop(['Survived'], axis=1)
Test = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(Train, Test, test_size = 0.2, random_state = 1)

# LR = LogisticRegression(solver='liblinear', max_iter=200)
# LR.fit(x_train, y_train)

# y_pred = LR.predict(x_test)

# accuracy = accuracy_score(y_pred, y_test)
# precision = precision_score(y_pred, y_test)

SVR = GridSearchCV(SVR(),
    {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': list(map(lambda x: x, range(1, 10))),
        'degree': list(map(lambda x: x, range(1, 5)))
    },
    n_jobs = 8
)

SVR.fit(x_train, y_train)
print(SVR.best_params_)

print('Logistic regression accuracy: {:.2f}%'.format(accuracy * 100))
print('Logistic regression precision: {:.2f}%'.format(precision * 100))

LR = LogisticRegression(solver='liblinear', max_iter=200)
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
LRAcc = accuracy_score(y_pred,y_test)
print('Logistic regression accuracy: {:.2f}%'.format(LRAcc*100))
