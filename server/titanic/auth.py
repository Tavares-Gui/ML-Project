import functools
import numpy as np
import pandas as pd

from joblib import dump, load
from flask import (Flask, Blueprint, flash, g, redirect, render_template, request, session, url_for)

bp = Blueprint('auth', __name__, url_prefix='/')

@bp.route('/', methods=['GET']) 
def index(): 
    return render_template('index.html') 

@bp.route('/submit', methods=['POST'])
def submit():
    Pclass = request.form.get('pClass')
    Sex = request.form.get('gender')
    Age = request.form.get('age')
    SibSp = request.form.get('sibSp')
    Parch = request.form.get('parch')
    Fare = request.form.get('fare')
    Embarked = request.form.get('embarked')
    
    Pclass = int(Pclass)

    Age = float(Age)
    
    SibSp = int(SibSp)
    
    Parch = int(Parch)
    
    Fare = float(Fare)
    
    data = [[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]]

    LRmodel = load('../models/LogisticRegression.pkl')
    ENmodel = load('../models/ElasticNet.pkl')

    df = pd.DataFrame(data)

    predict = LRmodel.predict(df)

    print(predict)

    pred = str(predict[0])

    print(pred)

    return pred

if __name__ == '__main__': 
    app.run()
