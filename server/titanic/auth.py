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
    # Pclass = request.form.get('pClass')
    # Sex = request.form.get('gender')
    # Age = request.form.get('age')
    # SibSp = request.form.get('sibSp')
    # Parch = request.form.get('parch')
    # Fare = request.form.get('fare')
    # Embarked = request.form.get('embarked')
    
    # if Sex == 'male':
    #     Sex = 0
    # if Sex == 'female':
    #     Sex = 1
    
    # if Embarked == 'Q':
    #     Embarked = 0
    # if Embarked == 'S':
    #     Embarked = 1
    # if Embarked == 'C':
    #     Embarked = 2
    
    data = request.form

    personInfos = {
        "Pclass": [data["pClass"]],
        "Sex": {'male': 0, 'female': 1 }[data["gender"]],
        "Age": [data["age"]],
        "SibSp": [data["sibSp"]],
        "Parch": [data["parch"]],
        "Fare": [data["fare"]],
        "Embarked": {'Q': 0, 'S': 1, 'C': 2}[data["embarked"]]
    }
    
    data = pd.DataFrame(personInfos)
        
    LRmodel = load('../models/LogisticRegression.pkl')
    ENmodel = load('../models/ElasticNet.pkl')

    predict = LRmodel.predict(data)

    print(predict)

    pred = str(predict[0])

    print(pred)

    return pred

if __name__ == '__main__': 
    app.run()
