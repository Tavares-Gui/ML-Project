import pickle
import functools
from joblib import dump, load

from flask import (Blueprint, flash, g, redirect, render_template, request, session, url_for)

bp = Blueprint('auth', __name__, url_prefix='/')

@bp.route('/submit', methods=('POST',))
def submit():
    age = request.form.get('age')
    fare = request.form.get('fare')
    pClass = request.form.get('pClass')
    gender = request.form.get('gender')
    embarked = request.form.get('embarked')
    
    LRmodel = load('../models/LogisticRegression.pkl')
    ENmodel = load('../models/ElasticNet.pkl')
    
    if gender == 'male':
        gender = 1
    else:
        gender = 0
        
    if embarked == 'Q':
        embarked = 1
    elif embarked == 'S':
        embarked = 2
    else:
        embarked = 3
        
    age = int(age)
    
    data = [[pClass, gender, age, embarked, fare]]
    
    Ypred = model.predict(data)
    
    return render_template('result.html', prediction=Ypred[0])

@bp.route('/', methods=('GET', 'POST'))
def register():
    return render_template('index.html')
