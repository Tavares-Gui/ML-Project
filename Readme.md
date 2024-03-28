primeiro teste feito com Logistic Regression

Foi escolhido a Logistic Regression pelo problema ser de 0 (morreu) e 1 (sobreviveu) para a predição, sendo um modelo mais facil para tal caso, sendo também otimo em performance

tenho este modelo de Machine learning, para prever se algm morre no titanic


```
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

scores = cross_val_score(ElasticNet(fit_intercept = True), X, Y, cv = 8)

ELmodel = GridSearchCV(ElasticNet(fit_intercept = True),
    {
        'alpha' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 7, 8, 9, 10],
        'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3, 4, 5],
        'fit_intercept': [True, False],
    },
    n_jobs = 4
)

ELmodel.fit(x_train, y_train)
print(ELmodel.best_params_)

model = ELmodel.best_estimator_

dump(model, 'model.pkl')

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
```

tenho o seguinte arquivo __init__ para o python, estou usando flask

 
```
import os

from flask import Flask

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    from . import auth
    app.register_blueprint(auth.bp)

    return app

```

o seguinte auth também em python para o flask




```
import functools

# from . import create_app
from flask import (Blueprint, flash, g, redirect, render_template, request, session, url_for)

# app = create_app()

bp = Blueprint('auth', __name__, url_prefix='/')

@bp.route('/submit', methods=('POST',))
def submit():
    age = request.form.get('age')
    fare = request.form.get('fare')
    pClass = request.form.get('pClass')
    gender = request.form.get('gender')
    embarked = request.form.get('embarked')
    
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

```

e as seguintes paginas html, index e result respectivamente


```
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Titanic</title>
        <style>
        </style>
    </head>

    <body>
        <h1 style="display: flex; background-color: lightBlue; justify-content: center;">FIND OUT IF YOU WOULD DIE ON THE TITANIC</h1>
        <div style="display: flex; justify-content: center;">
            <div style="display: flex; justify-content: center; border-style: ridge; border-width: 5px; border-color: black; width: 275px; height: 325px;">
                <form>
                    <fieldset>
                        <legend>Passenger class</legend>

                        <input name="pClass" type="radio" id="class1"/>
                            <label for="class1">1</label>

                        <input name="pClass" type="radio" id="class2"/>
                            <label for="class2">2</label>

                        <input name="pClass" type="radio" id="class3"/>
                            <label for="class3">3</label>
                    </fieldset>

                    <fieldset>
                        <legend>Passenger gender</legend>

                        <input name="gender" type="radio" id="male"/>
                            <label for="male">Male</label>

                        <input name="gender" type="radio" id="female"/>
                            <label for="female">Female</label>
                    </fieldset>

                    <fieldset>
                        <legend>Passenger age</legend>

                        <input name="age" type="number" id="age">
                    </fieldset>

                    <fieldset>
                        <legend>Passenger embarked</legend>

                        <input name="embarked" type="radio" id="q"/>
                            <label for="q">Q</label>

                        <input name="embarked" type="radio" id="s"/>
                            <label for="s">S</label>

                        <input name="embarked" type="radio" id="c"/>
                            <label for="c">C</label>
                    </fieldset>

                    <fieldset>
                        <legend>Passenger fare</legend>

                        <input name="fare" type="text" id="fare">
                    </fieldset>

                    <br>

                    <button href="{{ url_for('auth.submit') }}">Submit</button>
                </form>
            </div>
        </div>
    </body>
</html>

```


```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Titanic</title>
</head>
<body>
    <h1>{{ prediction }}</h1>
    <a href="{{ url_for('auth.index') }}">Voltar para o form</a>
</body>
</html>
```

preciso fazer com que ao enviar o formulario do index, a resposta do formulario seja utilizada no modelo feito em python, para que eu consiga saber se a pessoas iria sobreviver ou morre, o que falta para que isso aconteca
