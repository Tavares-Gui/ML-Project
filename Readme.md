primeiro teste feito com Logistic Regression

Foi escolhido a Logistic Regression pelo problema ser de 0 (morreu) e 1 (sobreviveu) para a predição, sendo um modelo mais facil para tal caso, sendo também otimo em performance

df['Embarked'] = df['Embarked'].map( {'Q': 0,'S':1,'C':2}).astype(int)
df['Sex'] = df['Sex'].map( {'female': 1,'male':0}).astype(int)

estas linhas estavam sendo usadas para transformar os tipos que estavam em string para numeros, a acuracia com estas linhas era de 92%

apos pesquisas, as duas linhas linhas foram substituidas pelas seguintes:

df['Embarked'] = le.fit_transform(df['Embarked'])
df['Sex'] = le.fit_transform(df['Sex'])

apos a alteração a acuracia passou de 92% para 98%

ao alteral uma linha que usava um "iloc" para uma coluna do df, a acuracia foi para 100%

tentei fazer o DTR e tambem o SVC e os dois me retornaram resultados muito ruins

save:
!pip install numpy
!pip install pandas
!pip install joblib
!pip install matplotlib
!pip install scikit-learn

https://www.kaggle.com/datasets/ashishkumarjayswal/titanic-datasets/data












estou programando em python usando flask


```
import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('auth', __name__, url_prefix='/auth')

@bp.route('/', methods=('GET', 'POST'))
def register():
    return render_template('index.html')

```

tenho este auth


```
import os

from flask import Flask

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/')
    def hello():
        return 'Hello, World!'
    
    from . import auth
    app.register_blueprint(auth.bp)

    return app

```

este init


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

                        <input type="radio" id="class1"/>
                            <label for="class1">1</label>

                        <input type="radio" id="class2"/>
                            <label for="class2">2</label>

                        <input type="radio" id="class3"/>
                            <label for="class3">3</label>
                    </fieldset>

                    <fieldset>
                        <legend>Passenger gender</legend>

                        <input type="radio" id="male"/>
                            <label for="male">Male</label>

                        <input type="radio" id="female"/>
                            <label for="female">Female</label>
                    </fieldset>

                    <fieldset>
                        <legend>Passenger age</legend>

                        <input type="number" id="age">
                    </fieldset>

                    <fieldset>
                        <legend>Passenger embarked</legend>

                        <input type="radio" id="q"/>
                            <label for="q">Q</label>

                        <input type="radio" id="s"/>
                            <label for="s">S</label>

                        <input type="radio" id="c"/>
                            <label for="c">C</label>
                    </fieldset>

                    <fieldset>
                        <legend>Passenger fare</legend>

                        <input type="text" id="fare">
                    </fieldset>

                    <br>

                    <button>Submit</button>
                </form>
            </div>
        </div>
    </body>
</html>

```

este html





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

e este modelo



preciso fazer com que os valores dos meus inputs no html seja passados para que eu consiga testalos usando o meu modelo, mas nao sei fazer isso
