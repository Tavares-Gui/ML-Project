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

save:
!pip install numpy
!pip install pandas
!pip install joblib
!pip install matplotlib
!pip install scikit-learn

https://www.kaggle.com/datasets/ashishkumarjayswal/titanic-datasets/data
