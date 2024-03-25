df['Embarked'] = df['Embarked'].map( {'Q': 0,'S':1,'C':2}).astype(int)
df['Sex'] = df['Sex'].map( {'female': 1,'male':0}).astype(int)

estas linhas estavam sendo usadas para transformar os tipos que estavam em string para numeros, a acuracia com estas linhas era de 92%

apos pesquisas, as duas linhas linhas foram substituidas pelas seguintes:

df['Embarked'] = le.fit_transform(df['Embarked'])
df['Sex'] = le.fit_transform(df['Sex'])

apos a alteração a acuracia passou de 92% para 98%
