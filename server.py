from flask import Flask

app = Flask(__name__)

@app.route('/')
def WriteData():
    ArrInfos = []
    
    PClass = input("Passenger class < 1 > < 2 > < 3 >:\n\t> ")
    ArrInfos.append(PClass)
    Sex = input("\n\nPassenger gender < male > < female >:\n\t> ")
    ArrInfos.append(Sex)
    Age = input("\n\nPassenger age:\n\t> ")
    ArrInfos.append(Age)
    Embarked = input("\n\nPassenger embarked: < Q > < S > < C >\n\t> ")
    ArrInfos.append(Embarked)
    Fare = input("\n\nPassenger fare:\n\t> ")
    ArrInfos.append(Fare)
    
    return ArrInfos



if __name__ == '__main__':
    app.run(debug=True)
