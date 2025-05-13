import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def ejecutar_naive_bayes():
    url = 'https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv'
    cancer = pd.read_csv(url)
    le = LabelEncoder()
    cancer['diagnosis'] = le.fit_transform(cancer['diagnosis'])
    X = cancer.iloc[:, 0:8]
    y = cancer.iloc[:, 8]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=22
    )
    model = GaussianNB()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy