import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def entrenar_modelo():
    url = 'https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv'
    cancer = pd.read_csv(url)

    le = LabelEncoder()
    cancer['diagnosis'] = le.fit_transform(cancer['diagnosis'])

    # Eliminamos 'id' y 'diagnosis'
    X = cancer.drop(columns=['id', 'diagnosis'])
    y = cancer['diagnosis']

    print("Columnas originales:", X.columns.tolist())
    print("Forma original:", X.shape)

    imputer = SimpleImputer(strategy='mean')
    X_imputed_array = imputer.fit_transform(X)

    print("Forma después de imputar:", X_imputed_array.shape)

    if X_imputed_array.shape[1] == X.shape[1]:
        X = pd.DataFrame(X_imputed_array, columns=X.columns)
    else:
        X = pd.DataFrame(X_imputed_array)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=22
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return model, le, X.columns.tolist(), accuracy
