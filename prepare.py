import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def prepare_iris():
    # Chargement
    iris_df = pd.read_csv("data/iris.csv")
    # Séparation train - test
    y = iris_df['class']
    X = iris_df.drop(labels='class', axis=1)
    return train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

def prepare_titanic():
    # Chargement
    titanic_df = pd.read_csv("data/titanic_train.csv")
    # Création des nouvelles colonnes
    titanic_df['FamilyNb'] = titanic_df['SibSp'] + titanic_df['Parch']
    titanic_df['Alone'] = (titanic_df['FamilyNb'] == 0)
    # Quantification Sex + Embarked
    sex_df = pd.get_dummies(titanic_df['Sex'], prefix='sex', drop_first=True)
    embarked_df = pd.get_dummies(titanic_df['Embarked'], prefix='embarked', dummy_na=True)
    titanic_df = pd.concat([titanic_df, embarked_df, sex_df], axis=1)
    # Suppression des colonnes non utilisées
    titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Sex', 'Embarked'], axis=1, inplace=True)
    # Séparation train-test
    y = titanic_df['Survived']
    X = titanic_df.drop(['Survived'], axis=1)
    train_X_titanic, test_X_titanic, train_y_titanic, test_y_titanic = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    col_names = train_X_titanic.columns
    # Imputation des données manquants
    titanic_imputer = SimpleImputer(strategy='mean')
    titanic_imputer.fit(train_X_titanic)
    train_X_titanic = titanic_imputer.transform(train_X_titanic)
    test_X_titanic = titanic_imputer.transform(test_X_titanic)
    # /!\ : En raison du passage par Scikit, les datasets sont maintenant des numpy.array et non des DataFrame Pandas ! 
    # On les retransforme pour garder les noms des colonnes
    train_X_titanic = pd.DataFrame(data=train_X_titanic, columns=col_names)
    test_X_titanic = pd.DataFrame(data=test_X_titanic, columns=col_names)
    return train_X_titanic, test_X_titanic, train_y_titanic, test_y_titanic

def prepare_boston():
    # Chargement
    names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    boston_df = pd.read_fwf("data/boston.txt", skiprows=22, header=None, names=names)
    # Séparation train - test
    y = boston_df['MEDV']
    X = boston_df.drop(labels='MEDV', axis=1)
    return train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)