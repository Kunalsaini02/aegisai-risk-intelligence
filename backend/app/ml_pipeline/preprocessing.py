import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str):
    df = pd.read_csv(path)
    return df


def split_data(df: pd.DataFrame):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    # Log transform Amount
    X_train["Amount"] = np.log1p(X_train["Amount"])
    X_test["Amount"] = np.log1p(X_test["Amount"])

    # Scale Amount
    scaler = StandardScaler()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])

    return X_train, X_test, scaler