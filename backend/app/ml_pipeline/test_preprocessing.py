from preprocessing import load_data, split_data, preprocess_data

# Adjust path to your dataset
DATA_PATH = "../../data/creditcard.csv"

df = load_data(DATA_PATH)

X_train, X_test, y_train, y_test = split_data(df)

X_train, X_test, scaler = preprocess_data(X_train, X_test)

print("Preprocessing Successful")
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)