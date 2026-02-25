from inference import FraudPredictor
import pandas as pd

# Load dataset to simulate a real transaction
df = pd.read_csv("../../data/creditcard.csv")

sample = df.drop("Class", axis=1).iloc[0].to_dict()

predictor = FraudPredictor()

result = predictor.predict(sample)

print(result)