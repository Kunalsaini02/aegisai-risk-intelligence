import os
import json
import joblib
import numpy as np
import pandas as pd
import shap


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../artifacts"))


class FraudPredictor:

    def __init__(self):
        self.model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        self.explainer = shap.TreeExplainer(self.model)

        with open(os.path.join(MODEL_DIR, "features.json"), "r") as f:
            self.features = json.load(f)

        with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
            config = json.load(f)

        self.threshold = config["threshold"]

    def preprocess(self, input_data: dict):

        df = pd.DataFrame([input_data])

        # Ensure correct feature order
        df = df[self.features]

        # Log transform Amount
        df["Amount"] = np.log1p(df["Amount"])

        # Scale Amount
        df["Amount"] = self.scaler.transform(df[["Amount"]])

        return df

    def predict(self, input_data: dict):

        processed = self.preprocess(input_data)

        prob = self.model.predict_proba(processed)[0][1]
        label = int(prob > self.threshold)
        if prob < 0.2:
            risk_level = "Low"
        elif prob < 0.5:
            risk_level = "Medium"
        elif prob < 0.8:
            risk_level = "High"
        else:
            risk_level = "Critical"

        # SHAP values
        shap_values = self.explainer.shap_values(processed)

        feature_contributions = dict(
            zip(self.features, shap_values[0])
        )

        # Sort by absolute impact
        top_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        explanation = [
            {
                "feature": feature,
                "impact": float(value)
            }
            for feature, value in top_features
        ]

        return {
            "fraud_probability": float(prob),
            "prediction": label,
            "risk_level": risk_level,
            "top_risk_factors": explanation
        }