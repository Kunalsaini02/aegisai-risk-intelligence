import os
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import json

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

from preprocessing import load_data, split_data, preprocess_data


DATA_PATH = "../../data/creditcard.csv"
MODEL_DIR = "../../artifacts"

os.makedirs(MODEL_DIR, exist_ok=True)


def train():

    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///../../mlflow.db")
    mlflow.set_experiment("AegisAI_Fraud_Detection")

    with mlflow.start_run():

        # -------------------------
        # Load & Preprocess
        # -------------------------
        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test = split_data(df)
        X_train, X_test, scaler = preprocess_data(X_train, X_test)

        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        # -------------------------
        # Logistic Regression
        # -------------------------
        lr_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )

        lr_model.fit(X_train, y_train)
        lr_probs = lr_model.predict_proba(X_test)[:, 1]

        lr_roc = roc_auc_score(y_test, lr_probs)
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
        lr_pr_auc = auc(lr_recall, lr_precision)

        mlflow.log_metric("lr_roc_auc", lr_roc)
        mlflow.log_metric("lr_pr_auc", lr_pr_auc)

        # -------------------------
        # XGBoost
        # -------------------------
        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )

        xgb_model.fit(X_train, y_train)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

        xgb_roc = roc_auc_score(y_test, xgb_probs)
        xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_probs)
        xgb_pr_auc = auc(xgb_recall, xgb_precision)

        mlflow.log_metric("xgb_roc_auc", xgb_roc)
        mlflow.log_metric("xgb_pr_auc", xgb_pr_auc)

        print("Logistic PR-AUC:", lr_pr_auc)
        print("XGBoost PR-AUC:", xgb_pr_auc)

        # -------------------------
        # Select Best Model
        # -------------------------
        if xgb_pr_auc > lr_pr_auc:
            best_model = xgb_model
            best_probs = xgb_probs
            best_name = "XGBoost"
            best_pr_auc = xgb_pr_auc
        else:
            best_model = lr_model
            best_probs = lr_probs
            best_name = "LogisticRegression"
            best_pr_auc = lr_pr_auc

        print("Selected Model:", best_name)
        mlflow.log_param("selected_model", best_name)

        # -------------------------
        # Cost-Sensitive Threshold Optimization
        # -------------------------

        fraud_cost = 5000
        false_positive_cost = 200

        best_threshold = 0.5
        min_loss = float("inf")

        for t in np.linspace(0.1, 0.95, 50):
            y_temp = (best_probs > t).astype(int)

            FP = np.sum((y_temp == 1) & (y_test == 0))
            FN = np.sum((y_temp == 0) & (y_test == 1))

            loss = FN * fraud_cost + FP * false_positive_cost

            if loss < min_loss:
                min_loss = loss
                best_threshold = t

        threshold = best_threshold
        y_pred = (best_probs > threshold).astype(int)

        mlflow.log_metric("business_loss", min_loss)
        mlflow.log_metric("optimized_threshold", threshold)

        print("Optimized Threshold:", threshold)
        print("Minimum Business Loss:", min_loss)

        roc = roc_auc_score(y_test, best_probs)

        precision, recall, _ = precision_recall_curve(y_test, best_probs)
        pr_auc = auc(recall, precision)

        report = classification_report(y_test, y_pred, output_dict=True)

        fraud_precision = report["1"]["precision"]
        fraud_recall = report["1"]["recall"]

        mlflow.log_metric("final_roc_auc", roc)
        mlflow.log_metric("final_pr_auc", pr_auc)
        mlflow.log_metric("fraud_precision", fraud_precision)
        mlflow.log_metric("fraud_recall", fraud_recall)
        mlflow.log_metric("threshold", threshold)

        # -------------------------
        # Save Best Model
        # -------------------------
        joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

        mlflow.log_artifact(os.path.join(MODEL_DIR, "model.pkl"))
        mlflow.log_artifact(os.path.join(MODEL_DIR, "scaler.pkl"))

        # Save feature columns
        feature_columns = list(X_train.columns)

        with open(os.path.join(MODEL_DIR, "features.json"), "w") as f:
            json.dump(feature_columns, f)

        # Save config
        config = {
            "threshold": threshold,
            "model_type": best_name,
            "version": "v2"
        }

        with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
            json.dump(config, f)

        mlflow.log_artifact(os.path.join(MODEL_DIR, "features.json"))
        mlflow.log_artifact(os.path.join(MODEL_DIR, "config.json"))

        print("Training complete.")
        print("Final PR-AUC:", best_pr_auc)
        print("Fraud Precision:", fraud_precision)
        print("Fraud Recall:", fraud_recall)


if __name__ == "__main__":
    train()