## **🛡️ AegisAI – Enterprise Fraud Risk Intelligence Platform**

A production-grade fraud detection and investigation system combining machine learning, cost-sensitive optimization, and LLM-powered risk reporting.

**🚀 Features**

- Extreme class imbalance handling
- PR-AUC based model benchmarking
- Logistic Regression vs XGBoost comparison
- Automated model selection
- Cost-sensitive threshold optimization
- MLflow experiment tracking
- FastAPI production backend
- Groq-powered LLM investigation agent
- Next.js enterprise dashboard
- Full artifact versioning (model, scaler, config)

**🧠 Model Engineering Highlights**

- Stratified train-test split
- PR-AUC optimized for rare fraud detection
- Business-loss based threshold tuning
- Model versioning with config tracking
- Precision/Recall tradeoff analysis

**🏗️ Architecture**

- See docs/architecture.png

**🖥️ Tech Stack**

- **Backend:**

    - Python
    - FastAPI
    - XGBoost
    - MLflow
    - Scikit-learn

- **Frontend:**

    - Next.js 16
    - Tailwind CSS
    - TypeScript

- **AI:**

    - Groq LLM API

**📊 Business Optimization**

- Threshold optimized using:
    - Loss = FN × ₹5000 + FP × ₹200
- Minimizes expected financial risk.

**📦 How to Run**

- Backend:

     - uvicorn app.main:app --reload

- Frontend:

    - npm run dev

- MLflow:

    - mlflow ui

**🎯 Use Cases**

- Banking fraud detection
- FinTech transaction monitoring
- Risk intelligence dashboards
- Enterprise AI copilots