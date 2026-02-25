import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class InvestigationAgent:

    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def generate_report(self, prediction_data: dict):

        try:
            prompt = f"""
    You are an enterprise financial risk investigation assistant.

    Transaction Risk Summary:
    - Fraud Probability: {prediction_data['fraud_probability']}
    - Risk Level: {prediction_data['risk_level']}

    Top Contributing Risk Factors:
    {prediction_data['top_risk_factors']}

    Generate a structured investigation report including:
    1. Risk Summary
    2. Key Contributing Factors Explanation
    3. Recommended Action
    4. Confidence Assessment
    """

            response = self.client.chat.completions.create(
                model=os.getenv("GROQ_MODEL"),
                messages=[
                    {"role": "system", "content": "You are a financial fraud risk analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=float(os.getenv("LLM_TEMPERATURE", 0.3)),
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"LLM report generation failed. Manual review recommended. Error: {str(e)}"