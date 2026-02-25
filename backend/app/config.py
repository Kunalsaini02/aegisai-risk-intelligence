import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME = "AegisAI Risk Intelligence Platform"
    VERSION = "1.0.0"
    DATABASE_URL = "sqlite:///./aegisai.db"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

settings = Settings()