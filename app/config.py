import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Gemini configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
    GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
    
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

    @classmethod
    def set_gemini_key(cls, api_key: str):
        """Set Gemini API key at runtime."""
        cls.GEMINI_API_KEY = api_key
        os.environ["GEMINI_API_KEY"] = api_key

    @classmethod
    def get_llm_config(cls) -> dict:
        """Get Gemini configuration."""
        if not cls.GEMINI_API_KEY:
            raise ValueError("Gemini API key not set. Please set it via .env or set_gemini_key()")
        return {
            "type": "gemini",
            "model": cls.GEMINI_MODEL,
            "api_key": cls.GEMINI_API_KEY,
            "embedding_model": cls.GEMINI_EMBEDDING_MODEL
        }