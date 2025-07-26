import os
from enum import Enum
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class ModelType(str, Enum):
    """Supported model types."""
    GEMINI = "gemini" 
class Config:
    """
    Centralized configuration class for Gemini model only.
    """

    
    MODEL_TYPE: ModelType = ModelType.GEMINI

    # Gemini configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-pro")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")

    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"

    @classmethod
    def set_gemini_key(cls, api_key: str):
        """
        Set Gemini API key.
        """
        cls.GEMINI_API_KEY = api_key
        os.environ["GEMINI_API_KEY"] = api_key

    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """
        Returns the Gemini configuration.
        """
        if not cls.GEMINI_API_KEY:
            raise ValueError("Gemini API key not set. Please set it via .env or set_gemini_key()")
        return {
            "type": "gemini",
            "model": cls.GEMINI_MODEL,
            "api_key": cls.GEMINI_API_KEY,
            "embedding_model": cls.GEMINI_EMBEDDING_MODEL
        }