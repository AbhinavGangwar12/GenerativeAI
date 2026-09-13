import os
from dotenv import load_dotenv

class Config:
    @staticmethod
    def get_api_key() -> str:
        """
        CORE REQUIREMENT: Securely load the OPENAI_API_KEY from the .env file.
        Raise a ValueError if missing.
        """
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        if key is None:
            raise ValueError("API key does not exist!")
        return key