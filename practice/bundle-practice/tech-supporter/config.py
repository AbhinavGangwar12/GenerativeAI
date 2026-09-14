import os
from dotenv import load_dotenv

class Config:
    @staticmethod
    def get_api_key() -> str:
        """Securely load the OPENAI_API_KEY from the .env file."""
        # raise NotImplementedError
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Key does not exist!")
        return key