import os
# Hint: You will likely need the 'dotenv' package here (pip install python-dotenv)
from dotenv import load_dotenv

class Config:
    @staticmethod
    def get_llm_api_key() -> str:
        """
        ENVIRONMENT MANAGEMENT CHALLENGE:
        Securely load the API key from the .env file.
        
        Constraints:
        1. Read the OPENAI_API_KEY from the environment.
        2. If the key is missing or empty, raise a ValueError("API Key missing!").
        
        Returns:
            str: The loaded API key.
        """
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API KEY not found!")
        return api_key
        