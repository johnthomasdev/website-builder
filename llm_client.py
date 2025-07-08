import os
import traceback
import google.generativeai as genai
from dotenv import load_dotenv

class ModelNotLoadedError(Exception):
    pass

class LLMClient:
    def __init__(self):
        self.model = None
        
    async def initialize(self):
        """Initialize the Google Generative AI client with the Gemini model."""
        try:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            
            genai.configure(api_key=api_key) # type: ignore
            
            # For demonstration, using a specific model. You can make this configurable.
            self.model = genai.GenerativeModel( # type: ignore
                'gemini-2.0-flash',
                 system_instruction="You are a world-class web developer and AI assistant. Your task is to generate or modify HTML, CSS, and JavaScript code based on user requests. Follow all instructions precisely. Return only the raw code for the requested file type, without any markdown formatting like ```html or ```."
            )
            print("Gemini 2.0 Flash model initialized successfully.")
        except Exception as e:
            print("="*50)
            print("!!! FAILED TO INITIALIZE GEMINI MODEL !!!")
            print(f"Underlying error: {e}")
            print("="*50)
            print("This might be due to:")
            print("1. An invalid or missing GOOGLE_API_KEY in your .env file.")
            print("2. Network issues preventing connection to Google's servers.")
            print("\nThe application will continue but will fail on any generation task.")
            traceback.print_exc()
            self.model = None
    
    async def generate(self, prompt: str, max_tokens: int = 8192) -> str:
        """Generate a response from the Gemini model."""
        if not self.model:
            raise ModelNotLoadedError("Gemini model is not loaded. Cannot generate text.")
        
        try:
            # The system instruction is part of the model, so we just send the prompt.
            response = await self.model.generate_content_async(prompt)
            
            generated_text = response.text
            return generated_text.strip()
                
        except Exception as e:
            print(f"LLM Generation Error: {e}")
            raise Exception(f"Failed to generate response: {e}")
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None