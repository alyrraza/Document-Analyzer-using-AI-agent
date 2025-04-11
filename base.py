# base.py
import google.generativeai as genai
import os
import logging
from abc import ABC, abstractmethod
from PIL import Image
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class BaseAI(ABC):
    def __init__(self, api_key: str = None):
        """Initialize the BaseAI class with an API key."""
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Gemini API key is required.")
        logging.info("Initializing BaseAI with API key.")
        genai.configure(api_key=self.api_key)
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """Abstract method to be implemented by child classes."""
        pass
    
    def call_gemini_api_with_image(self, image_bytes: bytes):
        """Extract text from an image using the Gemini API."""
        messages = [
    {
        "role": "system",
        "content": """
        You are an expert AI document analyzer and knowledge assistant. 
        Your job is to extract all details from the provided document image and organize them clearly.
        
        EXTRACTION GUIDELINES:
        1. Extract all visible text and data fields from the document
        2. Organize information by categories (Personal Info, Document Details, Institutional Info, etc.)
        3. Identify the document type (ID card, university card, admission card, etc.)
        
        RESPONSE STRUCTURE:
        1. Document Type: [Identify the document type]
        2. Key Information: [List all key fields and values]
        3. Additional Details: [Any other information visible]
        
        If the image is NOT a valid document, respond with:
        "Error: The provided image does not appear to be a valid document."
        """
    },
    {
        "role": "user", 
        "content": "Extract and organize all information from this document."
    }
]

        
        try:
            logging.info("Sending image to Gemini API for text extraction.")
            model = genai.GenerativeModel("gemini-1.5-flash")
            image = Image.open(io.BytesIO(image_bytes))
            
            prompt_text = "\n".join([msg["content"] for msg in messages])
            logging.info("Sending image data to Gemini API for text extraction.")
            
            response = model.generate_content([prompt_text, image])
            logging.info("Successfully received response from Gemini API.")
            return response.text  # Expected structured JSON output
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return {"error": f"Error processing image: {e}"}
