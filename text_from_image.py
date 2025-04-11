# text_from_image.py
from base import BaseAI
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TextFromImage(BaseAI):
    def __init__(self, api_key: str = None):
        """Initialize the text extraction class."""
        super().__init__(api_key)
        logging.info("Initialized TextFromImage class.")

    def process(self, image_path: str):
        """Extract text from a given document image."""
        
        if not os.path.exists(image_path):
            logging.error("Error: Image file not found.")
            return {"error": "Error: Image file not found."}
        
        try:
            logging.info("Opening image file for processing.")
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
            logging.info("Image successfully read. Sending for extraction.")
            return self.call_gemini_api_with_image(img_bytes)  # Using BaseAI method
        
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return {"error": f"Error processing image: {e}"}
