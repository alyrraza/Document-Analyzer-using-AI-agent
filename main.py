import streamlit as st
from text_from_image import TextFromImage
import google.generativeai as genai
import os

# Initialize API Key
API_KEY = os.getenv("GEMINI_API_KEY")

def call_gemini_chat(document_data, user_message):
    """Chat function that keeps conversation relevant to the document."""
    messages = [
        {
            "role": "system",
            "content": f"""
            You are an AI assistant specializing in document analysis and information retrieval.
            
            DOCUMENT CONTEXT:
            {document_data}
            
            GUIDELINES:
            1. Answer questions about information directly found in the document with precise details from the document.
            
            2. For terms, organizations, or concepts mentioned in the document (like a university name on an ID card):
            - Provide comprehensive information using both document data AND your broader knowledge
            - For example, if a user uploads a university ID card and asks "What is [University Name]?", provide detailed information about that university
            
            3. For questions completely unrelated to the document (like "What's the weather today?" or topics not mentioned in any way in the document):
            - Respond with: "I'm sorry, that question doesn't appear to be related to the document you've provided. I can answer questions about the document or entities mentioned in it."
            
            4. Keep responses clear, informative, and well-structured.
            """
        },
        {"role": "user", "content": user_message},
    ]
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([m["content"] for m in messages])
    return response.text

# Streamlit UI
st.title("AI Document Assistant")

uploaded_file = st.file_uploader("Upload a document image", type=["png", "jpg", "jpeg"])

document_data = None
if uploaded_file:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.read())
    
    extractor = TextFromImage(API_KEY)
    document_data = extractor.process("temp_image.jpg")
    
    if isinstance(document_data, dict) and "error" in document_data:
        st.error(document_data["error"])
        document_data = None
    else:
        st.success("Document processed successfully!")
        #st.write("Extracted Details:", document_data)

# Chat Section
if document_data:
    st.subheader("Ask Questions About the Document")
    user_input = st.text_input("Enter your question:")
    if st.button("Ask"):
        if user_input:
            response = call_gemini_chat(document_data, user_input)
            st.write("**Bot:**", response)
