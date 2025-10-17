import pdfplumber
import streamlit as st

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text
