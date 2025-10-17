import streamlit as st
from src.extract import extract_text_from_pdf
from src.analyzer import find_skills, calculate_score


st.title("📄 Resume Analyzer")
st.write("Upload a PDF resume to analyze")

uploaded_file = st.file_uploader("Choose PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text
    text = extract_text_from_pdf(uploaded_file)
    
    if text:
        # Find skills
        skills = find_skills(text)
        
        # Calculate score
        score = calculate_score(text, skills)
        
        st.subheader("📊 Resume Score")
        st.write(f"Score: {score}/100")

        st.subheader("💡 Skills Found")
        if skills:
            for skill in skills:
                st.write(f"✅ {skill}")
        else:
            st.write("No technical skills detected")

        st.subheader("📄 Resume Content")
        st.text_area("Full Text", text, height=300)
    else:
        st.error("Could not extract text from PDF")
