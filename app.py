import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and paste the job description to get a match score.")

resume_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
job_desc = st.text_area("Paste Job Description here")

if st.button("Analyze Resume"):
    if resume_file and job_desc.strip():
        resume_text = extract_text(resume_file)

        docs = [resume_text, job_desc]
        tfidf = TfidfVectorizer(stop_words="english")
        vectors = tfidf.fit_transform(docs)

        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
        score = round(score, 2)

        st.success(f"✅ Resume Match Score: {score}%")

        if score < 40:
            st.error("❌ Very low match. Add more relevant skills.")
        elif score < 70:
            st.warning("⚠️ Medium match. Improve keywords.")
        else:
            st.success("🎉 Excellent match!")

    else:
        st.error("Please upload resume AND paste job description.")


