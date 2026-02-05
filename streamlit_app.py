import streamlit as st
import pdfplumber
import docx
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(layout="wide")
st.title("AI Resume Screener")
st.markdown("Keyword + TF-IDF + Cosine Similarity")

class ResumeScreener:
    def __init__(self):
        self.keywords = [
            'python', 'pytorch', 'tensorflow', 'scikit-learn', 'pandas', 'numpy',
            'langchain', 'transformers', 'huggingface', 'llm', 'rag', 'faiss',
            'chromadb', 'streamlit', 'docker', 'aws', 'sql', 'jupyter', 
            'mlops', 'fastapi', 'git', 'github', 'nlp'
        ]
    
    def read_pdf(self, file):
        try:
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text
        except:
            return "PDF read error"
    
    def read_docx(self, file):
        try:
            doc = docx.Document(file)
            text = " ".join([para.text for para in doc.paragraphs])
            return text
        except:
            return "DOCX read error"
    
    def read_txt(self, file):
        return file.read().decode('utf-8')
    
    def load_resume(self, file):
        file.seek(0)  # Reset file pointer
        
        if 'pdf' in file.type:
            return self.read_pdf(file)
        elif 'docx' in file.type or file.name.endswith('.docx'):
            return self.read_docx(file)
        elif 'txt' in file.type or file.name.endswith('.txt'):
            return self.read_txt(file)
        else:
            # Fallback: try as text
            return file.read().decode('utf-8', errors='ignore')
    
    def extract_skills(self, text):
        text_lower = text.lower()
        return [skill for skill in self.keywords 
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower)]
    
    def score_resume(self, job_desc, resume_text):
        job_skills = self.extract_skills(job_desc)
        resume_skills = self.extract_skills(resume_text)
        
        matched = set(job_skills) & set(resume_skills)
        keyword_score = min(len(matched) / max(len(job_skills), 1) * 40, 40)
        
        tfidf = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_matrix = tfidf.fit_transform([job_desc, resume_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        tfidf_score = similarity * 30
        
        total_score = int(keyword_score + tfidf_score + 30)
        return {
            'score': min(total_score, 100),
            'matched': list(matched),
            'missing': list(set(job_skills) - set(resume_skills)),
            'job_skills': job_skills,
            'similarity': round(similarity, 3),
            'preview': resume_text[:500]
        }

# Install required packages first
if 'screener' not in st.session_state:
    st.session_state.screener = ResumeScreener()

screener = st.session_state.screener

col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.header("Job Description")
    job_desc = st.text_area("", height=300)

with col2:
    st.header("Upload Resume")
    resume_file = st.file_uploader(
        "Supports PDF, DOCX, TXT", 
        type=['pdf', 'docx', 'txt', 'doc']
    )
    if resume_file:
        st.info(f"📄 File: {resume_file.name} ({resume_file.type})")

if st.button("Screen Resume", type="primary", use_container_width=True):
    if job_desc and resume_file:
        with st.spinner("Analyzing resume..."):
            resume_text = screener.load_resume(resume_file)
            results = screener.score_resume(job_desc, resume_text)
        
        st.header(f"Score: **{results['score']}/100**")
        
        col1, col2, col3 = st.columns(3)
        verdict = "✅ HIRE" if results['score'] >= 80 else "⚠️ MAYBE" if results['score'] >= 60 else "❌ REJECT"
        with col1:
            st.metric("Verdict", verdict)
        with col2:
            st.metric("Skills Match", f"{len(results['matched'])}/{len(results['job_skills'])}")
        with col3:
            st.metric("Similarity", f"{results['similarity']:.0%}")
        
        if results['matched']:
            st.subheader("✅ Matched Skills")
            for skill in results['matched']:
                st.success(f"• **{skill.upper()}**")
        
        if results['missing']:
            st.subheader("⚠️ Missing Skills")
            for skill in results['missing']:
                st.warning(f"• **{skill.upper()}**")
        
        st.subheader("📄 Resume Preview")
        st.text_area("", results['preview'], height=150)
        
        st.download_button("💾 Download Report", f"Score: {results['score']}", "report.txt")
    else:
        st.error("Upload resume and add job description")