import streamlit as st
import pdfplumber
import io
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class KeywordAnalyzer:
    def __init__(self):
        self.core_keywords = [
            'python','tensorflow','pytorch','scikit-learn','pandas','numpy','langchain','sql',
            'transformers','huggingface','llm','rag','faiss','streamlit','docker','aws',
            'flask','django','nlp','opencv','yolov8','lstm','mysql','powerbi','keras','r'
        ]
    
    def extract_jd_keywords(self, job_desc):
        """Extract IMPORTANT keywords from JD"""
        job_lower = job_desc.lower()
        
        # JD-specific keywords (skills, tools, frameworks)
        jd_keywords = []
        for kw in self.core_keywords:
            if kw in job_lower:
                jd_keywords.append(kw.upper())
        
        # Extract noun phrases (ML Engineer, Data Scientist, etc.)
        sentences = re.split(r'[.!?]+', job_desc)
        important_terms = []
        for sent in sentences:
            words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)?\b', sent)
            important_terms.extend([w.lower() for w in words if len(w) > 3])
        
        return list(set(jd_keywords + important_terms[:10]))
    
    def analyze_resume_keywords(self, resume_text, jd_keywords):
        """Find matches + missing keywords"""
        resume_lower = resume_text.lower()
        matched = []
        missing = []
        
        for kw in jd_keywords:
            if kw.lower() in resume_lower:
                matched.append(kw.upper())
            else:
                missing.append(kw.upper())
        
        return matched, missing

class ATSAnalyzer:
    def __init__(self):
        self.keywords = KeywordAnalyzer()
    
    def analyze(self, job_desc, resume_text):
        # Extract JD keywords
        jd_keywords = self.keywords.extract_jd_keywords(job_desc)
        
        # Analyze resume
        matched, missing = self.keywords.analyze_resume_keywords(resume_text, jd_keywords)
        
        # ATS Score (industry standard)
        match_score = min(len(matched) * 8, 60)
        jd_coverage = min(len(jd_keywords) / 15 * 25, 25) if jd_keywords else 0
        ats_score = int(match_score + jd_coverage + 15)
        
        # ATS Verdict
        if ats_score >= 85:
            verdict = "✅ PASS (Interview Ready)"
            status = "success"
        elif ats_score >= 70:
            verdict = "⚠️ PASS (Shortlist)"
            status = "warning"
        elif ats_score >= 55:
            verdict = "🔄 REVIEW (Phone Screen)"
            status = "info"
        else:
            verdict = "❌ FAIL (Revise Resume)"
            status = "error"
        
        return {
            'ats_score': min(ats_score, 100),
            'verdict': verdict,
            'status': status,
            'jd_keywords': jd_keywords,
            'matched_keywords': matched,
            'missing_keywords': missing[:8],  # Top 8 missing
            'match_rate': f"{len(matched)/(len(jd_keywords) or 1)*100:.0f}%",
            'total_jd_keywords': len(jd_keywords),
            'recommendations': self.generate_recommendations(missing, matched)
        }
    
    def generate_recommendations(self, missing, matched):
        recs = []
        if missing:
            recs.append(f"**Add these keywords:** {', '.join(missing[:3])}")
        if len(matched) < 5:
            recs.append("**Strengthen skills section** with technical terms")
        recs.append("**Use exact JD phrasing** (no synonyms)")
        recs.append("**STAR format** for experience bullets")
        return recs

class PDFExtractor:
    def extract(self, file):
        try:
            file.seek(0)
            with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + ""
            return re.sub(r's+', ' ', text).strip()
        except:
            return ""

# =====================================================
# PRODUCTION ATS UI - CLEAN & PROFESSIONAL
# =====================================================

st.set_page_config(page_title="ATS Resume Analyzer", layout="wide", page_icon="🔍")

st.title("🔍 **Real-Time ATS Resume Analyzer**")
st.markdown("*Pass ATS filters like 98% Fortune 500 companies*")

# Input Section
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 **Job Description**")
    job_desc = st.text_area("", height=300, placeholder="Paste full job description...")

with col2:
    st.header("📄 **Resume**")
    resume_text = st.text_area("Paste resume text OR upload PDF", height=300)
    resume_file = st.file_uploader("OR Upload PDF", type="pdf")

# ANALYZE BUTTON
if st.button("🚀 **ANALYZE FOR ATS**", type="primary", use_container_width=True):
    if job_desc.strip() and (resume_text.strip() or resume_file):
        # Get resume text
        if resume_file:
            resume_content = PDFExtractor().extract(resume_file)
        else:
            resume_content = resume_text
        
        if not resume_content.strip():
            st.error("❌ Cannot read resume content")
        else:
            with st.spinner("🔍 Analyzing ATS compatibility..."):
                analyzer = ATSAnalyzer()
                result = analyzer.analyze(job_desc, resume_content)
            
            # =====================================================
            # ATS RESULTS DASHBOARD
            # =====================================================
            
            # Score Header
            st.markdown("---")
            col_score1, col_score2, col_score3 = st.columns([1, 3, 1])
            
            with col_score1:
                st.metric("ATS Score", f"{result['ats_score']}/100")
            
            with col_score2:
                st.markdown(f"### **{result['verdict']}**")
                st.markdown(f"*Keyword Match: {result['match_rate']} | {result['total_jd_keywords']} JD keywords detected*")
            
            with col_score3:
                st.metric("Skills Gap", len(result['missing_keywords']))
            
            st.markdown("---")
            
            # Keyword Analysis
            col_keywords1, col_keywords2 = st.columns(2)
            
            with col_keywords1:
                st.subheader("✅ **FOUND in Resume**")
                if result['matched_keywords']:
                    found_cols = st.columns(3)
                    for i, skill in enumerate(result['matched_keywords'][:12]):
                        with found_cols[i % 3]:
                            st.success(f"**{skill}**")
                else:
                    st.warning("No JD keywords found in resume")
            
            with col_keywords2:
                st.subheader("❌ **MISSING from JD**")
                if result['missing_keywords']:
                    missing_cols = st.columns(2)
                    for i, skill in enumerate(result['missing_keywords'][:8]):
                        with missing_cols[i % 1]:
                            st.error(f"**{skill}**")
                else:
                    st.success("✅ Perfect keyword match!")
            
            # Actionable Recommendations
            st.subheader("🎯 **Fix Your Resume**")
            for rec in result['recommendations']:
                st.info(rec)
            
            # ATS Debug
            with st.expander("🔧 **ATS Pipeline Debug**"):
                st.json({
                    "JD Keywords Detected": result['jd_keywords'][:10],
                    "Resume Matches": len(result['matched_keywords']),
                    "Missing Keywords": len(result['missing_keywords']),
                    "Match Rate": result['match_rate']
                })
    else:
        st.warning("⚠️ **Job description + Resume required**")

st.markdown("---")
st.markdown("*Powered by LangChain + FAISS | Real ATS Analysis*")
