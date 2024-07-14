import streamlit as st
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

load_dotenv()

st.title("ATS-Focused Resume Matcher")

job_description = st.text_area("Enter the job description:")

def extract_ats_keywords(text):
    ats_keywords = [

    "python", "java", "c++", "javascript", "sql", "aws", "azure", "machine learning",
    "data analysis", "project management", "agile", "scrum", "leadership",
    "communication", "problem-solving", "teamwork", "innovation", "strategic thinking",
    "customer service", "sales", "marketing", "financial analysis", "budgeting",
    "research", "product development", "quality assurance", "ui/ux design",
    "database management", "network security", "cloud computing", "devops",
    "business intelligence", "data visualization", "statistical analysis",
    "risk management", "compliance", "blockchain", "artificial intelligence",
    "natural language processing", "big data", "data mining", "predictive modeling",
    "a/b testing", "seo", "content management", "digital marketing", "social media marketing",
    "email marketing", "crm", "erp", "supply chain management", "logistics",
    "inventory management", "six sigma", "lean manufacturing", "quality control",
    "iso standards", "cybersecurity", "network administration", "virtualization",
    "mobile development", "web development", "api development", "microservices",
    "containerization", "docker", "kubernetes", "ci/cd", "version control", "git",
    "data science", "deep learning", "tensorflow", "pytorch", "scikit-learn",
    "tableau", "power bi", "excel", "r programming", "scala", "hadoop", "spark",
    "nosql", "mongodb", "cassandra", "redis", "elasticsearch", "kafka",
    "rest api", "graphql", "oauth", "jwt", "ssl/tls", "agile methodologies",
    "kanban", "jira", "confluence", "trello", "asana", "slack",
    "technical writing", "public speaking", "negotiation", "conflict resolution",
    "time management", "critical thinking", "analytical skills", "attention to detail",
    "creativity", "adaptability", "flexibility", "resourcefulness",
    "data cleaning", "data wrangling", "exploratory data analysis",
    "data modeling", "data visualization", "statistical analysis",
    "regression analysis", "hypothesis testing", "data interpretation",
    "data-driven decision making",
    "large-scale data processing", "distributed systems", "parallel computing",
    "data scalability", "data governance", "data security", "data streaming",
    "batch and real-time analytics", "data partitioning", "data reliability",
    "ETL (Extract, Transform, Load)", "data pipeline", "data architecture",
    "data warehouse", "data integration", "distributed computing", "data lakes",
    "real-time data processing", "batch processing", "data infrastructure",
    "predictive analytics", "machine learning models", "feature engineering",
    "model evaluation", "model deployment", "algorithm development",
    "data mining techniques", "natural language understanding", "sentiment analysis",
    "anomaly detection", "business process modeling", "stakeholder management",
    "business strategy", "market research", "competitive analysis",
    "SWOT analysis", "business metrics", "business intelligence tools",
    "data storytelling", "report generation", "business performance analysis",
    "data reporting", "data governance", "ETL", 'etl'

    ]
    
    text = text.lower()
    found_keywords = [keyword for keyword in ats_keywords if keyword in text]
    
    return found_keywords

def get_additional_keywords(job_description, resume_text, n=25):
    texts = [job_description, resume_text]

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_df=0.8, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    feature_names = vectorizer.get_feature_names_out()
    
    diff = tfidf_matrix[0] - tfidf_matrix[1]
    
    sorted_indices = np.argsort(diff.toarray()[0])[::-1]
    
    top_words = []
    for i in sorted_indices:
        word = feature_names[i]
        if (len(word) > 3 and 
            not re.match(r'\b(strong|like|requirements|change|enhance|experience|working|tools|exceptional|ability|skills)\b', word) and
            len(top_words) < n):
            top_words.append(word)
    
    return top_words

if st.button("Find Best Matching Resume"):
    if job_description:
        
        job_ats_keywords = extract_ats_keywords(job_description)

        loader = DirectoryLoader('./Resume/', glob="**/*.pdf", loader_cls=PyPDFLoader)
        resume_documents = loader.load()

        all_matches = []
        for resume in resume_documents:
            resume_text = resume.page_content
            resume_ats_keywords = extract_ats_keywords(resume_text)
        
            matching_keywords = set(job_ats_keywords) & set(resume_ats_keywords)
            score = len(matching_keywords) / len(job_ats_keywords) if job_ats_keywords else 0
            
            all_matches.append((resume, matching_keywords, resume_ats_keywords, score, resume_text))

        all_matches.sort(key=lambda x: x[3], reverse=True)

        if all_matches:
            best_resume, matching_keywords, resume_keywords, best_score, best_resume_text = all_matches[0]
            
            st.write(f"Best Matching Resume: {os.path.basename(best_resume.metadata['source'])}")
            st.write(f"Match Percentage: {best_score * 100:.2f}%")

            st.write("\nMatched ATS Keywords:")
            st.write(", ".join(matching_keywords))

            missing_ats_keywords = set(job_ats_keywords) - set(resume_keywords)
            additional_keywords = get_additional_keywords(job_description, best_resume_text)
            
            st.write("\nSuggested Keywords to Add:")
            st.write(", ".join(set(additional_keywords)))  # Using set to remove duplicates

        else:
            st.write("No resumes found in the specified directory.")

    else:
        st.write("Please enter a job description.")
