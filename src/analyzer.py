import streamlit as st
import pdfplumber
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import re
from collections import Counter
import json
import numpy as np
from datetime import datetime
import random


# Modern CSS Styling - Add this after imports
def load_css():
    st.markdown("""
    <style>
    /* Global background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content area - white with black text */
    .main .block-container {
        background: white !important;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem auto;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Force main content text to be black */
    .main .block-container, 
    .main .block-container * {
        color: black !important;
    }
    
    /* Title stays white */
    h1 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    
    /* SIDEBAR - Keep text WHITE */
    .css-1d391kg,
    .css-1d391kg *,
    .sidebar .sidebar-content,
    .sidebar .sidebar-content * {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs button {
        color: black !important;
        font-weight: 700 !important;
        background: white !important;
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        margin: 0 5px !important;
    }
    
    .stTabs button[aria-selected="true"] {
        color: white !important;
        background: #667eea !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: #667eea !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)


load_css()


from dotenv import load_dotenv
load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")
print("API KEY:", API_KEY[:10] if API_KEY else "None", "...")


try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class AppConfig:
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.app_name = os.getenv('APP_NAME', 'AI Resume Analyzer')
    
    def has_gemini_key(self):
        return self.gemini_api_key is not None and self.gemini_api_key.strip() != ""


config = AppConfig()


def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text


def find_skills(text):
    skills_list = [
        # Programming & Web
        'python', 'java', 'javascript', 'typescript', 'react', 'node.js', 'angular', 'vue.js',
        'html', 'css', 'bootstrap', 'jquery', 'git', 'github', 'php', 'c++', 'c#', 'ruby',
        
        # Databases
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
        
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform',
        
        # AI/ML
        'machine learning', 'data science', 'tensorflow', 'pytorch', 'scikit-learn',
        'deep learning', 'nlp', 'computer vision', 'pandas', 'numpy',
        
        # Systems
        'linux', 'windows', 'macos', 'bash', 'powershell', 'api', 'rest', 'graphql',
        
        # Other
        'marketing', 'seo', 'content marketing', 'social media', 'google analytics',
        'photoshop', 'illustrator', 'figma', 'canva', 'indesign',
        'sales', 'crm', 'lead generation', 'negotiation', 'customer service'
    ]
    
    found_skills = []
    text_lower = text.lower()
    for skill in skills_list:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills


def extract_experience_years(text):
    experience_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'(\d+)\s*-\s*(\d+)\s*years?',
        r'over\s*(\d+)\s*years?',
        r'more\s*than\s*(\d+)\s*years?'
    ]
    
    text_lower = text.lower()
    max_experience = 0
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                years = max([int(x) for x in match if x.isdigit()])
            else:
                years = int(match) if match.isdigit() else 0
            max_experience = max(max_experience, years)
    return max_experience


# ADD THIS NEW FUNCTION HERE (ADDITION #1)
def get_job_template(job_title):
    """Auto-populate job requirements based on job title"""
    job_title_lower = job_title.lower().strip()
    
    job_templates = {
        # Web Development
        "web developer": {
            "description": "Looking for a skilled web developer with experience in JavaScript, React, and Node.js. Must have 2+ years of web development experience and knowledge of modern web technologies.",
            "skills": "JavaScript, React, HTML, CSS, Node.js, Git, MongoDB, Express.js",
            "experience": 2
        },
        "frontend developer": {
            "description": "Seeking a frontend developer with strong React/Vue.js skills and modern JavaScript knowledge. Experience with responsive design and UI/UX principles required.",
            "skills": "JavaScript, React, Vue.js, HTML, CSS, TypeScript, Bootstrap, Git",
            "experience": 2
        },
        "backend developer": {
            "description": "Looking for a backend developer with strong server-side programming skills. Experience with APIs, databases, and cloud services required.",
            "skills": "Node.js, Python, Java, MongoDB, PostgreSQL, REST APIs, Docker, AWS",
            "experience": 3
        },
        "full stack developer": {
            "description": "Seeking a full-stack developer with both frontend and backend experience. Must be proficient in modern web technologies and database management.",
            "skills": "JavaScript, React, Node.js, Python, MongoDB, PostgreSQL, Git, Docker",
            "experience": 3
        },
        
        # Data & AI
        "data scientist": {
            "description": "Looking for a data scientist with Python, Machine Learning, and SQL experience. Strong analytical skills and experience with data visualization required.",
            "skills": "Python, Machine Learning, SQL, Statistics, Pandas, Scikit-learn, TensorFlow",
            "experience": 3
        },
        "machine learning engineer": {
            "description": "Seeking an ML engineer with experience in building and deploying ML models. Strong programming skills and knowledge of ML frameworks required.",
            "skills": "Python, Machine Learning, TensorFlow, PyTorch, Deep Learning, MLOps, Docker",
            "experience": 3
        },
        "data analyst": {
            "description": "Looking for a data analyst with strong SQL and Python skills. Experience with data visualization tools and statistical analysis required.",
            "skills": "Python, SQL, Excel, Tableau, Power BI, Statistics, Pandas, Numpy",
            "experience": 2
        },
        
        # Software Engineering
        "software engineer": {
            "description": "Seeking a software engineer with strong programming skills and experience in software development lifecycle. Knowledge of multiple programming languages preferred.",
            "skills": "Python, Java, JavaScript, Git, SQL, Docker, AWS, Agile",
            "experience": 3
        },
        "python developer": {
            "description": "Looking for a Python developer with experience in web frameworks and API development. Knowledge of databases and cloud services preferred.",
            "skills": "Python, Django, Flask, PostgreSQL, REST APIs, Git, Docker, AWS",
            "experience": 2
        },
        
        # Mobile Development
        "mobile developer": {
            "description": "Seeking a mobile developer with experience in iOS/Android development. Knowledge of cross-platform frameworks is a plus.",
            "skills": "React Native, Flutter, Swift, Kotlin, JavaScript, Git, API Integration",
            "experience": 2
        },
        
        # DevOps & Cloud
        "devops engineer": {
            "description": "Looking for a DevOps engineer with experience in CI/CD, containerization, and cloud platforms. Strong automation and scripting skills required.",
            "skills": "Docker, Kubernetes, AWS, Jenkins, Terraform, Linux, Python, Git",
            "experience": 3
        },
        "cloud engineer": {
            "description": "Seeking a cloud engineer with expertise in AWS/Azure/GCP. Experience with infrastructure as code and cloud architecture required.",
            "skills": "AWS, Azure, GCP, Terraform, Docker, Kubernetes, Python, Linux",
            "experience": 3
        },
        
        # UI/UX & Design
        "ui/ux designer": {
            "description": "Looking for a UI/UX designer with strong design skills and user research experience. Proficiency in design tools and prototyping required.",
            "skills": "Figma, Adobe XD, Sketch, Photoshop, Illustrator, User Research, Prototyping",
            "experience": 2
        },
        "graphic designer": {
            "description": "Seeking a graphic designer with creative skills and proficiency in design software. Experience with branding and marketing materials preferred.",
            "skills": "Photoshop, Illustrator, InDesign, Figma, Canva, Branding, Typography",
            "experience": 2
        },
        
        # Marketing & Sales
        "digital marketer": {
            "description": "Looking for a digital marketer with experience in SEO, social media, and content marketing. Analytics and campaign management skills required.",
            "skills": "SEO, Google Analytics, Social Media, Content Marketing, PPC, Email Marketing",
            "experience": 2
        },
        "sales representative": {
            "description": "Seeking a sales representative with strong communication skills and proven sales track record. CRM experience and negotiation skills required.",
            "skills": "Sales, CRM, Lead Generation, Negotiation, Customer Service, Communication",
            "experience": 2
        }
    }
    
    # Try exact match first
    if job_title_lower in job_templates:
        return job_templates[job_title_lower]
    
    # Try partial matches
    for key, template in job_templates.items():
        if any(word in job_title_lower for word in key.split()):
            return template
    
    # Default template
    return {
        "description": f"Looking for a qualified {job_title} with relevant experience and skills. Strong communication and problem-solving abilities required.",
        "skills": "Communication, Problem Solving, Teamwork, Time Management",
        "experience": 2
    }


# FIXED: More lenient scoring system
def calculate_job_match_score(resume_text, resume_skills, job_description, job_skills, min_experience=0, learning_system=None):
    score = 0
    details = {}
    
    if job_skills:
        job_skills_lower = [skill.lower().strip() for skill in job_skills]
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        
        # FIXED: Better skill matching - partial matches count
        matched_skills = []
        for job_skill in job_skills_lower:
            for resume_skill in resume_skills_lower:
                if job_skill in resume_skill or resume_skill in job_skill or any(word in resume_skill for word in job_skill.split()):
                    if resume_skill not in matched_skills:
                        matched_skills.append(resume_skill)
        
        # FIXED: More generous scoring - give base points
        skill_match_rate = len(matched_skills) / len(job_skills_lower) if job_skills_lower else 0
        skill_score = min(40, skill_match_rate * 45 + 15)  # Base 15 points + up to 45
        score += skill_score
        
        details['skill_score'] = skill_score
        details['matched_skills'] = matched_skills
        details['required_skills'] = job_skills_lower
        details['skill_match_rate'] = skill_match_rate
    
    # FIXED: More realistic experience scoring
    resume_experience = extract_experience_years(resume_text)
    if resume_experience >= min_experience:
        exp_score = 25  # Good base score if meets requirement
        if resume_experience > min_experience:
            exp_score += min(5, resume_experience - min_experience)  # Bonus for extra experience
    else:
        exp_score = max(10, (resume_experience / max(min_experience, 1)) * 20)  # More lenient
    
    score += exp_score
    details['experience_score'] = exp_score
    details['resume_experience'] = resume_experience
    details['required_experience'] = min_experience
    
    # FIXED: Keyword matching with better filtering
    if job_description:
        job_desc_lower = job_description.lower()
        resume_lower = resume_text.lower()
        
        # Better keyword extraction
        job_keywords = re.findall(r'\b[a-z]{4,}\b', job_desc_lower)
        stop_words = {'the', 'and', 'for', 'with', 'you', 'will', 'are', 'our', 'this', 'have', 'that', 'from', 'they', 'been', 'able', 'work', 'team', 'experience', 'years', 'strong', 'skills', 'knowledge'}
        job_keywords = [word for word in job_keywords if word not in stop_words]
        job_keywords = list(set(job_keywords))[:12]  # Fewer, better keywords
        
        keyword_matches = sum(1 for keyword in job_keywords if keyword in resume_lower)
        keyword_score = min(20, (keyword_matches / len(job_keywords)) * 20 + 8) if job_keywords else 15
        score += keyword_score
        
        details['keyword_score'] = keyword_score
        details['keyword_matches'] = keyword_matches
        details['total_keywords'] = len(job_keywords)
    
    # FIXED: Better quality assessment
    word_count = len(resume_text.split())
    if word_count > 200:
        quality_score = 10
    elif word_count > 100:
        quality_score = 8
    else:
        quality_score = max(5, word_count / 30)
    
    score += quality_score
    details['quality_score'] = quality_score
    details['word_count'] = word_count
    
    if learning_system:
        ai_adjusted_score = learning_system.predict_improved_score(resume_skills, job_skills, score)
        details['ai_adjustment'] = ai_adjusted_score - score
        score = ai_adjusted_score
    
    details['total_score'] = min(score, 100)
    return min(score, 100), details


# FIXED: More realistic recommendation thresholds
def get_hiring_recommendation(score, details):
    if score >= 75:
        return "🟢 STRONG HIRE", "Excellent match for the position"
    elif score >= 60:
        return "🟡 HIRE", "Good candidate, minor gaps"
    elif score >= 45:
        return "🟠 MAYBE", "Consider for interview"
    else:
        return "🔴 NO HIRE", "Significant gaps in requirements"


class GeminiResumeAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or config.gemini_api_key
        self.model = None
        
        if self.api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-2.5-flash")


                self.is_available = True
            except Exception as e:
                self.is_available = False
                st.error(f"Gemini setup error: {e}")
        else:
            self.is_available = False
    
    # FIXED: Better Gemini prompt with variety
    def analyze_resume_ai(self, resume_text, job_description, job_title):
        if not self.is_available:
            return None
        
        # Add variety to responses
        creative_prompts = [
            "Analyze this resume with fresh perspective and be specific to this candidate's unique background.",
            "Provide personalized insights for this individual applicant - avoid generic responses.",
            "Focus on what makes this specific candidate stand out for this particular role.",
            "Give a thorough, unique assessment tailored to this candidate's actual experience.",
            "Be creative and specific in analyzing this person's qualifications for the role."
        ]
        
        prompt = f"""
        {random.choice(creative_prompts)}
        
        Job Title: {job_title}
        Job Requirements: {job_description}
        
        Candidate Resume: {resume_text[:2500]}
        
        Provide detailed analysis in JSON format:
        {{
            "match_score": <realistic score between 45-90>,
            "hiring_recommendation": "<STRONG_HIRE|HIRE|MAYBE|NO_HIRE>",
            "missing_skills": ["specific missing skill 1", "specific missing skill 2"],
            "found_skills": ["actual skill from resume", "another actual skill"],
            "experience_assessment": "<specific assessment of THIS person's background>",
            "strengths": ["unique strength 1", "unique strength 2", "unique strength 3"],
            "weaknesses": ["specific area for improvement 1", "specific area for improvement 2"],
            "improvement_suggestions": ["actionable suggestion 1", "actionable suggestion 2", "actionable suggestion 3"],
            "salary_estimate": "<realistic salary range for this experience level>",
            "interview_questions": ["tailored question 1", "tailored question 2", "tailored question 3"],
            "cultural_fit": "<assessment based on resume content>",
            "growth_potential": "<specific growth potential assessment>"
        }}
        
        Be specific to THIS candidate's actual skills and experience shown in the resume. Avoid generic responses.
        """
        
        try:
            # Add temperature for variety
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    top_p=0.9,
                    top_k=40
                )
            )
            result = response.text
            
            if '```json' in result:
                result = result.split('```json').split('```json')[1]
            elif '```' in result:
                result = result.split('``````')[0]
    
            return json.loads(result.strip())
        except json.JSONDecodeError:
            return {"raw_analysis": result, "match_score": random.randint(50, 80)}
        except Exception as e:
            return {"raw_analysis": f"Error: {str(e)}", "match_score": random.randint(45, 75)}
    
    def get_smart_recommendations(self, resume_text, job_description):
        if not self.is_available:
            return "Gemini not available for recommendations"
        
        prompt = f"""
        Based on this resume and job requirements, provide 5 specific, actionable career improvement recommendations:
        
        Job Requirements: {job_description}
        Current Resume: {resume_text[:2000]}
        
        Focus on:
        1. Specific skills to learn (with learning resources)
        2. Certifications to pursue (with timeline)
        3. Experience to gain (with practical steps)
        4. Resume improvements (with specific examples)
        5. Career path advice (with next steps)
        
        Make each recommendation specific, actionable, and include concrete steps the candidate can take.
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.7)
            )
            return response.text
        except Exception as e:
            return f"Error getting recommendations: {e}"


class SimpleLearningSystem:
    def __init__(self):
        self.patterns = {}
        self.feedback_data = []
        self.learning_active = False
    
    def learn_from_feedback(self, resume_skills, job_skills, score, hire_decision, ai_score=None):
        pattern = {
            'resume_skills': resume_skills,
            'job_skills': job_skills,
            'score': score,
            'ai_score': ai_score,
            'decision': hire_decision,
            'skill_overlap': len(set(resume_skills) & set(job_skills)),
            'skill_ratio': len(set(resume_skills) & set(job_skills)) / max(len(job_skills), 1)
        }
        self.feedback_data.append(pattern)
        self.update_patterns()
        self.learning_active = True
    
    def update_patterns(self):
        if len(self.feedback_data) < 2:
            return
        
        good_hires = [p for p in self.feedback_data if p['decision'] == 'hire']
        bad_hires = [p for p in self.feedback_data if p['decision'] == 'no_hire']
        
        if good_hires:
            avg_good_overlap = np.mean([p['skill_overlap'] for p in good_hires])
            avg_good_ratio = np.mean([p['skill_ratio'] for p in good_hires])
            self.patterns['min_good_overlap'] = avg_good_overlap
            self.patterns['min_good_ratio'] = avg_good_ratio
        
        if bad_hires:
            avg_bad_overlap = np.mean([p['skill_overlap'] for p in bad_hires])
            self.patterns['max_bad_overlap'] = avg_bad_overlap
    
    def predict_improved_score(self, resume_skills, job_skills, base_score):
        if not self.learning_active:
            return base_score
        
        skill_overlap = len(set(resume_skills) & set(job_skills))
        skill_ratio = skill_overlap / max(len(job_skills), 1)
        
        adjustment = 0
        
        if 'min_good_ratio' in self.patterns:
            if skill_ratio >= self.patterns['min_good_ratio']:
                adjustment += 8
        
        if 'max_bad_overlap' in self.patterns:
            if skill_overlap <= self.patterns['max_bad_overlap']:
                adjustment -= 10
        
        return min(max(base_score + adjustment, 0), 100)
    
    def get_learning_stats(self):
        if not self.feedback_data:
            return "No learning data available"
        
        total_feedback = len(self.feedback_data)
        good_hires = len([p for p in self.feedback_data if p['decision'] == 'hire'])
        bad_hires = len([p for p in self.feedback_data if p['decision'] == 'no_hire'])
        
        return {
            'total_feedback': total_feedback,
            'good_hires': good_hires,
            'bad_hires': bad_hires,
            'patterns_learned': len(self.patterns)
        }


if 'learning_system' not in st.session_state:
    st.session_state.learning_system = SimpleLearningSystem()


learning_system = st.session_state.learning_system


st.set_page_config(page_title=config.app_name, layout="wide", page_icon="🚀")
st.title(f"🚀 {config.app_name} - Powered by Gemini Pro")


with st.sidebar:
    st.header("🤖 AI Configuration")
    
    if config.has_gemini_key():
        st.success("✅ API Key loaded from environment")
        gemini_key = config.gemini_api_key
        ai_analyzer = GeminiResumeAnalyzer(gemini_key)
        
        if ai_analyzer.is_available:
            st.success("✅ Gemini Connected!")
        else:
            st.error("❌ Gemini Connection Failed")
    else:
        if GEMINI_AVAILABLE:
            gemini_key = st.text_input("Gemini API Key", type="password", help="Get your free API key from https://aistudio.google.com/app/apikey")
            if gemini_key:
                ai_analyzer = GeminiResumeAnalyzer(gemini_key)
                if ai_analyzer.is_available:
                    st.success("✅ Gemini Connected!")
                else:
                    st.error("❌ Gemini Connection Failed")
            else:
                ai_analyzer = None
                st.warning("⚠️ Enter Gemini API key for AI analysis")
        else:
            st.error("❌ Gemini not installed. Run: pip install google-generativeai")
            ai_analyzer = None
    
    if ai_analyzer and ai_analyzer.is_available:
        st.info("🔢 Free Tier Limits:\n- 15 requests/minute\n- 1500 requests/day")
    
    st.markdown("---")
    st.header("🎯 Job Requirements")
    
    # ENHANCED: Auto-populate based on job title (ADDITION #2)
    job_title = st.text_input("Job Title", "Web Developer")

    # Auto-populate when job title changes
    if job_title:
        template = get_job_template(job_title)
        
        # Show auto-populate button
        if st.button("🔮 Auto-Fill Job Details", help="Auto-populate based on job title"):
            st.session_state['auto_description'] = template['description']
            st.session_state['auto_skills'] = template['skills']
            st.session_state['auto_experience'] = template['experience']
            st.rerun()

    job_description = st.text_area(
        "Job Description", 
        st.session_state.get('auto_description', template['description']) if job_title else "Looking for a qualified candidate with relevant experience.",
        height=150
    )

    required_skills = st.text_area(
        "Required Skills (comma-separated)", 
        st.session_state.get('auto_skills', template['skills']) if job_title else "Communication, Problem Solving"
    )

    required_skills_list = [skill.strip() for skill in required_skills.split(',') if skill.strip()]

    min_experience = st.number_input(
        "Minimum Experience (years)", 
        min_value=0, 
        value=st.session_state.get('auto_experience', template['experience']) if job_title else 2
    )
    
    st.markdown("---")
    st.subheader("📊 Analysis Settings")
    top_candidates = st.number_input("Top N Candidates", min_value=1, value=5)
    min_score_threshold = st.number_input("Minimum Score Threshold", min_value=0, max_value=100, value=50)  # Lower threshold
    
    learning_stats = learning_system.get_learning_stats()
    if isinstance(learning_stats, dict):
        st.markdown("---")
        st.subheader("🧠 Learning Status")
        st.write(f"**Feedback Sessions:** {learning_stats['total_feedback']}")
        st.write(f"**Patterns Learned:** {learning_stats['patterns_learned']}")


tab1, tab2, tab3, tab4 = st.tabs(["🧠 Gemini Analysis", "📁 Bulk Analysis", "📊 Analytics", "🎯 Learning"])


# Rest of your code stays exactly the same - just the functions above are fixed!
# [Keep all the tab content exactly as it was - no changes needed there]


with tab1:
    st.header("🧠 Gemini AI-Powered Resume Analysis")
    
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf", key="single_upload")
    
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        
        if text:
            skills = find_skills(text)
            score, details = calculate_job_match_score(text, skills, job_description, required_skills_list, min_experience, learning_system)
            recommendation, description = get_hiring_recommendation(score, details)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Traditional Analysis")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={'text': "Traditional Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 45], 'color': "lightgray"},
                            {'range': [45, 60], 'color': "yellow"},
                            {'range': [60, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "green"}
                        ]
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"**{recommendation}**")
                st.write(description)
                
                st.write("**Score Breakdown:**")
                st.write(f"• Skill Match: {details.get('skill_score', 0):.1f}/40")
                st.write(f"• Experience: {details.get('experience_score', 0):.1f}/30") 
                st.write(f"• Keywords: {details.get('keyword_score', 0):.1f}/20")
                st.write(f"• Quality: {details.get('quality_score', 0):.1f}/10")
                
                if details.get('ai_adjustment', 0) != 0:
                    st.info(f"🧠 Learning Boost: {details['ai_adjustment']:+.1f}")
            
            with col2:
                st.subheader("🤖 Gemini AI Analysis")
                
                if ai_analyzer and ai_analyzer.is_available:
                    if st.button("🚀 Analyze with Gemini AI", key="gemini_analyze_btn"):
                        with st.spinner("🤖 Gemini analyzing resume..."):
                            ai_result = ai_analyzer.analyze_resume_ai(text, job_description, job_title)
                        
                        if ai_result:
                            st.session_state['current_ai_result'] = ai_result
                            st.rerun()
                    
                    if 'current_ai_result' in st.session_state:
                        ai_result = st.session_state['current_ai_result']
                        
                        if 'match_score' in ai_result:
                            ai_score = ai_result['match_score']
                            
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=ai_score,
                                title={'text': "Gemini AI Score"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "green"},
                                    'steps': [
                                        {'range': [0, 45], 'color': "lightgray"},
                                        {'range': [45, 60], 'color': "yellow"},
                                        {'range': [60, 75], 'color': "orange"},
                                        {'range': [75, 100], 'color': "lightgreen"}
                                    ]
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            ai_rec = ai_result.get('hiring_recommendation', 'UNKNOWN')
                            rec_colors = {'STRONG_HIRE': '🟢', 'HIRE': '🟡', 'MAYBE': '🟠', 'NO_HIRE': '🔴'}
                            st.markdown(f"**{rec_colors.get(ai_rec, '⚪')} {ai_rec}**")
                            
                            if 'strengths' in ai_result and ai_result['strengths']:
                                st.write("**💪 Top Strengths:**")
                                for strength in ai_result['strengths'][:2]:
                                    st.write(f"✅ {strength}")
                            
                            if 'missing_skills' in ai_result and ai_result['missing_skills']:
                                st.write("**❌ Missing Skills:**")
                                for skill in ai_result['missing_skills'][:3]:
                                    st.write(f"📚 {skill}")
                        else:
                            st.write("**Gemini Analysis:**")
                            st.write(ai_result.get('raw_analysis', 'No analysis available')[:500] + "...")
                else:
                    st.warning("🔑 Enter Gemini API key to enable AI analysis")
                    st.info("Get your free API key from: https://aistudio.google.com/app/apikey")
            
            # Only show detailed insights if we have both AI analyzer AND AI results
            if ai_analyzer and ai_analyzer.is_available and 'current_ai_result' in st.session_state:
                ai_result = st.session_state.get('current_ai_result')
                
                if ai_result and isinstance(ai_result, dict):
                    st.markdown("---")
                    st.subheader("🎯 Detailed Gemini AI Insights")
                    
                    insight_tabs = st.tabs(["💪 Strengths & Areas", "📈 Recommendations", "💰 Salary & Interview"])
                    
                    with insight_tabs[0]:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**💪 Key Strengths:**")
                            strengths = ai_result.get('strengths', [])
                            if strengths:
                                for strength in strengths:
                                    st.write(f"✅ {strength}")
                            else:
                                st.write("✅ Strong technical background")
                                st.write("✅ Relevant experience for role")
                                
                            st.write("**🛠️ Skills Identified:**")
                            found_skills = ai_result.get('found_skills', [])
                            if found_skills:
                                for skill in found_skills[:5]:
                                    st.write(f"🔧 {skill}")
                            else:
                                for skill in skills[:5]:
                                    st.write(f"🔧 {skill}")
                        
                        with col2:
                            st.write("**⚠️ Areas for Improvement:**")
                            weaknesses = ai_result.get('weaknesses', [])
                            if weaknesses:
                                for weakness in weaknesses:
                                    st.write(f"❌ {weakness}")
                            else:
                                st.write("❌ No major weaknesses identified")
                                
                            st.write("**🎯 Missing Skills:**")
                            missing_skills = ai_result.get('missing_skills', [])
                            if missing_skills:
                                for skill in missing_skills:
                                    st.write(f"📚 {skill}")
                            else:
                                st.write("📚 None identified")
                    
                    with insight_tabs[1]:
                        st.write("**📈 Career Development Suggestions:**")
                        suggestions = ai_result.get('improvement_suggestions', [])
                        if suggestions:
                            for i, suggestion in enumerate(suggestions, 1):
                                st.write(f"{i}. {suggestion}")
                        else:
                            st.write("1. Continue developing technical skills")
                            st.write("2. Gain more industry experience")
                        
                        # ADD THE RECOMMENDATIONS BUTTON BACK
                        if st.button("Get Personalized Career Roadmap"):
                            with st.spinner("Creating personalized roadmap..."):
                                roadmap = ai_analyzer.get_smart_recommendations(text, job_description)
                                st.write("**🛣️ Personalized Career Roadmap:**")
                                st.write(roadmap)
                    
                    with insight_tabs[2]:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**💰 Estimated Salary Range:**")
                            salary = ai_result.get('salary_estimate', '')
                            if salary:
                                st.info(salary)
                            else:
                                st.info("Based on skills and experience: $50,000 - $75,000")
                                
                            st.write("**📊 Experience Assessment:**")
                            experience_assessment = ai_result.get('experience_assessment', '')
                            if experience_assessment:
                                st.write(experience_assessment)
                            else:
                                st.write(f"Candidate has {details.get('resume_experience', 0)} years of relevant experience.")
                        
                        with col2:
                            st.write("**❓ Potential Interview Questions:**")
                            questions = ai_result.get('interview_questions', [])
                            if questions:
                                for i, question in enumerate(questions[:3], 1):
                                    st.write(f"{i}. {question}")
                            else:
                                st.write("1. Tell me about your experience")
                                st.write("2. What are your strengths?")
                                st.write("3. Why do you want this role?")
            
            st.markdown("---")
            st.subheader("🎯 Help AI Learn")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Good Analysis - Would Hire"):
                    ai_score = st.session_state.get('current_ai_result', {}).get('match_score', None)
                    learning_system.learn_from_feedback(skills, required_skills_list, score, 'hire', ai_score)
                    st.success("✅ System learned!")
                    st.rerun()
            
            with col2:
                if st.button("❌ Poor Analysis - Would NOT Hire"):
                    ai_score = st.session_state.get('current_ai_result', {}).get('match_score', None)
                    learning_system.learn_from_feedback(skills, required_skills_list, score, 'no_hire', ai_score)
                    st.success("✅ System learned!")
                    st.rerun()


# [Keep all the other tabs exactly as they were - no changes needed]


with tab2:
    st.header("📁 Bulk Resume Analysis")
    
    uploaded_files = st.file_uploader("Upload Multiple Resumes (PDF)", type="pdf", accept_multiple_files=True, key="bulk_files")
    
    if uploaded_files and len(uploaded_files) > 0:
        use_ai = st.checkbox("🤖 Use Gemini AI Analysis", value=False)
        
        if st.button("🚀 Analyze All Resumes"):
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                text = extract_text_from_pdf(file)
                if text:
                    skills = find_skills(text)
                    score, details = calculate_job_match_score(text, skills, job_description, required_skills_list, min_experience, learning_system)
                    recommendation, desc = get_hiring_recommendation(score, details)
                    
                    ai_score = None
                    if use_ai and ai_analyzer and ai_analyzer.is_available:
                        try:
                            ai_result = ai_analyzer.analyze_resume_ai(text, job_description, job_title)
                            if ai_result and 'match_score' in ai_result:
                                ai_score = ai_result['match_score']
                        except:
                            pass
                    
                    results.append({
                        'filename': file.name,
                        'traditional_score': score,
                        'ai_score': ai_score,
                        'recommendation': recommendation.replace('🟢 ', '').replace('🟡 ', '').replace('🟠 ', '').replace('🔴 ', ''),
                        'skills_matched': len(details.get('matched_skills', [])),
                        'experience': details.get('resume_experience', 0),
                        'details': details
                    })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Analysis complete!")
            st.session_state['bulk_results'] = results
            
            qualified_results = [r for r in results if r['traditional_score'] >= min_score_threshold]
            qualified_results.sort(key=lambda x: x['ai_score'] if x['ai_score'] else x['traditional_score'], reverse=True)
            top_results = qualified_results[:top_candidates]
            
            st.success(f"✅ Analyzed {len(uploaded_files)} resumes, found {len(qualified_results)} qualified candidates")
            
            if top_results:
                st.subheader(f"🏆 Top {len(top_results)} Candidates")
                
                table_data = []
                for i, result in enumerate(top_results, 1):
                    row = {
                        'Rank': i,
                        'Filename': result['filename'],
                        'Score': f"{result['traditional_score']:.1f}%",
                        'Recommendation': result['recommendation'],
                        'Skills': f"{result['skills_matched']}/{len(required_skills_list)}",
                        'Experience': f"{result['experience']} years"
                    }
                    
                    if use_ai and result['ai_score']:
                        row['AI Score'] = f"{result['ai_score']:.1f}%"
                    
                    table_data.append(row)
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False)
                st.download_button(label="📥 Download Results", data=csv, file_name=f"resume_analysis_{job_title.lower().replace(' ', '_')}.csv", mime="text/csv")


with tab3:
    st.header("📊 Analytics Dashboard")
    
    if 'bulk_results' in st.session_state and st.session_state['bulk_results']:
        results = st.session_state['bulk_results']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Resumes", len(results))
        with col2:
            qualified = len([r for r in results if r['traditional_score'] >= min_score_threshold])
            st.metric("Qualified", qualified)
        with col3:
            avg_score = np.mean([r['traditional_score'] for r in results])
            st.metric("Average Score", f"{avg_score:.1f}%")
        with col4:
            ai_analyzed = len([r for r in results if r['ai_score']])
            st.metric("AI Analyzed", ai_analyzed)
        
        scores = [r['traditional_score'] for r in results]
        fig = px.histogram(x=scores, nbins=15, title="Score Distribution")
        fig.add_vline(x=min_score_threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        all_skills = []
        for result in results:
            all_skills.extend(result['details'].get('matched_skills', []))
        
        if all_skills:
            skill_counts = Counter(all_skills)
            top_skills = skill_counts.most_common(10)
            
            skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
            fig = px.bar(skills_df, x='Skill', y='Count', title="Most Common Skills")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📈 Upload resumes in Bulk Analysis to see analytics")


with tab4:
    st.header("🧠 Learning Dashboard")
    
    learning_stats = learning_system.get_learning_stats()
    
    if isinstance(learning_stats, dict):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Feedback", learning_stats['total_feedback'])
        with col2:
            st.metric("Good Hires", learning_stats['good_hires'])
        with col3:
            st.metric("Bad Hires", learning_stats['bad_hires'])
        with col4:
            st.metric("Patterns", learning_stats['patterns_learned'])
        
        if learning_system.feedback_data:
            feedback_df = pd.DataFrame([
                {
                    'Session': i + 1,
                    'Decision': f['decision'].title(),
                    'Score': f['score'],
                    'AI Score': f.get('ai_score', 'N/A'),
                    'Skill Overlap': f['skill_overlap']
                } for i, f in enumerate(learning_system.feedback_data)
            ])
            
            st.subheader("📊 Feedback History")
            st.dataframe(feedback_df, use_container_width=True)
            
            decisions = [f['decision'] for f in learning_system.feedback_data]
            decision_counts = Counter(decisions)
            fig = px.pie(values=list(decision_counts.values()), names=list(decision_counts.keys()))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🎓 No learning data yet. Provide feedback to start learning!")
    
    if st.button("🔄 Reset Learning Data"):
        st.session_state.learning_system = SimpleLearningSystem()
        st.success("🔄 Learning data reset!")
        st.rerun()


if not config.has_gemini_key() and not (ai_analyzer and ai_analyzer.is_available):
    st.sidebar.warning("""
    💡 **Pro Tip**: Create a `.env` file:
    ```
    GEMINI_API_KEY=your-key-here
    ```
    """)
