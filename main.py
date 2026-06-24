from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 
import pdfplumber
import io
from sentence_transformers import SentenceTransformer, util
import os
import json
from dotenv import load_dotenv

from google import genai
from google.genai import types

from sqlalchemy.orm import Session 
import models 
from database import engine, SessionLocal
import spacy

# Change your skillNer imports to exactly this:
# Replace your skillNer imports with this exact snippet:
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB
from spacy.matcher import PhraseMatcher



from collections import Counter       # Helps count the most common skills
from sqlalchemy.sql import func       # Helps calculate the average score in SQLite

load_dotenv()

# It automatically looks for "GEMINI_API_KEY" in your .env file
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="AI Resume Analyzer API")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
# Command SQLAlchemy to build database tables if they do not exist
models.Base.metadata.create_all(bind=engine)

# Database session dependency provider
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("en_core_web_sm")
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

#SKILL_DB = {
 #   "python", "fastapi", "flask", "django", "react", "node.js", 
 #   "javascript", "typescript", "html", "css", "sql", "mongodb", 
 #   "postgresql", "git", "github", "docker", "aws", "azure", 
 #   "machine learning", "deep learning", "nlp", "natural language processing",
  #  "opencv", "scikit-learn", "tensorflow", "pytorch", "c++", "java"
#

# --- PYDANTIC SCHEMAS ---
class MatchSummary(BaseModel):
    resume_word_count: int
    jd_word_count: int

class AIAnalysis(BaseModel):
    strengths: list[str]
    skill_gaps: list[str]
    improvement_suggestions: list[str]

class AnalysisResponse(BaseModel):
    filename: str
    ats_score: str
    extracted_skills: list[str]
    match_summary: MatchSummary
    ai_analysis: AIAnalysis

class ScanHistoryResponse(BaseModel):
    id: int
    filename: str
    ats_score: float
    skills_found: str
    scan_date: str 

    class Config:
        from_attributes = True 

class DashboardAnalytics(BaseModel):
    total_scans: int
    average_score: float
    top_skills: list[str]

class LinkedInAudit(BaseModel):
    profile_strength: str  # e.g., "Beginner", "Intermediate", "All-Star"
    summary_feedback: str
    missing_optimization_keywords: list[str]
    suggested_headline_updates: list[str]

class LinkedInAnalysisResponse(BaseModel):
    filename: str
    word_count: int
    audit: LinkedInAudit

# --- API ROUTES ---

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "AI Resume Analyzer Backend is running successfully!"}


@app.post("/upload-resume", response_model=AnalysisResponse)
async def upload_resume(file: UploadFile = File(...), job_description: str = Form(...), db: Session = Depends(get_db)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # 1. PDF Text Extraction
        file_bytes = await file.read()
        raw_text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"
        #safeguard Check if PDF text is empty (scanned image
        if not raw_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="Could not read text from this PDF. Please ensure it isn't a scanned image/image-only file."
            )
        # 2. NLP Skill Extraction w/o NER
        #clean_text = raw_text.lower()
        #doc = nlp(clean_text)
        #extracted_skills = set()
        
        #for token in doc:
        #    if token.text in SKILL_DB:
        #        extracted_skills.add(token.text)
        # 2.  NLP Skill Extraction (nER)
        clean_text = raw_text.lower()
    
        annotations = skill_extractor.annotate(clean_text)
        
        extracted_skills = set()
        
        # Unpack the matches discovered by the NER model
        # skill_extractor catches "Hard Skills" (tech) and "Soft Skills"
        # 🛡️ FIX: Changed "ngram_matches" to "ngram_scored" and used .get() to prevent crashes
        for match_type in ["full_matches", "ngram_scored"]:
            for match in annotations.get("results", {}).get(match_type, []):
                skill_id = match.get("skill_id")
                if skill_id in SKILL_DB:
                    skill_name = SKILL_DB[skill_id]["skill_name"]
                    extracted_skills.add(skill_name)
                
        #for skill in SKILL_DB: without ner
          #  if " " in skill and skill in clean_text:
             #   extracted_skills.add(skill)

        # 3. TF-IDF & Cosine Similarity Scoring
        # 3. Semantic Scoring (Dense Embeddings)
        clean_jd = job_description.lower()
        # Translate the text into 384 dimen arrays
        resume_vector = embedder.encode(clean_text, convert_to_tensor=True)
        jd_vector = embedder.encode(clean_jd, convert_to_tensor=True)
        # Calculate the PyTorch cosine similarity angle between the two meanings
        similarity_tensor = util.cos_sim(resume_vector, jd_vector)
        
        # Extract the raw float value
        raw_score = float(similarity_tensor[0][0])
        
        # Semantic math can occasionally return slight negative numbers for total polar opposites.
        # We clamp the absolute minimum at 0.0 for a cleaner UI percentage.
        ats_score = round(max(0.0, raw_score) * 100, 2)
# 4. Generative AI Logic
        prompt = f"""
        You are an expert technical recruiter and advanced ATS system.
        Analyze the following Resume against the provided Job Description.
        Provide a contextual skill gap analysis and actionable improvement suggestions.

        Resume Text:
        {raw_text}

        Job Description:
        {job_description}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AIAnalysis, # ✅ Forces Gemini to output exactly your Pydantic structure
            ),
        )
        
        # ✅ No more manual markdown cleaning needed!
        llm_analysis = json.loads(response.text)
        #
        # 💾 Save tracking data history into DB (Keeps your dashboard active!)
        skills_string = ", ".join(list(extracted_skills)) if extracted_skills else "None"
        new_scan = models.ScanRecord(
            filename=file.filename,
            ats_score=float(ats_score), 
            skills_found=skills_string
        )
        db.add(new_scan)
        db.commit()
        db.refresh(new_scan)
        # 5. Return structured output matching AnalysisResponse schema
        return {
            "filename": file.filename,
            "ats_score": f"{ats_score}%",
            "extracted_skills": list(extracted_skills),
            "match_summary": {
                "resume_word_count": len(clean_text.split()),
                "jd_word_count": len(clean_jd.split())
            },
            "ai_analysis": llm_analysis
        }
        
    except Exception as e:
        import traceback
        # This will print the precise line that broke directly to your Uvicorn terminal console
        print("--- BACKEND CRASH TRACEBACK ---")
        print(traceback.format_exc())
        print("-------------------------------")
        raise HTTPException(status_code=500, detail=f"Python Error: {str(e)}")
#linkedi
@app.post("/analyze-linkedin", response_model=LinkedInAnalysisResponse)
async def analyze_linkedin(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # 1. Reuse our PDF Extraction Engine
        file_bytes = await file.read()
        raw_text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"
        
        # 2. Specialized Prompt for LinkedIn Profile Optimization
        prompt = f"""
        You are an expert personal branding coach and corporate technical recruiter.
        Analyze the following text extracted from a LinkedIn Profile PDF export.
        
        Provide a comprehensive audit focusing on profile optimization, search discoverability, and career positioning.
        Do not output markdown, conversational text, or explanations.
        Respond strictly with a valid JSON object using exactly these keys:
        - "profile_strength": A single string rating the profile structure (Choose from: "Needs Work", "Good", "All-Star").
        - "summary_feedback": A brief 2-3 sentence critique of their professional summary/about section.
        - "missing_optimization_keywords": A list of 4 core industry technical keywords they should add to rank higher in recruiter searches.
        - "suggested_headline_updates": A list of 2 alternative, high-impact headlines they could use under their name.

        LinkedIn Profile Text:
        {raw_text}
        """

        # 3. Request analysis from the Gemini Client
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        
        # Clean potential markdown wrappers
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1)
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0]
        response_text = response_text.strip()
        
        llm_audit = json.loads(response_text)

        # 4. Return structural response
        return {
            "filename": file.filename,
            "word_count": len(raw_text.split()),
            "audit": llm_audit
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LinkedIn Audit Failed: {str(e)}")

@app.get("/scans", response_model=list[ScanHistoryResponse])
def get_scan_history(db: Session = Depends(get_db)):
    try:
        records = db.query(models.ScanRecord).order_by(models.ScanRecord.scan_date.desc()).all()
        
        formatted_records = []
        for record in records:
            formatted_records.append({
                "id": record.id,
                "filename": record.filename,
                "ats_score": record.ats_score,
                "skills_found": record.skills_found,
                "scan_date": record.scan_date.strftime("%Y-%m-%d %H:%M:%S") if record.scan_date else "N/A"
            })
            
        return formatted_records
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Fetch Error: {str(e)}")


@app.get("/analytics", response_model=DashboardAnalytics)
def get_analytics(db: Session = Depends(get_db)):
    try:
        # Calculate Total Scans
        total = db.query(models.ScanRecord).count()
        
        if total == 0:
            return {"total_scans": 0, "average_score": 0.0, "top_skills": []}

        # Calculate Average ATS Score
        avg_score = db.query(func.avg(models.ScanRecord.ats_score)).scalar()
        
        # Compile Top 5 Most Common Skills
        all_records = db.query(models.ScanRecord.skills_found).all()
        
        all_skills = []
        for record in all_records:
            if record[0]: 
                skills = [skill.strip() for skill in record[0].split(",")]
                all_skills.extend(skills)
                
        top_5_skills = [skill[0] for skill in Counter(all_skills).most_common(5)]

        return {
            "total_scans": total,
            "average_score": round(avg_score, 2) if avg_score else 0.0,
            "top_skills": top_5_skills
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analytics Fetch Error: {str(e)}")
import uvicorn

if __name__ == "__main__":
    # Read the dynamic port assigned by Render, default to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    # Run the server, listening on all available network interfaces
    uvicorn.run(app, host="0.0.0.0", port=port)