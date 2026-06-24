#  AI Talent Suite & Resume Analyzer
**[Live Web Application] (https://ai-resume-frontend-tau.vercel.app/)** | **[Frontend Code Repository] (https://github.com/Krishtiy/ai-resume-frontend.)**

A hybrid AI platform designed to automate technical recruitment. This system moves beyond basic keyword matching by combining traditional deterministic extraction with modern dense-vector semantic mathematics and generative AI reasoning.

![Demo Link](https://img.shields.io/badge/Live_Demo-Available_Here-blue?style=for-the-badge)

## System Architecture
The project is split into two distinct environments:
1. **The Client (Frontend):** A lightweight, static web interface hosted on Vercel for instant load times and responsive design. 
2. **The Engine (Backend - This Repo):** A heavy-lifting Python API containerized via Docker and deployed on Hugging Face Spaces to provide the necessary 16GB RAM for machine learning inference.
   
This project utilizes a dual engine NLP approach to grade candidate viability:

1. **Deterministic Extraction  (NER):** Utilizes `skillNer` and `spaCy` to dynamically extract frameworks, libraries, and tools from unstructured text using context-aware Named Entity Recognition.
2. **Semantic Math Engine (Dense Embeddings):** uses HuggingFace's `SentenceTransformers` (`all-MiniLM-L6-v2`). It maps the resume and Job Description into a 384-dimensional vector space to calculate true conceptual alignment via Cosine Similarity.
3. **Generative Reasoning Engine:** Integrates Google's `Gemini 2.5 Flash` to act as an AI career coach, providing JSON-structured gap analysis, strength identification, and actionable career roadmaps.

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **Database:** SQLite, SQLAlchemy ORM (Persistent Scan Analytics)
* **Machine Learning:** HuggingFace `SentenceTransformers`, `spaCy`, `skillNer`
* **Generative AI:** Google Gemini API
* **Frontend:** HTML5, TailwindCSS, Vanilla JS (Decoupled REST API architecture)
* **Document Parsing:** `pdfplumber`

## 📊 Features
* **ATS Resume Matcher:** Upload a PDF and Job Description for instant vector-based scoring and AI feedback.
* **LinkedIn Profile Auditor:** Upload a LinkedIn PDF export to generate personal branding optimizations and SEO keyword suggestions.
* **Recruiter Analytics Hub:** View live dashboard metrics, global average scores, and database-driven historical scan logs.
