import sqlite3
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# MOCK DATA

EMPLOYEES = [
    {"emp_id": "EMP001", "name": "Abhay Kumar",     "skills": "Python, Machine Learning, Data Analysis, SQL, Power BI",           "experience": 4, "location": "Mumbai",    "gender": "Male",   "process": "Analytics",    "status": "bench",  "designation": "Senior Analyst"},
    {"emp_id": "EMP002", "name": "Priya Sharma",    "skills": "Java, Spring Boot, Microservices, AWS, Docker",                    "experience": 6, "location": "Bangalore", "gender": "Female", "process": "Engineering",  "status": "bench",  "designation": "Software Engineer"},
    {"emp_id": "EMP003", "name": "Rohan Mehta",     "skills": "Python, LangChain, RAG, LLM, FastAPI, GenAI",                      "experience": 3, "location": "Hyderabad", "gender": "Male",   "process": "AI Team",      "status": "bench",  "designation": "AI Engineer"},
    {"emp_id": "EMP004", "name": "Sneha Iyer",      "skills": "React, JavaScript, TypeScript, Node.js, CSS",                      "experience": 5, "location": "Chennai",   "gender": "Female", "process": "Frontend",     "status": "bench",  "designation": "Frontend Developer"},
    {"emp_id": "EMP005", "name": "Vikram Singh",    "skills": "Data Science, Python, TensorFlow, Deep Learning, NLP",             "experience": 7, "location": "Mumbai",    "gender": "Male",   "process": "Data Science", "status": "bench",  "designation": "Lead Data Scientist"},
    {"emp_id": "EMP006", "name": "Anita Desai",     "skills": "SQL, Power BI, DAX, Excel, Data Visualization, Tableau",           "experience": 4, "location": "Pune",      "gender": "Female", "process": "Analytics",    "status": "bench",  "designation": "Business Analyst"},
    {"emp_id": "EMP007", "name": "Karan Patel",     "skills": "Azure, DevOps, Kubernetes, Terraform, CI/CD, Docker",              "experience": 5, "location": "Bangalore", "gender": "Male",   "process": "DevOps",       "status": "bench",  "designation": "DevOps Engineer"},
    {"emp_id": "EMP008", "name": "Meera Nair",      "skills": "Python, scikit-learn, Feature Engineering, A/B Testing, Statistics","experience": 3, "location": "Hyderabad","gender": "Female", "process": "Analytics",    "status": "bench",  "designation": "Data Analyst"},
    {"emp_id": "EMP009", "name": "Arjun Reddy",     "skills": "Java, Python, Microservices, REST API, SQL, Agile",                "experience": 8, "location": "Mumbai",    "gender": "Male",   "process": "Engineering",  "status": "bench",  "designation": "Senior Developer"},
    {"emp_id": "EMP010", "name": "Divya Krishnan",  "skills": "LLM, Prompt Engineering, Python, RAG, Vector Databases, LangChain","experience": 2, "location": "Chennai",   "gender": "Female", "process": "AI Team",      "status": "bench",  "designation": "AI Engineer"},
    {"emp_id": "EMP011", "name": "Siddharth Joshi", "skills": "Angular, Vue.js, React, JavaScript, HTML, CSS, Bootstrap",         "experience": 4, "location": "Pune",      "gender": "Male",   "process": "Frontend",     "status": "bench",  "designation": "UI Developer"},
    {"emp_id": "EMP012", "name": "Kavya Menon",     "skills": "Python, Apache Spark, Hadoop, Data Engineering, ETL, Airflow",    "experience": 6, "location": "Bangalore", "gender": "Female", "process": "Data Engg",    "status": "bench",  "designation": "Data Engineer"},
    {"emp_id": "EMP013", "name": "Rahul Gupta",     "skills": "Cybersecurity, Network Security, SIEM, Penetration Testing",      "experience": 5, "location": "Delhi",     "gender": "Male",   "process": "Security",     "status": "bench",  "designation": "Security Analyst"},
    {"emp_id": "EMP014", "name": "Pooja Agarwal",   "skills": "Project Management, Agile, Scrum, JIRA, Stakeholder Management",  "experience": 9, "location": "Mumbai",    "gender": "Female", "process": "PMO",          "status": "bench",  "designation": "Project Manager"},
    {"emp_id": "EMP015", "name": "Nikhil Bhatt",    "skills": "Python, Django, Flask, PostgreSQL, REST API, Redis",              "experience": 4, "location": "Ahmedabad", "gender": "Male",   "process": "Backend",      "status": "bench",  "designation": "Backend Developer"},
]

JOBS = [
    {"job_id": "JOB001", "title": "AI Engineer",              "required_skills": "Python, LLM, RAG, LangChain, Vector Databases, GenAI",          "location": "Mumbai",    "department": "Technology",   "openings": 2, "experience_required": 2},
    {"job_id": "JOB002", "title": "Data Scientist",           "required_skills": "Python, Machine Learning, Deep Learning, NLP, TensorFlow",        "location": "Bangalore", "department": "Analytics",    "openings": 1, "experience_required": 4},
    {"job_id": "JOB003", "title": "Frontend Developer",       "required_skills": "React, JavaScript, TypeScript, CSS, Node.js",                     "location": "Chennai",   "department": "Engineering",  "openings": 3, "experience_required": 2},
    {"job_id": "JOB004", "title": "DevOps Engineer",          "required_skills": "Azure, Kubernetes, Docker, Terraform, CI/CD, DevOps",             "location": "Bangalore", "department": "Infrastructure","openings": 2, "experience_required": 3},
    {"job_id": "JOB005", "title": "Business Analyst",         "required_skills": "SQL, Power BI, Excel, Data Visualization, DAX, Tableau",          "location": "Mumbai",    "department": "Analytics",    "openings": 2, "experience_required": 2},
    {"job_id": "JOB006", "title": "Backend Developer",        "required_skills": "Python, Django, Flask, REST API, PostgreSQL, Redis",               "location": "Pune",      "department": "Engineering",  "openings": 2, "experience_required": 3},
    {"job_id": "JOB007", "title": "Data Engineer",            "required_skills": "Python, Apache Spark, ETL, Airflow, Hadoop, Data Pipelines",       "location": "Bangalore", "department": "Data",         "openings": 1, "experience_required": 4},
    {"job_id": "JOB008", "title": "Senior Software Engineer", "required_skills": "Java, Microservices, REST API, Spring Boot, SQL, Agile",           "location": "Hyderabad", "department": "Engineering",  "openings": 2, "experience_required": 5},
    {"job_id": "JOB009", "title": "ML Engineer",              "required_skills": "Python, Machine Learning, scikit-learn, Feature Engineering, MLOps","location": "Mumbai",   "department": "AI",           "openings": 1, "experience_required": 3},
    {"job_id": "JOB010", "title": "Security Analyst",         "required_skills": "Cybersecurity, Network Security, SIEM, Penetration Testing",       "location": "Delhi",     "department": "Security",     "openings": 1, "experience_required": 3},
    {"job_id": "JOB011", "title": "Project Manager",          "required_skills": "Project Management, Agile, Scrum, JIRA, Stakeholder Management",   "location": "Mumbai",    "department": "PMO",          "openings": 2, "experience_required": 6},
    {"job_id": "JOB012", "title": "Prompt Engineer",          "required_skills": "LLM, Prompt Engineering, Python, GenAI, RAG, LangChain",           "location": "Hyderabad", "department": "AI",           "openings": 2, "experience_required": 1},
]


# CREATE DATABASE

def create_database():
    print("Creating SQLite database...")
    conn = sqlite3.connect("rmg_database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            emp_id TEXT PRIMARY KEY, name TEXT, skills TEXT,
            experience INTEGER, location TEXT, gender TEXT,
            process TEXT, status TEXT, designation TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY, title TEXT, required_skills TEXT,
            location TEXT, department TEXT, openings INTEGER,
            experience_required INTEGER
        )
    """)

    for emp in EMPLOYEES:
        cursor.execute("INSERT OR REPLACE INTO employees VALUES (?,?,?,?,?,?,?,?,?)",
            (emp["emp_id"], emp["name"], emp["skills"], emp["experience"],
             emp["location"], emp["gender"], emp["process"], emp["status"], emp["designation"]))

    for job in JOBS:
        cursor.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?)",
            (job["job_id"], job["title"], job["required_skills"], job["location"],
             job["department"], job["openings"], job["experience_required"]))

    conn.commit()
    conn.close()
    print(f"  ✅ Database created: {len(EMPLOYEES)} employees, {len(JOBS)} jobs")


#  BUILD VECTOR STORES


def get_embeddings():
    print("  Loading HuggingFace embedding model")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def build_job_vectorstore(embeddings):
    print("Building job vector store")
    job_docs = []
    for job in JOBS:
        content = f"""
Job Title: {job['title']}
Required Skills: {job['required_skills']}
Department: {job['department']}
Location: {job['location']}
Openings: {job['openings']}
Experience Required: {job['experience_required']} years
""".strip()
        from langchain.schema import Document
        job_docs.append(Document(
            page_content=content,
            metadata={"job_id": job["job_id"], "title": job["title"],
                      "location": job["location"], "openings": job["openings"]}
        ))
    vs = FAISS.from_documents(job_docs, embeddings)
    vs.save_local("vectorstore_jobs")
    print(f"  ✅ Job vector store saved: {len(job_docs)} jobs")

def build_employee_vectorstore(embeddings):
    print("Building employee vector store")
    from langchain.schema import Document
    emp_docs = []
    for emp in EMPLOYEES:
        content = f"""
Employee: {emp['name']}
Designation: {emp['designation']}
Skills: {emp['skills']}
Experience: {emp['experience']} years
Location: {emp['location']}
Gender: {emp['gender']}
Process: {emp['process']}
""".strip()
        emp_docs.append(Document(
            page_content=content,
            metadata={"emp_id": emp["emp_id"], "name": emp["name"],
                      "designation": emp["designation"], "location": emp["location"],
                      "gender": emp["gender"], "status": emp["status"]}
        ))
    vs = FAISS.from_documents(emp_docs, embeddings)
    vs.save_local("vectorstore_employees")
    print(f"  ✅ Employee vector store saved: {len(emp_docs)} employees")


if __name__ == "__main__":
    print("\n🚀 Setting up RMG Chatbot \n")
    create_database()
    embeddings = get_embeddings()
    build_job_vectorstore(embeddings)
    build_employee_vectorstore(embeddings)
    print("\n✅ Setup complete!")