# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# This creates a file named "resume_scans.db" in your main folder
SQLALCHEMY_DATABASE_URL = "sqlite:///./resume_scans.db"

# Create the database engine
# "check_same_thread": False is needed only for SQLite in FastAPI
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a session factory to talk to the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This Base class is what our database tables will inherit from
Base = declarative_base()