# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime
import datetime
from database import Base

class ScanRecord(Base):
    __tablename__ = "scan_records"

    # Define the columns for our table
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    ats_score = Column(Float)
    skills_found = Column(String)  # We will store skills as a comma-separated string
    scan_date = Column(DateTime, default=datetime.datetime.utcnow)