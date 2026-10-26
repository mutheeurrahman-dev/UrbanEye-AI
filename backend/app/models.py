from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

CLASS_LIST = [
    "pothole",
    "garbage",
    "waterlogging",
    "broken_light",
    "crack",
    "road_damage",
]

DEPARTMENT_MAP = {
    "pothole": "Public Works Department",
    "road_damage": "Public Works Department",
    "crack": "Structural Engineering Cell",
    "garbage": "Sanitation Department",
    "waterlogging": "Stormwater/Drainage Department",
    "broken_light": "Electricity Board",
}


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String)
    category = Column(String, index=True)
    confidence = Column(Float, default=0.0)
    severity = Column(Float, default=0.0)
    latitude = Column(Float)
    longitude = Column(Float)
    ward = Column(String, default="Unassigned")
    department = Column(String)
    status = Column(String, default="reported")  # reported / assigned / resolved
    duplicate_of = Column(Integer, ForeignKey("reports.id"), nullable=True)
    phash = Column(String, nullable=True)  # perceptual hash, for duplicate detection
    needs_review = Column(Integer, default=0)  # 1 = low-confidence, flagged for manual review
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


def route_issue(category: str) -> str:
    return DEPARTMENT_MAP.get(category, "General Municipal Office")
