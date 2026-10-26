"""Seed the DB with ~35 synthetic historical reports so the map/analytics
dashboard looks populated on first load, per the dev guide's Hour 26-33 advice.
Run from backend/: python seed.py
"""
import random
from datetime import datetime, timedelta
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import engine, SessionLocal
from app.models import Base, Report, route_issue, CLASS_LIST

Base.metadata.create_all(bind=engine)
db = SessionLocal()

# Trichy-ish coordinates so the demo map centers somewhere real
CENTER_LAT, CENTER_LNG = 10.7905, 78.7047
WARDS = ["Ward 3 - Srirangam", "Ward 7 - Cantonment", "Ward 12 - Thillai Nagar",
         "Ward 15 - K.K. Nagar", "Ward 21 - Woraiyur"]
STATUSES = ["reported", "reported", "assigned", "assigned", "resolved"]

random.seed(42)
for i in range(35):
    category = random.choice(CLASS_LIST)
    lat = CENTER_LAT + random.uniform(-0.03, 0.03)
    lng = CENTER_LNG + random.uniform(-0.03, 0.03)
    severity = round(random.uniform(0.2, 0.95), 2)
    status = random.choice(STATUSES)
    created = datetime.utcnow() - timedelta(hours=random.randint(1, 240))
    resolved = created + timedelta(hours=random.randint(4, 72)) if status == "resolved" else None

    r = Report(
        image_path="/uploads/placeholder.jpg",
        category=category,
        confidence=round(random.uniform(0.6, 0.97), 2),
        severity=severity,
        latitude=lat,
        longitude=lng,
        ward=random.choice(WARDS),
        department=route_issue(category),
        status=status,
        phash=None,
        needs_review=0,
        created_at=created,
        resolved_at=resolved,
    )
    db.add(r)

db.commit()
print(f"Seeded {db.query(Report).count()} total reports.")
db.close()
