import os
import shutil
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import engine, get_db
from app.models import Base, Report, route_issue, CLASS_LIST
from app.ml.inference import run_inference, severity_score, CONFIDENCE_THRESHOLD
from app.ml.duplicates import compute_phash, find_duplicate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="UrbanEye AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/api/health")
def health():
    return {"status": "ok", "classes": CLASS_LIST}


@app.post("/api/report")
async def create_report(
    image: UploadFile = File(...),
    lat: float = Form(...),
    lng: float = Form(...),
    ward: Optional[str] = Form("Unassigned"),
    db: Session = Depends(get_db),
):
    # 1. save image
    ext = os.path.splitext(image.filename)[1] or ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # 2. run inference
    result = run_inference(fpath)
    category, confidence, box_area_ratio = (
        result["category"], result["confidence"], result["box_area_ratio"],
    )
    needs_review = 1 if confidence < CONFIDENCE_THRESHOLD else 0

    # 3. severity
    severity = severity_score(box_area_ratio, category)

    # 4. duplicate check
    phash = compute_phash(fpath)
    existing = db.query(Report).filter(Report.category == category).all()
    dup = find_duplicate(existing, phash, category, lat, lng)

    # 5. routing
    department = route_issue(category)

    report = Report(
        image_path=f"/uploads/{fname}",
        category=category,
        confidence=confidence,
        severity=severity,
        latitude=lat,
        longitude=lng,
        ward=ward,
        department=department,
        status="reported",
        duplicate_of=dup.id if dup else None,
        phash=phash,
        needs_review=needs_review,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "id": report.id,
        "category": category,
        "confidence": confidence,
        "severity": severity,
        "department": department,
        "needs_review": bool(needs_review),
        "duplicate_of": dup.id if dup else None,
        "duplicate_notice": (
            f"This looks like a duplicate of report #{dup.id} nearby." if dup else None
        ),
        "image_path": report.image_path,
        "latitude": lat,
        "longitude": lng,
        "status": report.status,
    }


@app.get("/api/reports")
def list_reports(
    status: Optional[str] = None,
    category: Optional[str] = None,
    department: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(Report)
    if status:
        q = q.filter(Report.status == status)
    if category:
        q = q.filter(Report.category == category)
    if department:
        q = q.filter(Report.department == department)
    reports = q.order_by(Report.created_at.desc()).all()
    return [_serialize(r) for r in reports]


@app.get("/api/reports/{report_id}")
def get_report(report_id: int, db: Session = Depends(get_db)):
    r = db.query(Report).filter(Report.id == report_id).first()
    if not r:
        raise HTTPException(404, "Report not found")
    return _serialize(r)


@app.patch("/api/reports/{report_id}/status")
def update_status(report_id: int, status: str = Form(...), db: Session = Depends(get_db)):
    r = db.query(Report).filter(Report.id == report_id).first()
    if not r:
        raise HTTPException(404, "Report not found")
    if status not in ("reported", "assigned", "resolved"):
        raise HTTPException(400, "Invalid status")
    r.status = status
    if status == "resolved":
        r.resolved_at = datetime.utcnow()
    db.commit()
    return _serialize(r)


@app.get("/api/analytics/summary")
def analytics_summary(db: Session = Depends(get_db)):
    total = db.query(Report).count()
    by_category = dict(
        db.query(Report.category, func.count(Report.id)).group_by(Report.category).all()
    )
    by_department = dict(
        db.query(Report.department, func.count(Report.id)).group_by(Report.department).all()
    )
    by_status = dict(
        db.query(Report.status, func.count(Report.id)).group_by(Report.status).all()
    )
    by_ward = dict(
        db.query(Report.ward, func.count(Report.id)).group_by(Report.ward).all()
    )

    resolved = db.query(Report).filter(Report.status == "resolved", Report.resolved_at.isnot(None)).all()
    if resolved:
        avg_resolution_hours = sum(
            (r.resolved_at - r.created_at).total_seconds() for r in resolved
        ) / len(resolved) / 3600
    else:
        avg_resolution_hours = None

    return {
        "total_reports": total,
        "by_category": by_category,
        "by_department": by_department,
        "by_status": by_status,
        "by_ward": by_ward,
        "avg_resolution_hours": round(avg_resolution_hours, 2) if avg_resolution_hours else None,
    }


# Mounted LAST and deliberately after every /api/* route above: Starlette
# matches routes in registration order, and a "/" mount matches every path
# by prefix, so it must never be registered before the API routes or it
# would shadow them.
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def _serialize(r: Report) -> dict:
    return {
        "id": r.id,
        "image_path": r.image_path,
        "category": r.category,
        "confidence": r.confidence,
        "severity": r.severity,
        "latitude": r.latitude,
        "longitude": r.longitude,
        "ward": r.ward,
        "department": r.department,
        "status": r.status,
        "duplicate_of": r.duplicate_of,
        "needs_review": bool(r.needs_review),
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
    }
