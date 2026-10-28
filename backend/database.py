import json
from datetime import datetime, timezone
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.config import settings
from backend.services.auth_service import get_password_hash

# Hosted Postgres providers often hand out postgres:// URLs, which SQLAlchemy 2 rejects
database_url = settings.DATABASE_URL.replace("postgres://", "postgresql://", 1)

if database_url.startswith("sqlite"):
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
else:
    # pre_ping recovers from serverless Postgres (e.g. Neon) dropping idle connections
    engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def seed_data():
    from backend.models import User, Complaint
    db = SessionLocal()
    try:
        # Check if admin exists
        admin = db.query(User).filter(User.phone_or_email == "admin@janasutra.in").first()
        if not admin:
            admin = User(
                id=str(uuid.uuid4()),
                phone_or_email="admin@janasutra.in",
                password_hash=get_password_hash(settings.ADMIN_PASSWORD),
                is_admin=True
            )
            db.add(admin)
            db.commit()

        # Check if we have complaints
        if db.query(Complaint).count() == 0:
            demo_user = db.query(User).filter(User.phone_or_email == "demo@janasutra.in").first()
            if not demo_user:
                demo_user = User(
                    id=str(uuid.uuid4()),
                    phone_or_email="demo@janasutra.in",
                    password_hash=get_password_hash(settings.DEMO_PASSWORD),
                    is_admin=False
                )
                db.add(demo_user)
                db.commit()
            
            # Add demo complaints
            complaints_data = [
                {"desc": "Huge pothole near Koramangala 4th Block Sony World signal", "cat": "pothole", "dept": "BBMP Roads", "pri": "P1", "lat": 12.9345, "lon": 77.6265, "status": "submitted"},
                {"desc": "Streetlight not working on 100ft road Indiranagar", "cat": "streetlight", "dept": "BESCOM", "pri": "P3", "lat": 12.9784, "lon": 77.6408, "status": "in_progress"},
                {"desc": "Garbage dumped near Whitefield tech park", "cat": "garbage", "dept": "BBMP SWM", "pri": "P2", "lat": 12.9698, "lon": 77.7499, "status": "resolved"},
                {"desc": "Drainage overflowing in Jayanagar 4th Block", "cat": "drainage", "dept": "BWSSB", "pri": "P1", "lat": 12.9298, "lon": 77.5833, "status": "submitted"},
                {"desc": "Traffic signal malfunctioning at MG Road", "cat": "traffic_signal", "dept": "Bangalore Traffic Police", "pri": "P1", "lat": 12.9738, "lon": 77.6119, "status": "submitted"},
                {"desc": "Tree branch fallen blocking road in Basavanagudi", "cat": "tree_fall", "dept": "BBMP Horticulture", "pri": "P2", "lat": 12.9406, "lon": 77.5738, "status": "in_progress"},
                {"desc": "Water pipe leaking near HSR Layout Sector 2", "cat": "water_leak", "dept": "BWSSB", "pri": "P2", "lat": 12.9081, "lon": 77.6476, "status": "submitted"},
                {"desc": "Deep pothole on Outer Ring Road Bellandur", "cat": "pothole", "dept": "BBMP Roads", "pri": "P1", "lat": 12.9304, "lon": 77.6784, "status": "submitted"},
                {"desc": "Sewage backup in Malleshwaram 8th Cross", "cat": "sewage", "dept": "BWSSB", "pri": "P1", "lat": 13.0034, "lon": 77.5706, "status": "submitted"},
                {"desc": "Broken pavement in CBD area", "cat": "road_damage", "dept": "BBMP Roads", "pri": "P3", "lat": 12.9716, "lon": 77.5946, "status": "resolved"},
                {"desc": "Live wire hanging in Banashankari", "cat": "electricity", "dept": "BESCOM", "pri": "P1", "lat": 12.9254, "lon": 77.5467, "status": "submitted"},
                {"desc": "Illegal garbage burning in Electronic City", "cat": "waste", "dept": "BBMP SWM", "pri": "P2", "lat": 12.8399, "lon": 77.6770, "status": "submitted"},
                {"desc": "Dead animal on the road in BTM Layout", "cat": "other", "dept": "BBMP General", "pri": "P2", "lat": 12.9165, "lon": 77.6101, "status": "submitted"},
                {"desc": "Pothole filled with water in Yelahanka", "cat": "pothole", "dept": "BBMP Roads", "pri": "P2", "lat": 13.1006, "lon": 77.5963, "status": "submitted"},
                {"desc": "Streetlight pole bent dangerously near Marathahalli", "cat": "streetlight", "dept": "BESCOM", "pri": "P1", "lat": 12.9569, "lon": 77.7011, "status": "submitted"}
            ]

            for i, c in enumerate(complaints_data):
                t_id = f"JS-2026-{str(i+1).zfill(5)}"
                complaint = Complaint(
                    id=str(uuid.uuid4()),
                    ticket_id=t_id,
                    user_id=demo_user.id,
                    description=c["desc"],
                    images="[]",
                    lat=c["lat"],
                    lon=c["lon"],
                    location_source="manual",
                    category=c["cat"],
                    department=c["dept"],
                    priority=c["pri"],
                    urgency=2,
                    severity=2,
                    status=c["status"],
                    ai_analysis="{}",
                    is_authentic=True,
                    master_ticket_id=None
                )
                db.add(complaint)
            db.commit()
    finally:
        db.close()
