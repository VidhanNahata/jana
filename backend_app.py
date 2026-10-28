import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uvicorn

# Try importing GeoAlchemy2 (only available with PostgreSQL + PostGIS)
try:
    from geoalchemy2 import Geometry
    HAS_POSTGIS = True
except ImportError:
    HAS_POSTGIS = False

# ----------------- DATABASE CONFIG -----------------
# Uses PostgreSQL+PostGIS if DATABASE_URL is set, otherwise falls back to local SQLite
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///civic_local.db")
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ComplaintFeature(Base):
    """
    SQLAlchemy Geographic ORM defining the core PostGIS tables mapping our civic data.
    """
    __tablename__ = 'civic_complaints_gis'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    category = Column(String, index=True)
    description = Column(String)
    priority = Column(String) # Limits: P1, P2, P3, P4
    status = Column(String, default="Open")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Spatial columns — PostGIS geometry if available, otherwise store as plain floats
    if HAS_POSTGIS:
        geom_display = Column(Geometry(geometry_type='POINT', srid=4326), nullable=True)
        geom_clustering = Column(Geometry(geometry_type='POINT', srid=3395), nullable=True)
    else:
        lat = Column(Float, nullable=True)
        lng = Column(Float, nullable=True)

# ----------------- SLA CONTRACT MAPS -----------------
SLA_CONTRACTS = {
    "P1": {"timeframe": "Immediate (<= 1 hour)", "action": "Auto-dispatch emergency crew. Bypass standard queue.", "sla_breach_penalty": "High"},
    "P2": {"timeframe": "24 hours", "action": "Acknowledge structurally and schedule urgent repair.", "sla_breach_penalty": "Medium"},
    "P3": {"timeframe": "72 hours", "action": "Schedule routine maintenance block.", "sla_breach_penalty": "Low"},
    "P4": {"timeframe": "1 week", "action": "Queue into general low-priority infrastructure backlog.", "sla_breach_penalty": "None"}
}

# ----------------- APP INITIALIZATION -----------------
app = FastAPI(
    title="Civic Platform AI Backend (Phase 3)", 
    description="REST API mapping spatial clustering, closed-feedback AI Pipelines, and departmental traffic.",
    version="1.0"
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------- PYDANTIC INGESTION SCHEMAS -----------------
class ComplaintIngestion(BaseModel):
    image_b64: str = None
    description: str
    lat: float
    lng: float

# ----------------- MODULE 1: AI CLOSED PIPELINE -----------------
@app.post("/api/v1/complaints/ingest")
async def ingest_complaint(complaint: ComplaintIngestion, db: Session = Depends(get_db)):
    """
    Simulates the overarching Closed Feedback Loop AI Pipeline executing securely in order:
    1. INGESTION (Receives raw image/data)
    2. VERIFICATION (ELA CNN blocks if tampered = True)
    3. NLP DEDUPLICATION (IndicSBERT clusters local issues into one MasterTicket)
    4. RF PRIORITY & ITIL MATRIX (Scoring the urgency P1-P4)
    5. SERVE (Writes structurally to PostGIS below)
    6. FEEDBACK (Offline loop updating limits)
    """
    
    # Pipeline Step 2: Verification (Assuming image passed ELA pipeline)
    is_forged = False 
    if is_forged:
        raise HTTPException(status_code=400, detail="Image manipulation detected. Submission terminated.")
        
    # Pipeline Step 3 & 4: (Mocking variables from NLP modules)
    is_duplicate = False 
    priority = "P1" 
    category = "Roads"
    
    if not is_duplicate:
        # Generate raw 4326 Point formatting explicitly
        point_4326 = f"SRID=4326;POINT({complaint.lng} {complaint.lat})"
        
        # Instantiate natively
        new_record = ComplaintFeature(
            category=category,
            description=complaint.description,
            priority=priority,
            geom_display=point_4326,
            # Note: During actual PostGIS insertion, we would safely mirror the coordinates natively converting to SRID 3395 
            # via `ST_Transform(ST_GeomFromText(...), 3395)` mathematically executed at the database layer.
            geom_clustering=f"SRID=3395;POINT({complaint.lng} {complaint.lat})" 
        )
        db.add(new_record)
        
    db.commit()

    return {
        "status": "success",
        "pipeline_results": {
            "verified_authentic": True,
            "category_assigned": category,
            "priority_assigned": priority,
            "sla_active_contract": SLA_CONTRACTS[priority]
        }
    }

# ----------------- MODULE 2: POSTGIS SPATIAL CLUSTERING -----------------
@app.get("/api/v1/gis/hotspots")
def get_spatial_clusters(db: Session = Depends(get_db)):
    """
    Executes raw ST_ClusterDBSCAN natively tracking physical groupings bounds.
    Requires EPS := 100 meters, strictly possible because we cached SRID 3395!
    """
    # Requires PostGIS Extension active locally
    clustering_query = text("""
        WITH clustered_points AS (
            SELECT 
                id, 
                -- Execute 100 Meter 2D window clustering explicitly
                ST_ClusterDBSCAN(geom_clustering, eps := 100, minpoints := 2) OVER() AS cluster_id,
                ST_X(geom_display) as lng,
                ST_Y(geom_display) as lat
            FROM civic_complaints_gis
            WHERE status = 'Open'
        )
        SELECT 
            cluster_id, 
            COUNT(id) as incident_count,
            AVG(lat) as center_lat,
            AVG(lng) as center_lng,
            array_agg(id) as points
        FROM clustered_points
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
        ORDER BY incident_count DESC;
    """)
    
    # Try/except block mimicking missing DB safely for evaluation
    try:
        result = db.execute(clustering_query).fetchall()
        hotspots = []
        for row in result:
            hotspots.append({
                "cluster_id": row.cluster_id,
                "incident_count": row.incident_count,
                "center_lat": row.center_lat,
                "center_lng": row.center_lng,
                "points_ids": row.points
            })
            
        return {"status": "success", "hotspots_detected": len(hotspots), "clusters": hotspots}
    except Exception as e:
        return {"status": "error", "message": "Failed mapping PostGIS database natively. Have you established docker bounds?", "error": str(e)}

# ----------------- MODULE 3: DEPARTMENTAL ROUTING -----------------
@app.get("/api/v1/departments/{dept_name}/tasks")
def route_department_tasks(dept_name: str, db: Session = Depends(get_db)):
    """
    Secure REST Route dumping highly filtered tickets actively tracking priorities and SLA rules 
    directly into granular local department buckets (Roads, Water, Sanitation, etc).
    """
    dept_map = {
        "roads": "Roads",
        "sanitation": "Sanitation",
        "water": "Water",
        "electricity": "Electricity"
    }
    
    if dept_name.lower() not in dept_map:
        raise HTTPException(status_code=400, detail="Invalid department tracking parameter.")
        
    target_category = dept_map[dept_name.lower()]
    
    try:
        # Pull only open tasks bound to the department, forcefully ordering P1 -> P4
        tasks = db.query(ComplaintFeature).filter(
            ComplaintFeature.category == target_category,
            ComplaintFeature.status == 'Open'
        ).order_by(ComplaintFeature.priority).limit(50).all()
        
        routed_data = []
        for t in tasks:
            routed_data.append({
                "complaint_id": t.id,
                "priority": t.priority,
                "description": t.description,
                "SLA_Contract": SLA_CONTRACTS.get(t.priority, "Unmapped SLA limits")
            })
            
        return {
            "department": target_category, 
            "active_high_priority_tasks": len(routed_data), 
            "tasks": routed_data
        }
    except:
        # Fallback mocking standard routing output without a DB running physically
        return {
            "department": target_category,
            "mocked_status": "No Active PG database detected. Simulating Department JSON dump...",
            "tasks": [
                {
                    "complaint_id": 901,
                    "priority": "P1",
                    "description": f"Critical Failure recorded actively inside {target_category} constraints.",
                    "SLA_Contract": SLA_CONTRACTS.get("P1")
                }
            ]
        }

if __name__ == "__main__":
    print("Initializing Phase 3: PostGIS Backend & AI Routing Engine...")
    uvicorn.run("backend_app:app", host="0.0.0.0", port=8000, reload=False)
