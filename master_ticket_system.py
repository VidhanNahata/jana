import os
import math
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()

class MasterTicket(Base):
    """
    Central Database structure capturing massive compounded civic hazards dynamically.
    Instead of flooding city dashboards with redundant issues, duplicate issues are 
    funneled to update a single underlying MasterTicket entity.
    """
    __tablename__ = 'master_tickets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(50), nullable=False)
    base_description = Column(Text, nullable=False)  # The text of the initial complaint formulating the ticket
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # Dynamic Aggregation Statistics
    report_count = Column(Integer, default=1)
    priority_score = Column(Float, default=1.0)      # Baseline calculated risk urgency
    status = Column(String(50), default="Open")
    
    # Store aggregated reporter IDs securely spanning a dynamic array
    # Note: On native PostgreSQL, switching this structurally to JSONB natively optimizes massive queries
    reporter_ids = Column(JSON, default=list) 
    
    # Time Tracking Metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Structural DB relational mapping
    child_complaints = relationship("ChildComplaint", back_populates="master_ticket", cascade="all, delete-orphan")


class ChildComplaint(Base):
    """
    Forensic tracking table retaining EXACT raw copies of all duplicate reports 
    ever mapped into the overarching MasterTicket structure.
    """
    __tablename__ = 'child_complaints'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    master_ticket_id = Column(Integer, ForeignKey('master_tickets.id'), nullable=False)
    original_text = Column(Text, nullable=False)
    reporter_id = Column(String(100), nullable=False)
    
    # IndicSBERT Cosine Alignment Score (Retained mapping why the AI locked this into the specific MasterTicket)
    similarity_score_to_master = Column(Float, nullable=True) 
    
    reported_at = Column(DateTime, default=datetime.utcnow)
    
    master_ticket = relationship("MasterTicket", back_populates="child_complaints")


def init_db(db_url):
    """Safely initialize SQL bindings."""
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return engine, sessionmaker(autocommit=False, autoflush=False, bind=engine)

def calculate_priority(report_count, base_severity=1.0):
    """
    Function compounding priority urgency logarithmically.
    Every overlapping unique report structurally raises the visibility score, 
    alerting city administrators instantly to dense hazard areas.
    """
    # Severity multiplies structurally as (1 + log(report_count) * scaling_weight)
    urgency_multiplier = 1.0 + (math.log1p(report_count - 1) * 0.5) 
    return round(base_severity * urgency_multiplier, 2)


def merge_or_create_ticket(SessionLocal, new_complaint_dict, is_duplicate=False, existing_master_id=None, similarity_score=None):
    """
    Handles dynamically routing and scaling a new incoming payload into the architecture.
    """
    session = SessionLocal()
    try:
        new_reporter = new_complaint_dict['reporter_id']
        base_severity = new_complaint_dict.get('base_severity', 1.0)
        
        if is_duplicate and existing_master_id:
            # --- MERGE SCENARIO ---
            # 1. Access overarching Master Ticket
            master = session.query(MasterTicket).filter(MasterTicket.id == existing_master_id).first()
            if not master:
                raise ValueError(f"CRITICAL: MasterTicket {existing_master_id} boundary lost.")
                
            # 2. Increment active compound count
            master.report_count += 1
            
            # 3. Aggregate user network IDs utilizing strict array lists
            # We strictly instantiate a new underlying object map to ensure SQLAlchemy detects the JSON structural edit natively
            current_reporters = list(master.reporter_ids) if master.reporter_ids else []
            if new_reporter not in current_reporters:
                current_reporters.append(new_reporter)
            master.reporter_ids = current_reporters
            
            # 4. Trigger Dynamic Urgency Spike (Based on quantity of identical overlapping reports)
            master.priority_score = calculate_priority(master.report_count, base_severity=base_severity)
            
            # 5. Lock raw duplicate payload into Child Table
            child = ChildComplaint(
                master_ticket_id=master.id,
                original_text=new_complaint_dict['text'],
                reporter_id=new_reporter,
                similarity_score_to_master=similarity_score
            )
            session.add(child)
            session.commit()
            
            print(f"[Deduplication] Compounded active duplicate into MasterTicket #{master.id}. Urgency Priority spied to: {master.priority_score}")
            return master.id
            
        else:
            # --- CREATION SCENARIO ---
            # Generate entirely isolated standalone Master representation
            master = MasterTicket(
                category=new_complaint_dict['category'],
                base_description=new_complaint_dict['text'],
                latitude=new_complaint_dict['latitude'],
                longitude=new_complaint_dict['longitude'],
                report_count=1,
                reporter_ids=[new_reporter],
                priority_score=calculate_priority(1, base_severity)
            )
            session.add(master)
            session.commit() 
            
            # Instantiate intrinsic baseline payload locally
            child = ChildComplaint(
                master_ticket_id=master.id,
                original_text=new_complaint_dict['text'],
                reporter_id=new_reporter,
                similarity_score_to_master=1.0 # 100% boundary match mapped identically against itself intrinsically 
            )
            session.add(child)
            session.commit()
            
            print(f"[Instantiation] Successfully created novel MasterTicket #{master.id} mapped to priority {master.priority_score}")
            return master.id
            
    except Exception as e:
        session.rollback()
        print(f"SQL Execution Failure Bounds: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    print("Initializing SQLAlchemy Relational Master Ticket Engine...")
    
    # Provide strict production PostgreSQL bounds strings
    # PROD_DB_URL = "postgresql+psycopg2://admin_user:secret123@localhost:5432/civic_db"
    
    # We map seamlessly toward memory-bound SQLite dynamically bypassing complex external container execution failures for immediate demonstration logic
    TEST_DB_URL = "sqlite:///:memory:"
    engine, SessionLocal = init_db(TEST_DB_URL)
    
    # Scenario 1: User physically submits a generic pothole complaint 
    complaint_1 = {
        'reporter_id': 'USR_A001',
        'text': 'A terrifyingly massive pothole completely tore apart my tire on MG Road!',
        'category': 'Roads',
        'latitude': 18.5204,
        'longitude': 73.8567,
        'base_severity': 2.0
    }
    print("\n--- Phase 1: Submitting Isolated Complaint ---")
    master_id = merge_or_create_ticket(SessionLocal, complaint_1, is_duplicate=False)
    
    # Scenario 2: Days later, the IndicSBERT NLP architecture mathematically tags a Hindi code-mixed duplicate 
    complaint_2 = {
        'reporter_id': 'USR_B882',
        'text': 'सड़क पर बहुत बड़ा गड्ढा है जो दुर्घटनाओं का कारण बन रहा है।', 
        'category': 'Roads',
        'latitude': 18.5205, # Distance offset ~50 meters intrinsically mapped
        'longitude': 73.8568,
        'base_severity': 2.0 
    }
    print("\n--- Phase 2: Intercepting Cross-Lingual Overlapping Duplicate ---")
    merge_or_create_ticket(
        SessionLocal, 
        complaint_2, 
        is_duplicate=True, 
        existing_master_id=master_id, 
        similarity_score=0.91 # Score inherently pushed backward by our external 'indicsbert_deduplication' boundaries
    )
    
    # Validate entire SQLAlchemy Data State mappings
    session = SessionLocal()
    master = session.query(MasterTicket).first()
    children = session.query(ChildComplaint).all()
    
    print("\n--- Final Aggregate Analytical Ticket State Evaluator ---")
    print(f"MasterTicket Active ID: {master.id}")
    print(f"Core Hazard Sub-Category: {master.category}")
    print(f"Compounded Total Network Reports: {master.report_count}")
    print(f"Algorithmically Escalated Priority Score: {master.priority_score}")
    print(f"Tracked Associated Reporter DB Arrays: {master.reporter_ids}")
    
    print("\n--- Nested Inherited Child Records Mapping ---")
    for child in children:
        print(f" -> Child Tracking Row #{child.id} | Cosine Similarity Matrix Value: {child.similarity_score_to_master} | Original Output: '{child.original_text}'")
    
    session.close()
