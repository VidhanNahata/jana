import uuid
from sqlalchemy import Column, String, Float, Integer, DateTime
from datetime import datetime, timezone
from backend.database import Base

class MasterTicket(Base):
    __tablename__ = "master_tickets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    canonical_description = Column(String, nullable=False)
    category = Column(String, nullable=False)
    department = Column(String, nullable=False)
    priority = Column(String, nullable=False)
    child_count = Column(Integer, default=1)
    centroid_lat = Column(Float, nullable=False)
    centroid_lon = Column(Float, nullable=False)
    status = Column(String, default="submitted")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
