import uuid
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from backend.database import Base

class Complaint(Base):
    __tablename__ = "complaints"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    ticket_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    description = Column(String, nullable=False)
    images = Column(Text, nullable=False) # JSON array
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    location_source = Column(String, nullable=False)
    category = Column(String, nullable=False)
    department = Column(String, nullable=False)
    priority = Column(String, nullable=False)
    urgency = Column(Integer, nullable=False)
    severity = Column(Integer, nullable=False)
    status = Column(String, default="submitted")
    ai_analysis = Column(Text, nullable=False) # JSON
    is_authentic = Column(Boolean, default=True)
    master_ticket_id = Column(String, ForeignKey("master_tickets.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User")
    master_ticket = relationship("MasterTicket")
