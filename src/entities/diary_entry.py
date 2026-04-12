"""Diary Entry entity."""
from sqlalchemy import Column, Integer, Text, DateTime, BigInteger
from datetime import datetime, timezone
from .base import Base

class DiaryEntry(Base):
    """Rappresenta una memoria salvata da Ahri per un utente."""
    __tablename__ = 'diary_entry'

    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    memory_text = Column(Text, nullable=False)
    date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
