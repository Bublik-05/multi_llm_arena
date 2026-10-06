from datetime import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ModelRecord(Base):
    __tablename__ = "models"

    name = Column(String, primary_key=True)
    provider = Column(String, nullable=False)
    elo = Column(Float, default=1200.0, nullable=False)
    wins = Column(Integer, default=0, nullable=False)
    losses = Column(Integer, default=0, nullable=False)
    errors = Column(Integer, default=0, nullable=False)
    score_sum = Column(Float, default=0.0, nullable=False)
    latency_sum = Column(Float, default=0.0, nullable=False)
    evaluations_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    judge_model = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    responses = relationship("Response", back_populates="evaluation", cascade="all, delete-orphan")


class Response(Base):
    __tablename__ = "responses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id"), nullable=False)
    model_name = Column(String, ForeignKey("models.name"), nullable=False)
    text = Column(Text, nullable=False)
    latency = Column(Float, nullable=False)
    label = Column(String, nullable=False)  # A, B, C...
    correctness = Column(Float, nullable=True)
    clarity = Column(Float, nullable=True)
    conciseness = Column(Float, nullable=True)
    relevance = Column(Float, nullable=True)
    overall = Column(Float, nullable=True)
    is_winner = Column(Integer, default=0, nullable=False)
    judge_raw = Column(JSON, nullable=True)

    evaluation = relationship("Evaluation", back_populates="responses")


class RatingHistory(Base):
    __tablename__ = "rating_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, ForeignKey("models.name"), nullable=False)
    elo = Column(Float, nullable=False)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)