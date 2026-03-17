"""
Apex Black Box V40 - Database Layer
=====================================
SQLAlchemy 2.0 ORM models and DatabaseManager for persisting match data
and oracle scan results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

if TYPE_CHECKING:
    from apex_black_box.core import OracleOutput


# ---------------------------------------------------------------------------
# ORM Base & Models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base."""


class MatchModel(Base):
    """Persisted record for a single football match.

    Attributes
    ----------
    id:                 Auto-incremented primary key.
    home_team:          Home team name.
    away_team:          Away team name.
    match_date:         Scheduled kick-off datetime (UTC).
    score_ht:           Half-time score string (e.g. "1-0").
    score_ft:           Full-time score string (e.g. "2-1").
    pre_match_data_json: JSON-serialised PreMatchData dict.
    created_at:         Row creation timestamp (UTC).
    """

    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    home_team: Mapped[str] = mapped_column(String(120), nullable=False)
    away_team: Mapped[str] = mapped_column(String(120), nullable=False)
    match_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    score_ht: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    score_ft: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    pre_match_data_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # One match → many scans
    scans: Mapped[List["ScanModel"]] = relationship(
        "ScanModel",
        back_populates="match",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dictionary."""
        pre_data = None
        if self.pre_match_data_json:
            try:
                pre_data = json.loads(self.pre_match_data_json)
            except (json.JSONDecodeError, TypeError):
                pre_data = self.pre_match_data_json

        return {
            "id": self.id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "match_date": (
                self.match_date.isoformat() if self.match_date else None
            ),
            "score_ht": self.score_ht,
            "score_ft": self.score_ft,
            "pre_match_data": pre_data,
            "created_at": self.created_at.isoformat(),
        }


class ScanModel(Base):
    """A single oracle scan result linked to a match.

    Attributes
    ----------
    id:           Auto-incremented primary key.
    match_id:     Foreign key to :class:`MatchModel`.
    scan_time:    In-game minute when the scan was taken.
    probs_json:   JSON-serialised probability dict.
    confidence:   Oracle confidence score (0-100).
    vix:          Oracle volatility index (0-100).
    alerts_json:  JSON-serialised list of alert strings.
    raw_data_json: JSON-serialised full OracleOutput dict.
    created_at:   Row creation timestamp (UTC).
    """

    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False
    )
    scan_time: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    probs_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vix: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    alerts_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_data_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    match: Mapped["MatchModel"] = relationship("MatchModel", back_populates="scans")

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dictionary."""
        probs = None
        if self.probs_json:
            try:
                probs = json.loads(self.probs_json)
            except (json.JSONDecodeError, TypeError):
                probs = self.probs_json

        alerts = None
        if self.alerts_json:
            try:
                alerts = json.loads(self.alerts_json)
            except (json.JSONDecodeError, TypeError):
                alerts = self.alerts_json

        raw = None
        if self.raw_data_json:
            try:
                raw = json.loads(self.raw_data_json)
            except (json.JSONDecodeError, TypeError):
                raw = self.raw_data_json

        return {
            "id": self.id,
            "match_id": self.match_id,
            "scan_time": self.scan_time,
            "probs": probs,
            "confidence": self.confidence,
            "vix": self.vix,
            "alerts": alerts,
            "raw_data": raw,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Database Manager
# ---------------------------------------------------------------------------


class DatabaseManager:
    """High-level interface for all database operations.

    Parameters
    ----------
    db_url: SQLAlchemy connection URL. Defaults to a local SQLite file.
    """

    def __init__(self, db_url: str = "sqlite:///apex_black_box.db") -> None:
        self._db_url = db_url
        self._engine = create_engine(db_url, echo=False)

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        """Create all ORM tables if they do not already exist."""
        Base.metadata.create_all(self._engine)

    def drop_tables(self) -> None:
        """Drop all ORM tables (⚠️ destructive – use only in tests)."""
        Base.metadata.drop_all(self._engine)

    # ------------------------------------------------------------------
    # Match operations
    # ------------------------------------------------------------------

    def save_match(
        self,
        home: str,
        away: str,
        date: Optional[datetime],
        pre_data: Any,
    ) -> int:
        """Persist a new match record and return its generated id.

        Parameters
        ----------
        home:     Home team name.
        away:     Away team name.
        date:     Kick-off datetime (UTC-aware preferred, or None).
        pre_data: :class:`PreMatchData` instance or plain dict.
                  Serialised to JSON automatically.

        Returns
        -------
        int: Newly created match_id.
        """
        # Serialise pre_data to JSON
        if hasattr(pre_data, "__dict__"):
            pre_json = json.dumps(pre_data.__dict__)
        elif isinstance(pre_data, dict):
            pre_json = json.dumps(pre_data)
        else:
            pre_json = str(pre_data)

        with Session(self._engine) as session:
            match = MatchModel(
                home_team=home,
                away_team=away,
                match_date=date,
                pre_match_data_json=pre_json,
            )
            session.add(match)
            session.commit()
            session.refresh(match)
            return match.id  # type: ignore[return-value]

    def update_scores(
        self,
        match_id: int,
        score_ht: Optional[str] = None,
        score_ft: Optional[str] = None,
    ) -> bool:
        """Update half-time or full-time score strings for a match.

        Parameters
        ----------
        match_id: Target match primary key.
        score_ht: Half-time score string (e.g. "0-0"), or None to skip.
        score_ft: Full-time score string (e.g. "2-1"), or None to skip.

        Returns
        -------
        bool: True if the match was found and updated.
        """
        with Session(self._engine) as session:
            match = session.get(MatchModel, match_id)
            if match is None:
                return False
            if score_ht is not None:
                match.score_ht = score_ht
            if score_ft is not None:
                match.score_ft = score_ft
            session.commit()
            return True

    # ------------------------------------------------------------------
    # Scan operations
    # ------------------------------------------------------------------

    def save_scan(
        self,
        match_id: int,
        output: "OracleOutput",
        scan_time: Optional[int] = None,
    ) -> int:
        """Persist an oracle scan result and return its generated id.

        Parameters
        ----------
        match_id:  Foreign key to the associated match.
        output:    OracleOutput bundle from the engine.
        scan_time: Elapsed match minutes when the scan was taken (optional).

        Returns
        -------
        int: Newly created scan_id.
        """
        raw_dict: Dict[str, Any] = {
            "probs": output.probs,
            "confidence": output.confidence,
            "vix": output.vix,
            "lambdas": output.lambdas,
            "alerts": output.alerts,
        }

        with Session(self._engine) as session:
            scan = ScanModel(
                match_id=match_id,
                scan_time=scan_time,
                probs_json=json.dumps(output.probs),
                confidence=output.confidence,
                vix=output.vix,
                alerts_json=json.dumps(output.alerts),
                raw_data_json=json.dumps(raw_dict),
            )
            session.add(scan)
            session.commit()
            session.refresh(scan)
            return scan.id  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_recent_matches(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recently created match records.

        Parameters
        ----------
        limit: Maximum number of rows to return (default 20).

        Returns
        -------
        List[Dict[str, Any]]: List of match dicts ordered by created_at desc.
        """
        with Session(self._engine) as session:
            stmt = (
                select(MatchModel)
                .order_by(MatchModel.created_at.desc())
                .limit(limit)
            )
            rows = session.scalars(stmt).all()
            return [row.to_dict() for row in rows]

    def get_scans_for_match(self, match_id: int) -> List[Dict[str, Any]]:
        """Return all oracle scans for a given match.

        Parameters
        ----------
        match_id: The match primary key.

        Returns
        -------
        List[Dict[str, Any]]: Scan dicts ordered by scan_time ascending.
        """
        with Session(self._engine) as session:
            stmt = (
                select(ScanModel)
                .where(ScanModel.match_id == match_id)
                .order_by(ScanModel.scan_time.asc(), ScanModel.created_at.asc())
            )
            rows = session.scalars(stmt).all()
            return [row.to_dict() for row in rows]

    def get_match(self, match_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single match by primary key.

        Parameters
        ----------
        match_id: The match primary key.

        Returns
        -------
        Optional[Dict[str, Any]]: Match dict or None if not found.
        """
        with Session(self._engine) as session:
            match = session.get(MatchModel, match_id)
            return match.to_dict() if match else None
