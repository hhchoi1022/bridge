from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Iterable
from astropy.time import Time
import numpy as np


@dataclass
class HistoricalPhotometryPoint:
    """Single historical photometry measurement."""

    obs_time: Time
    mag: Optional[float] = None
    magerr: Optional[float] = None
    filter_: Optional[str] = None
    system: str = "AB"
    depth_5sig: Optional[float] = None
    is_detection: bool = True
    is_upper_limit: bool = False
    note: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.obs_time, Time):
            self.obs_time = Time(self.obs_time)

        if self.mag is not None:
            self.mag = float(self.mag)
        if self.magerr is not None:
            self.magerr = float(self.magerr)
        if self.depth_5sig is not None:
            self.depth_5sig = float(self.depth_5sig)

        if self.filter_ is not None:
            self.filter_ = str(self.filter_)
        if self.system is not None:
            self.system = str(self.system)
        if self.note is not None:
            self.note = str(self.note)
        if self.source is not None:
            self.source = str(self.source)

        # safety checks
        if self.is_upper_limit:
            self.is_detection = False

        if self.magerr is not None and self.mag is None:
            raise ValueError("magerr is given but mag is None.")


@dataclass
class HistoricalPhotometry:
    """Container for historical photometry points."""

    points: List[HistoricalPhotometryPoint] = field(default_factory=list)

    def add(self, point: HistoricalPhotometryPoint) -> None:
        self.points.append(point)

    def extend(self, points: Iterable[HistoricalPhotometryPoint]) -> None:
        self.points.extend(points)

    def sort_by_time(self) -> None:
        self.points.sort(key=lambda p: p.obs_time.mjd)

    def filters(self) -> List[str]:
        return sorted({p.filter_ for p in self.points if p.filter_ is not None})

    def detections(self) -> List[HistoricalPhotometryPoint]:
        return [p for p in self.points if p.is_detection and p.mag is not None]

    def upper_limits(self) -> List[HistoricalPhotometryPoint]:
        return [p for p in self.points if p.is_upper_limit]

    def by_filter(self, filter_: str) -> List[HistoricalPhotometryPoint]:
        return [p for p in self.points if p.filter_ == filter_]

    def latest_detection(self, filter_: str | None = None) -> Optional[HistoricalPhotometryPoint]:
        pts = self.detections()
        if filter_ is not None:
            pts = [p for p in pts if p.filter_ == filter_]
        if not pts:
            return None
        return max(pts, key=lambda p: p.obs_time.mjd)

    def brightest_detection(self, filter_: str | None = None) -> Optional[HistoricalPhotometryPoint]:
        pts = self.detections()
        if filter_ is not None:
            pts = [p for p in pts if p.filter_ == filter_]
        if not pts:
            return None
        return min(pts, key=lambda p: p.mag)

    def to_dict_list(self):
        out = []
        for p in self.points:
            out.append({
                "obs_time": p.obs_time.isot,
                "mag": p.mag,
                "magerr": p.magerr,
                "filter_": p.filter_,
                "system": p.system,
                "depth_5sig": p.depth_5sig,
                "is_detection": p.is_detection,
                "is_upper_limit": p.is_upper_limit,
                "note": p.note,
                "source": p.source,
            })
        return out