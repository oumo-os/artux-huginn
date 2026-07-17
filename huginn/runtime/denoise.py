"""
runtime/denoise.py — Event denoising engine for Huginn.

Collapses consecutive events with identical content into range format,
preserving frequency information while reducing noise.

Design principles:
  - Denoise only new events since last ConsN update (not all events)
  - Simple heuristic: same source + event_type + normalized_data -> collapse
  - Output format: "ts.n since ts.n-x" (newest first)
  - ConsN already contains denoised narrative -- new events
    are denoised fresh, natural compression updates time in narrative
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class DenoisedEvent:
    """
    A denoised event with range information.

    Attributes
    ----------
    id : str
        Event ID (newest in the group)
    ts : str
        Timestamp (newest in the group)
    source : str
        Event source
    type : str
        Event type
    data : dict
        Event payload (newest)
    confidence : float
        Confidence score
    denoise_count : int
        Number of events collapsed (1 = no collapse)
    denoise_span_s : float
        Time span of collapsed events in seconds
    """
    id: str
    ts: str
    source: str
    type: str
    data: dict
    confidence: float = 1.0
    denoise_count: int = 1
    denoise_span_s: float = 0.0

    @property
    def payload(self) -> dict:
        """Alias for data -- compatibility with STMEvent interface."""
        return self.data

    @property
    def range_text(self) -> Optional[str]:
        """Human-readable range: 'ts.n since ts.n-x' or None if single event."""
        if self.denoise_count <= 1:
            return None
        try:
            newest = datetime.fromisoformat(self.ts.replace("Z", "+00:00"))
            oldest_ts = newest.timestamp() - self.denoise_span_s
            oldest = datetime.fromtimestamp(oldest_ts, tz=newest.tzinfo)
            return f"{self.ts} since {oldest.isoformat()}"
        except Exception:
            return f"x{self.denoise_count} events"


class DenoiseEngine:
    """
    Collapse consecutive events with same hash into range format.

    Only denoises events since last ConsN update.
    Older events in ConsN are already denoised -- natural
    compression updates time in narrative (e.g., "light on from
    10:00 to 12:00" becomes "light on from 10:00 to 12:05" when
    new identical event arrives).
    """

    def denoise(
        self,
        events: list,
        last_event_id: str = "",
    ) -> list[DenoisedEvent]:
        """
        Denoise events since last ConsN update.

        Parameters
        ----------
        events : list[STMEvent]
            All new events in STM
        last_event_id : str
            Last event ID in ConsN (boundary). Only events
            after this ID are denoised.

        Returns
        -------
        list[DenoisedEvent]
            Denoised events with range information
        """
        if not events:
            return []

        # Filter to events after ConsN boundary
        new_events = events
        if last_event_id:
            new_events = [e for e in events if e.id > last_event_id]

        if not new_events:
            return []

        # Group by consecutive hash
        groups = self._group_by_consecutive_hash(new_events)

        # Collapse each group
        return [self._collapse_group(g) for g in groups]

    def _group_by_consecutive_hash(self, events: list) -> list[list]:
        """
        Group consecutive events with same hash.

        Two events are considered duplicates if they share source, type,
        and normalized data content.
        """
        if not events:
            return []

        groups = []
        current_group = [events[0]]
        current_hash = self._event_hash(events[0])

        for event in events[1:]:
            event_hash = self._event_hash(event)
            if event_hash == current_hash:
                current_group.append(event)
            else:
                groups.append(current_group)
                current_group = [event]
                current_hash = event_hash

        groups.append(current_group)
        return groups

    def _event_hash(self, event) -> str:
        """
        Hash event for deduplication.

        Hashes source + type + normalized data (strips timestamps, IDs).
        """
        source = getattr(event, "source", "") or ""
        event_type = getattr(event, "type", "") or ""

        payload = getattr(event, "payload", None) or getattr(event, "data", None) or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {"raw": payload}

        normalized = self._normalize_event_data(payload)
        hash_input = f"{source}:{event_type}:{normalized}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _normalize_event_data(self, data: dict) -> str:
        """
        Normalize event data for hashing.

        Strips variable fields (timestamps, IDs) and keeps semantic content.
        """
        if not isinstance(data, dict):
            return str(data)

        skip_keys = {"timestamp", "id", "event_id", "ts", "created_at", "_denoised_count", "_denoised_span_s"}
        normalized = {k: v for k, v in data.items() if k not in skip_keys}
        return json.dumps(normalized, sort_keys=True, default=str)

    def _collapse_group(self, group: list) -> DenoisedEvent:
        """
        Collapse group of identical events into single event with range.

        If group has 1 event, returns it as-is with denoise_count=1.
        """
        if not group:
            raise ValueError("Cannot collapse empty group")

        if len(group) == 1:
            event = group[0]
            return DenoisedEvent(
                id=getattr(event, "id", ""),
                ts=getattr(event, "ts", ""),
                source=getattr(event, "source", ""),
                type=getattr(event, "type", ""),
                data=getattr(event, "payload", None) or getattr(event, "data", {}) or {},
                confidence=getattr(event, "confidence", 1.0),
                denoise_count=1,
                denoise_span_s=0.0,
            )

        newest = group[-1]
        oldest = group[0]

        span_s = 0.0
        try:
            newest_dt = datetime.fromisoformat(
                getattr(newest, "ts", "2000-01-01T00:00:00+00:00").replace("Z", "+00:00")
            )
            oldest_dt = datetime.fromisoformat(
                getattr(oldest, "ts", "2000-01-01T00:00:00+00:00").replace("Z", "+00:00")
            )
            span_s = (newest_dt - oldest_dt).total_seconds()
        except Exception:
            pass

        return DenoisedEvent(
            id=getattr(newest, "id", ""),
            ts=getattr(newest, "ts", ""),
            source=getattr(newest, "source", ""),
            type=getattr(newest, "type", ""),
            data=getattr(newest, "payload", None) or getattr(newest, "data", {}) or {},
            confidence=getattr(newest, "confidence", 1.0),
            denoise_count=len(group),
            denoise_span_s=span_s,
        )
