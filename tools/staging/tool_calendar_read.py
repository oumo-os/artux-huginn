"""
HUGINN_MANIFEST
tool_id:            tool.calendar.read.v1
title:              Calendar Read
capability_summary: >
  Read upcoming calendar events from a local .ics file or a CalDAV URL.
  Use when the user asks about meetings, appointments, what's on today,
  what's coming up, or when their next event is. Read-only, safe for aug_call.
polarity:           read
permission_scope:   [calendar.read]
inputs:
  source:       {type: string, default: "",    description: "Path to .ics file or CalDAV URL. Falls back to HUGINN_CALENDAR_SOURCE env var."}
  days_ahead:   {type: integer, default: 7,    description: "How many days ahead to look"}
  max_events:   {type: integer, default: 20,   description: "Maximum events to return"}
  include_past: {type: boolean, default: false, description: "Include events that have already ended today"}
outputs:
  events:       {type: array,  description: "[{uid, title, start, end, location, description, all_day, calendar}]"}
  count:        {type: integer}
  next_event:   {type: object, description: "The very next upcoming event, or null"}
  summary:      {type: string, description: "Plain English summary for speech"}
dependencies:
  - icalendar>=5.0
  - requests>=2.28
perception_capable: false
handler:            handle
END_MANIFEST

Calendar read tool supporting .ics files and CalDAV.

Source resolution order:
  1. 'source' argument
  2. HUGINN_CALENDAR_SOURCE environment variable
  3. ~/Library/Calendars (macOS) — scans for exportable .ics files
  4. ~/.local/share/gnome-calendar (Linux)

Multiple sources:
  Set HUGINN_CALENDAR_SOURCES to a colon-separated list of paths/URLs.
  All are merged and deduplicated by UID.

CalDAV authentication:
  Set HUGINN_CALDAV_USERNAME and HUGINN_CALDAV_PASSWORD.
  Only used when the source URL uses http/https scheme.

The 'summary' field is formatted for TTS. Examples:
  "You have 3 events today. Next up: standup at 10 AM."
  "Nothing scheduled for today. Next event is dinner with Tom on Thursday at 7 PM."
"""

from __future__ import annotations

import os
import datetime
from typing import Any, Optional


def handle(
    source:       str  = "",
    days_ahead:   int  = 7,
    max_events:   int  = 20,
    include_past: bool = False,
) -> dict:
    """
    Read upcoming events from a calendar source.
    Returns structured events and a speech-ready summary.
    """
    from icalendar import Calendar
    import pytz

    sources = _resolve_sources(source)
    if not sources:
        return _empty("No calendar source configured. "
                      "Set HUGINN_CALENDAR_SOURCE to a .ics file path or URL.")

    now      = datetime.datetime.now(tz=datetime.timezone.utc)
    end_dt   = now + datetime.timedelta(days=days_ahead)
    events   = []
    seen_uid = set()

    for src in sources:
        try:
            cal_data = _fetch_source(src)
            cal      = Calendar.from_ical(cal_data)
            _extract_events(cal, now, end_dt, include_past,
                            events, seen_uid, src)
        except Exception as e:
            # Non-fatal: skip bad sources, surface in summary
            events.append({
                "uid": f"_error_{src[:30]}",
                "title": f"[Calendar error: {src[:40]}]",
                "start": "", "end": "", "location": "",
                "description": str(e), "all_day": False,
                "calendar": src,
            })

    events.sort(key=lambda e: e.get("start") or "")
    events = events[:max_events]

    next_event = next((e for e in events if not e["title"].startswith("[Calendar error")), None)
    summary    = _build_summary(events, next_event, now)

    return {
        "events":     events,
        "count":      len(events),
        "next_event": next_event,
        "summary":    summary,
    }


def _resolve_sources(source: str) -> list[str]:
    sources = []
    if source:
        sources.append(source)
    env_single = os.environ.get("HUGINN_CALENDAR_SOURCE", "")
    if env_single and env_single not in sources:
        sources.append(env_single)
    env_multi = os.environ.get("HUGINN_CALENDAR_SOURCES", "")
    for s in env_multi.split(":"):
        s = s.strip()
        if s and s not in sources:
            sources.append(s)
    if not sources:
        sources.extend(_discover_local_ics())
    return sources


def _discover_local_ics() -> list[str]:
    """Try standard calendar locations on macOS and Linux."""
    import glob, platform
    candidates = []
    if platform.system() == "Darwin":
        candidates = glob.glob(
            os.path.expanduser("~/Library/Calendars/**/*.ics"), recursive=True
        )[:5]
    else:
        base = os.path.expanduser("~/.local/share/gnome-calendar")
        if os.path.isdir(base):
            candidates = glob.glob(os.path.join(base, "**/*.ics"), recursive=True)[:5]
    return candidates


def _fetch_source(source: str) -> bytes:
    if source.startswith("http://") or source.startswith("https://"):
        import requests
        user = os.environ.get("HUGINN_CALDAV_USERNAME", "")
        pwd  = os.environ.get("HUGINN_CALDAV_PASSWORD", "")
        auth = (user, pwd) if user else None
        resp = requests.get(source, auth=auth, timeout=10)
        resp.raise_for_status()
        return resp.content
    with open(source, "rb") as f:
        return f.read()


def _extract_events(
    cal, now, end_dt, include_past, events, seen_uid, source_label
):
    from icalendar import Event as ICalEvent
    import pytz

    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        uid   = str(component.get("UID", ""))
        if uid in seen_uid:
            continue
        seen_uid.add(uid)

        title    = str(component.get("SUMMARY", "Untitled"))
        location = str(component.get("LOCATION", "") or "")
        desc     = str(component.get("DESCRIPTION", "") or "")[:200]

        dtstart  = component.get("DTSTART")
        dtend    = component.get("DTEND")
        if dtstart is None:
            continue

        start_val = dtstart.dt
        end_val   = dtend.dt if dtend else start_val

        all_day = isinstance(start_val, datetime.date) and \
                  not isinstance(start_val, datetime.datetime)

        # Normalise to aware datetimes
        if all_day:
            start_dt = datetime.datetime.combine(
                start_val, datetime.time.min,
                tzinfo=datetime.timezone.utc
            )
            end_dt_ev = datetime.datetime.combine(
                end_val, datetime.time.min,
                tzinfo=datetime.timezone.utc
            )
        else:
            start_dt  = _ensure_aware(start_val)
            end_dt_ev = _ensure_aware(end_val)

        if end_dt_ev < now and not include_past:
            continue
        if start_dt > end_dt:
            continue

        events.append({
            "uid":         uid,
            "title":       title,
            "start":       start_dt.isoformat(),
            "end":         end_dt_ev.isoformat(),
            "location":    location,
            "description": desc,
            "all_day":     all_day,
            "calendar":    source_label,
        })


def _ensure_aware(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def _build_summary(events: list, next_event: Optional[dict], now) -> str:
    today_events = [
        e for e in events
        if e.get("start", "").startswith(now.date().isoformat())
        and not e["title"].startswith("[Calendar error")
    ]

    if not events or all(e["title"].startswith("[Calendar error") for e in events):
        return "I couldn't read your calendar."

    parts = []
    if today_events:
        parts.append(
            f"You have {len(today_events)} event{'s' if len(today_events) != 1 else ''} today."
        )
    else:
        parts.append("Nothing scheduled for today.")

    if next_event:
        start_str = next_event.get("start", "")
        try:
            start_dt = datetime.datetime.fromisoformat(start_str)
            day_diff = (start_dt.date() - now.date()).days
            if day_diff == 0:
                time_str = start_dt.strftime("at %-I:%M %p").replace(" 0", " ")
                day_word = "today"
            elif day_diff == 1:
                time_str = start_dt.strftime("at %-I:%M %p")
                day_word = "tomorrow"
            else:
                time_str = start_dt.strftime("on %A at %-I:%M %p")
                day_word = ""
            loc = f" at {next_event['location']}" if next_event.get("location") else ""
            phrase = f"Next up: {next_event['title']}"
            if day_word:
                phrase += f" {day_word} {time_str}"
            else:
                phrase += f" {time_str}"
            phrase += f"{loc}."
            parts.append(phrase)
        except Exception:
            parts.append(f"Next up: {next_event['title']}.")

    return " ".join(parts)


def _empty(reason: str) -> dict:
    return {
        "events": [], "count": 0, "next_event": None,
        "summary": reason,
    }
