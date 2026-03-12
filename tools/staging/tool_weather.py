"""
HUGINN_MANIFEST
tool_id:            tool.weather.v1
title:              Weather
capability_summary: >
  Get current weather conditions and a short forecast for any location.
  Uses wttr.in — no API key required. Read-only, safe for aug_call.
  Use when the user asks about weather, temperature, rain, or conditions.
polarity:           read
permission_scope:   []
inputs:
  location:     {type: string,  default: "",   description: "City name, postcode, or lat,lon. Empty = auto-detect by IP."}
  units:        {type: string,  default: "metric", description: "metric | imperial"}
outputs:
  location:         {type: string,  description: "Resolved location name"}
  condition:        {type: string,  description: "Current condition description"}
  temp_c:           {type: number,  description: "Current temperature in Celsius"}
  temp_f:           {type: number,  description: "Current temperature in Fahrenheit"}
  feels_like_c:     {type: number}
  humidity_pct:     {type: integer}
  wind_kph:         {type: number}
  wind_direction:   {type: string}
  forecast:         {type: array,   description: "Next 3 days: [{date, max_c, min_c, condition}]"}
  summary:          {type: string,  description: "One-sentence plain English summary for speech"}
dependencies:
  - requests>=2.28
perception_capable: false
handler:            handle
END_MANIFEST

Weather tool via wttr.in JSON API.

wttr.in is a free weather service with a JSON API that requires no
authentication. Queries are made over HTTPS with a 5-second timeout.

Location resolution:
  ""              — auto-detect from the server (IP geolocation)
  "London"        — city name
  "EC1A 1BB"      — UK postcode
  "51.5074,-0.1278" — lat,lon decimal degrees
  "~Eiffel Tower" — landmark name (tilde prefix)

The 'summary' field is pre-formatted for TTS — short, natural, no units
confusion. Example: "Currently 14°C with light rain. Highs of 16 tomorrow."

Units env override: HUGINN_WEATHER_UNITS=metric|imperial
"""

from __future__ import annotations

import os
from typing import Any

import requests


_WTTR_URL = "https://wttr.in/{location}?format=j1"
_TIMEOUT  = 8   # seconds


def handle(location: str = "", units: str = "metric") -> dict:
    """
    Fetch current weather and 3-day forecast from wttr.in.

    Returns a dict with current conditions, temperatures, wind, humidity,
    and a 3-entry forecast list. Always returns a 'summary' string suitable
    for TTS output.
    """
    units = os.environ.get("HUGINN_WEATHER_UNITS", units)
    loc   = location.strip()

    url = _WTTR_URL.format(location=requests.utils.quote(loc) if loc else "")

    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return _error_result(str(e))

    try:
        current  = data["current_condition"][0]
        area     = data.get("nearest_area", [{}])[0]
        area_name = (
            area.get("areaName", [{}])[0].get("value", "")
            or area.get("region",   [{}])[0].get("value", "")
            or loc or "your location"
        )
        country = area.get("country", [{}])[0].get("value", "")
        if country and country not in area_name:
            area_name = f"{area_name}, {country}"

        temp_c       = float(current["temp_C"])
        temp_f       = float(current["temp_F"])
        feels_c      = float(current["FeelsLikeC"])
        humidity     = int(current["humidity"])
        wind_kph     = float(current["windspeedKmph"])
        wind_dir_16  = current.get("winddir16Point", "")
        condition    = current.get("weatherDesc", [{}])[0].get("value", "")

        # 3-day forecast
        forecast = []
        for day in data.get("weather", [])[:3]:
            forecast.append({
                "date":      day.get("date", ""),
                "max_c":     float(day.get("maxtempC", 0)),
                "min_c":     float(day.get("mintempC", 0)),
                "max_f":     float(day.get("maxtempF", 0)),
                "min_f":     float(day.get("mintempF", 0)),
                "condition": day.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", ""),
            })

        summary = _build_summary(area_name, temp_c, temp_f, condition,
                                  forecast, units)

        return {
            "location":      area_name,
            "condition":     condition,
            "temp_c":        temp_c,
            "temp_f":        temp_f,
            "feels_like_c":  feels_c,
            "humidity_pct":  humidity,
            "wind_kph":      wind_kph,
            "wind_direction": wind_dir_16,
            "forecast":      forecast,
            "summary":       summary,
        }
    except (KeyError, IndexError, ValueError) as e:
        return _error_result(f"parse error: {e}")


def _build_summary(
    location: str,
    temp_c: float,
    temp_f: float,
    condition: str,
    forecast: list,
    units: str,
) -> str:
    if units == "imperial":
        temp_str = f"{temp_f:.0f}°F"
    else:
        temp_str = f"{temp_c:.0f}°C"

    cond_lower = condition.lower()
    summary    = f"Currently {temp_str} in {location}"
    if cond_lower:
        summary += f" with {cond_lower}"
    summary += "."

    if forecast:
        tom = forecast[0]
        if units == "imperial":
            hi = f"{tom['max_f']:.0f}°F"
        else:
            hi = f"{tom['max_c']:.0f}°C"
        tom_cond = tom["condition"].lower()
        summary += f" Tomorrow: highs of {hi}"
        if tom_cond:
            summary += f", {tom_cond}"
        summary += "."

    return summary


def _error_result(reason: str) -> dict:
    return {
        "location": "",
        "condition": "",
        "temp_c": 0.0,
        "temp_f": 0.0,
        "feels_like_c": 0.0,
        "humidity_pct": 0,
        "wind_kph": 0.0,
        "wind_direction": "",
        "forecast": [],
        "summary": f"I couldn't retrieve the weather right now ({reason}).",
    }
