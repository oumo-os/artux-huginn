"""
HUGINN_MANIFEST
tool_id:            tool.web.search.v1
title:              Web Search
capability_summary: >
  Search the web for current information using DuckDuckGo. No API key required.
  Returns a concise answer (when available) plus top organic results with titles,
  URLs, and snippets. Use when the user asks about recent events, facts, or
  anything that may have changed since Artux's last LTM update.
polarity:           read
permission_scope:   []
inputs:
  query:        {type: string,  description: "Search query"}
  max_results:  {type: integer, default: 5,    description: "Max organic results to return (1-10)"}
  region:       {type: string,  default: "wt-wt", description: "DuckDuckGo region code, e.g. gb-en, us-en"}
  safe_search:  {type: string,  default: "moderate", description: "off | moderate | strict"}
outputs:
  answer:       {type: string,  description: "Instant answer if DuckDuckGo has one, else empty"}
  answer_type:  {type: string,  description: "DuckDuckGo answer type: D=disambiguation, A=article, etc."}
  results:      {type: array,   description: "[{title, url, snippet}]"}
  count:        {type: integer}
  query_used:   {type: string}
  summary:      {type: string,  description: "One-paragraph synthesis for Sagax to speak or use in reasoning"}
dependencies:
  - requests>=2.28
perception_capable: false
handler:            handle
END_MANIFEST

Web search via DuckDuckGo Instant Answer API + HTML scrape fallback.

DuckDuckGo provides a free JSON API at api.duckduckgo.com that returns
instant answers (definitions, calculations, entity summaries) without
requiring authentication. For broader web results, the tool optionally
scrapes the HTML search results page using the lite endpoint.

Rate limits: DuckDuckGo is generous for low-volume use. Do not loop calls.
Use max_results=3 for quick lookups, max_results=8 for research tasks.

Privacy: DuckDuckGo does not track searches or link queries to users.
No API key is needed and no account is created.

Environment overrides:
  HUGINN_SEARCH_REGION=wt-wt
  HUGINN_SEARCH_SAFE=moderate
  HUGINN_SEARCH_MAX=5
"""

from __future__ import annotations

import os
import html
import re
import time
from typing import Any

import requests

_DDG_API   = "https://api.duckduckgo.com/"
_DDG_LITE  = "https://lite.duckduckgo.com/lite/"
_HEADERS   = {
    "User-Agent": "Mozilla/5.0 (compatible; Huginn/0.1; +https://github.com/oumo-os/artux-huginn)"
}
_TIMEOUT   = 8


def handle(
    query:       str = "",
    max_results: int = 5,
    region:      str = "wt-wt",
    safe_search: str = "moderate",
) -> dict:
    query       = query.strip()
    max_results = max(1, min(10, int(os.environ.get("HUGINN_SEARCH_MAX", max_results))))
    region      = os.environ.get("HUGINN_SEARCH_REGION", region)
    safe_search = os.environ.get("HUGINN_SEARCH_SAFE",   safe_search)

    if not query:
        return _empty("", "No query provided.")

    # Step 1: Instant Answer API
    answer, answer_type, ia_results = _instant_answer(query, region)

    # Step 2: Lite HTML results (if instant answer is thin)
    html_results = []
    if len(ia_results) < max_results:
        html_results = _lite_results(query, region, safe_search,
                                     max_results - len(ia_results))

    results = _deduplicate(ia_results + html_results)[:max_results]
    summary = _build_summary(query, answer, results)

    return {
        "answer":      answer,
        "answer_type": answer_type,
        "results":     results,
        "count":       len(results),
        "query_used":  query,
        "summary":     summary,
    }


def _instant_answer(query: str, region: str):
    """Call DuckDuckGo JSON API. Returns (answer, answer_type, results_list)."""
    try:
        resp = requests.get(
            _DDG_API,
            params={
                "q":      query,
                "format": "json",
                "kl":     region,
                "no_html": "1",
                "skip_disambig": "0",
            },
            headers = _HEADERS,
            timeout = _TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return "", "", []

    answer      = data.get("AbstractText", "").strip()
    answer_type = data.get("Type", "")

    # Related topics as results
    results = []
    for topic in data.get("RelatedTopics", []):
        if "Topics" in topic:   # nested group — skip
            continue
        url  = topic.get("FirstURL", "")
        text = topic.get("Text",     "")
        if url and text:
            results.append({
                "title":   text.split(" - ")[0][:80],
                "url":     url,
                "snippet": text[:200],
            })

    return answer, answer_type, results


def _lite_results(query: str, region: str, safe: str, n: int) -> list[dict]:
    """Scrape DuckDuckGo Lite for organic results."""
    safe_map = {"off": "-2", "moderate": "-1", "strict": "1"}
    try:
        resp = requests.get(
            _DDG_LITE,
            params={
                "q":  query,
                "kl": region,
                "kp": safe_map.get(safe, "-1"),
            },
            headers = _HEADERS,
            timeout = _TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.text
    except Exception:
        return []

    results = []
    # Extract result links and snippets from DDG Lite HTML
    # Pattern: <a class="result-link" href="...">title</a> ... <td class="result-snippet">...</td>
    link_pat    = re.compile(
        r'<a[^>]+class=["\']result-link["\'][^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        re.DOTALL
    )
    snippet_pat = re.compile(
        r'<td[^>]+class=["\']result-snippet["\'][^>]*>(.*?)</td>',
        re.DOTALL
    )
    links    = link_pat.findall(text)
    snippets = snippet_pat.findall(text)

    for i, (url, title) in enumerate(links[:n]):
        snippet = snippets[i] if i < len(snippets) else ""
        results.append({
            "title":   _clean(title)[:100],
            "url":     url.strip(),
            "snippet": _clean(snippet)[:250],
        })

    return results


def _deduplicate(results: list[dict]) -> list[dict]:
    seen = set()
    out  = []
    for r in results:
        k = r.get("url", r.get("title", ""))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _build_summary(query: str, answer: str, results: list[dict]) -> str:
    if answer:
        return answer[:500]
    if not results:
        return f"No results found for '{query}'."
    titles = [r["title"] for r in results[:3] if r.get("title")]
    snippets = [r["snippet"] for r in results[:2] if r.get("snippet")]
    if snippets:
        return snippets[0][:400]
    return f"Found {len(results)} results including: {', '.join(titles[:3])}."


def _empty(query: str, reason: str) -> dict:
    return {
        "answer": "", "answer_type": "", "results": [],
        "count": 0, "query_used": query, "summary": reason,
    }
