"""
HUGINN_MANIFEST
tool_id:            tool.system.status.v1
title:              System Status
capability_summary: >
  Return current system resource usage — CPU, memory, disk, uptime, and
  process count. Read-only, safe for aug_call. Use when the user asks how
  the system is doing, whether Artux is running well, or for diagnostics.
polarity:           read
permission_scope:   []
inputs:
  include_processes: {type: boolean, default: false, description: "Include top-5 CPU processes"}
outputs:
  cpu_pct:          {type: number,  description: "CPU usage 0-100"}
  memory_total_gb:  {type: number}
  memory_used_gb:   {type: number}
  memory_pct:       {type: number}
  disk_total_gb:    {type: number}
  disk_used_gb:     {type: number}
  disk_pct:         {type: number}
  uptime_s:         {type: integer, description: "System uptime in seconds"}
  uptime_human:     {type: string,  description: "e.g. '2 days, 4 hours'"}
  process_count:    {type: integer}
  top_processes:    {type: array,   description: "Top-5 by CPU [{name, pid, cpu_pct, mem_pct}] — only if include_processes=true"}
  summary:          {type: string,  description: "One-sentence plain English summary for speech"}
dependencies:
  - psutil>=5.9
perception_capable: false
handler:            handle
END_MANIFEST

System resource status using psutil.

Returns a concise snapshot of CPU, memory, disk, and uptime.
The 'summary' field is pre-formatted for TTS output:
  "CPU at 12%, memory 4.2 GB of 16 GB, disk 58% full, up 2 days."

Disk is measured on the root filesystem by default. Override with
HUGINN_STATUS_DISK_PATH (e.g. /home or a mounted drive).
"""

from __future__ import annotations

import os
import datetime


def handle(include_processes: bool = False) -> dict:
    """Return current system status as a structured dict."""
    import psutil

    disk_path = os.environ.get("HUGINN_STATUS_DISK_PATH", "/")

    cpu_pct  = psutil.cpu_percent(interval=0.3)
    mem      = psutil.virtual_memory()
    disk     = psutil.disk_usage(disk_path)
    uptime_s = int(datetime.datetime.now().timestamp() - psutil.boot_time())
    procs    = len(psutil.pids())

    top = []
    if include_processes:
        proc_list = []
        for p in psutil.process_iter(["name", "pid", "cpu_percent", "memory_percent"]):
            try:
                proc_list.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        # Second pass for accurate cpu_percent (first call always 0.0)
        proc_list = sorted(proc_list, key=lambda x: x.get("cpu_percent", 0),
                           reverse=True)[:5]
        top = [
            {
                "name":     p.get("name", ""),
                "pid":      p.get("pid",  0),
                "cpu_pct":  round(p.get("cpu_percent",  0.0), 1),
                "mem_pct":  round(p.get("memory_percent", 0.0), 2),
            }
            for p in proc_list
        ]

    result = {
        "cpu_pct":         round(cpu_pct, 1),
        "memory_total_gb": round(mem.total  / 1e9, 2),
        "memory_used_gb":  round(mem.used   / 1e9, 2),
        "memory_pct":      round(mem.percent, 1),
        "disk_total_gb":   round(disk.total / 1e9, 1),
        "disk_used_gb":    round(disk.used  / 1e9, 1),
        "disk_pct":        round(disk.percent, 1),
        "uptime_s":        uptime_s,
        "uptime_human":    _human_uptime(uptime_s),
        "process_count":   procs,
        "top_processes":   top,
        "summary":         "",
    }
    result["summary"] = _build_summary(result)
    return result


def _human_uptime(seconds: int) -> str:
    days    = seconds // 86400
    hours   = (seconds % 86400) // 3600
    minutes = (seconds % 3600)  // 60

    parts = []
    if days:    parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:   parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes and not days: parts.append(f"{minutes} min")
    return ", ".join(parts) if parts else "just started"


def _build_summary(r: dict) -> str:
    mem_used  = r["memory_used_gb"]
    mem_total = r["memory_total_gb"]
    return (
        f"CPU at {r['cpu_pct']}%, "
        f"memory {mem_used:.1f} GB of {mem_total:.0f} GB, "
        f"disk {r['disk_pct']}% full, "
        f"up {r['uptime_human']}."
    )
