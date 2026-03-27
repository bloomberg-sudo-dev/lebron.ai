#!/usr/bin/env python3
"""
Daily stock brief — runs scan and sends Telegram message via OpenClaw.
"""
import json
import subprocess
import os
import sys
from datetime import datetime

REPORT_FILE = os.path.expanduser("~/.openclaw/workspace/data/stock_report.json")
MONITOR_SCRIPT = os.path.expanduser("~/.openclaw/workspace/scripts/stock_monitor.py")

def run_scan():
    result = subprocess.run(["python3", MONITOR_SCRIPT], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Scan error: {result.stderr}")
        return False
    return True

def format_message(report):
    now = datetime.now().strftime("%a %b %d, %Y")
    alerts = report.get("alerts", [])
    sectors = report.get("sectors", {})
    
    lines = [f"📊 *Daily Stock Brief — {now}*\n"]

    if alerts:
        lines.append(f"🔔 *{len(alerts)} DIP ALERT(S) — >10% from 30-day high:*")
        for a in alerts:
            direction = "🔴" if a["pct_change_1d"] < 0 else "🟢"
            lines.append(
                f"{direction} *{a['ticker']}* ({a['name']})\n"
                f"   ↓ {abs(a['pct_from_high'])}% from 30d high | "
                f"Today: {'+' if a['pct_change_1d'] > 0 else ''}{a['pct_change_1d']}% | "
                f"${a['current_price']}"
            )
    else:
        lines.append("✅ No dips >10% today across watchlist.")

    lines.append("\n📈 *Full Watchlist Snapshot:*")
    for sector, stocks in sectors.items():
        lines.append(f"\n*{sector}:*")
        for s in stocks:
            if "error" in s:
                continue
            arrow = "🔴" if s["pct_change_1d"] < 0 else "🟢"
            dip_flag = " ⚠️" if s.get("dip_alert") else ""
            lines.append(
                f"{arrow} {s['ticker']}: ${s['current_price']} "
                f"({'+' if s['pct_change_1d'] > 0 else ''}{s['pct_change_1d']}% today, "
                f"{s['pct_from_high']}% from 30d high){dip_flag}"
            )

    lines.append(f"\n_Ask me for a full financial report on any ticker anytime._")
    return "\n".join(lines)

def send_via_openclaw(message):
    """Send message by writing to a temp file for openclaw to pick up via cron agent call."""
    # Use openclaw CLI to send a message to the main telegram session
    cmd = ["openclaw", "message", "send", "--channel", "telegram", message]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback: just print (cron will log it)
        print(f"OpenClaw send failed: {result.stderr}")
        print("MESSAGE:\n" + message)
    else:
        print("Message sent successfully.")

if __name__ == "__main__":
    print(f"[{datetime.now()}] Running stock scan...")
    if run_scan():
        with open(REPORT_FILE) as f:
            report = json.load(f)
        message = format_message(report)
        print(message)
        # Output to stdout for openclaw cron to capture and forward
        sys.stdout.flush()
    else:
        print("Scan failed.")
