#!/usr/bin/env python3
"""
Stock Monitor - Iran War & Tech Dip Tracker
Monitors stocks for >10% dips from 30-day high.
Sends daily summary at 8am via OpenClaw Telegram.
"""

import json
import yfinance as yf
from datetime import datetime, timedelta
import os

# === WATCHLIST ===
WATCHLIST = {
    "Iran-War Related": {
        # Energy
        "VLO": "Valero Energy",
        "MPC": "Marathon Petroleum",
        "SLB": "Schlumberger (SLB)",
        "XOM": "ExxonMobil",
        "CVX": "Chevron",
        # Shipping
        "ZIM": "ZIM Integrated Shipping",
        "AMKBY": "Maersk ADR",
        # Reconstruction/Engineering
        "FLR": "Fluor Corp",
        "KBR": "KBR Inc",
        "J": "Jacobs Solutions",
        # Cybersecurity
        "CRWD": "CrowdStrike",
        "PANW": "Palo Alto Networks",
        "FTNT": "Fortinet",
        # Defense/Infrastructure
        "RTX": "Raytheon Technologies",
        "LMT": "Lockheed Martin",
        # Agriculture/Food Security
        "ADM": "Archer-Daniels-Midland",
        "BG": "Bunge Global",
    },
    "Tech": {
        "NVDA": "NVIDIA",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet",
        "META": "Meta Platforms",
        "AMZN": "Amazon",
        "TSLA": "Tesla",
        "AMD": "AMD",
        "INTC": "Intel",
        "ASML": "ASML Holding",
        "TSM": "TSMC",
        "AVGO": "Broadcom",
        "QCOM": "Qualcomm",
        "ORCL": "Oracle",
        "CRM": "Salesforce",
        "SNOW": "Snowflake",
        "PLTR": "Palantir",
    }
}

DIP_THRESHOLD = 0.10  # 10%
OUTPUT_FILE = os.path.expanduser("~/.openclaw/workspace/data/stock_report.json")

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="35d")
        if hist.empty or len(hist) < 5:
            return None
        
        current_price = hist["Close"].iloc[-1]
        high_30d = hist["Close"].rolling(30).max().iloc[-1]
        prev_close = hist["Close"].iloc[-2]
        
        pct_from_high = (current_price - high_30d) / high_30d
        pct_change_1d = (current_price - prev_close) / prev_close
        
        volume = hist["Volume"].iloc[-1]
        avg_volume = hist["Volume"].rolling(20).mean().iloc[-1]
        
        info = stock.fast_info
        market_cap = getattr(info, 'market_cap', None)
        
        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "high_30d": round(high_30d, 2),
            "pct_from_high": round(pct_from_high * 100, 2),
            "pct_change_1d": round(pct_change_1d * 100, 2),
            "volume": int(volume),
            "avg_volume": int(avg_volume) if avg_volume else None,
            "volume_ratio": round(volume / avg_volume, 2) if avg_volume else None,
            "market_cap": market_cap,
            "dip_alert": bool(pct_from_high <= -DIP_THRESHOLD)
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

def run_scan():
    results = {
        "timestamp": datetime.now().isoformat(),
        "dip_threshold_pct": DIP_THRESHOLD * 100,
        "sectors": {},
        "alerts": []
    }
    
    for sector, stocks in WATCHLIST.items():
        results["sectors"][sector] = []
        for ticker, name in stocks.items():
            print(f"Fetching {ticker}...")
            data = get_stock_data(ticker)
            if data:
                data["name"] = name
                results["sectors"][sector].append(data)
                if data.get("dip_alert"):
                    results["alerts"].append({
                        "sector": sector,
                        "ticker": ticker,
                        "name": name,
                        "pct_from_high": data["pct_from_high"],
                        "current_price": data["current_price"],
                        "pct_change_1d": data["pct_change_1d"]
                    })
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Scan complete. {len(results['alerts'])} dip alerts found.")
    print(f"📁 Report saved to {OUTPUT_FILE}")
    
    if results["alerts"]:
        print("\n🔔 DIP ALERTS:")
        for alert in results["alerts"]:
            print(f"  {alert['ticker']} ({alert['name']}): {alert['pct_from_high']}% from 30d high | Today: {alert['pct_change_1d']}%")
    
    return results

if __name__ == "__main__":
    run_scan()
