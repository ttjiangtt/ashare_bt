"""
examples/portfolio_signals.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Given a list of tickers, shows today's Williams and Nine Turns status
for each and emails the table.

Usage:
    python examples/portfolio_signals.py --tickers 600519,000001,601318
    python examples/portfolio_signals.py --file my_tickers.txt
    python examples/portfolio_signals.py --tickers 600519,000001 --as_of 2026-04-23

Ticker file format (one per line, or comma-separated):
    600519
    000001
    601318,600036,000858
"""

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_williams_status(df, as_of: date) -> str:
    """
    Returns 'BUY', 'SHORT', or '—' depending on whether Williams
    fired a signal on as_of.
    """
    try:
        from signals.williams_signals import WilliamsSwings, WilliamsSignals
        sg = WilliamsSignals(WilliamsSwings(df).fit()).fit()
        if not sg.signals:
            return "—"
        last = sg.signals[-1]
        if last.date.date() == as_of:
            return "BUY" if last.direction == 1 else "SHORT"
        return "—"
    except Exception:
        return "ERR"


def get_nt_status(df, perfect: bool = False) -> str:
    """
    Returns the current Nine Turns buy/sell count as a string like
    'BUY 7/9', 'SHORT 3/9', or '—' if no active setup.
    If both buy and sell counts are active, returns the higher one.
    """
    try:
        from signals.nine_turns import NineTurnsSignals
        nt = NineTurnsSignals(df, perfect=perfect).fit()
        if nt.counts is None:
            return "—"
        buy_stage  = int(nt.counts["buy_count"].iloc[-1])
        sell_stage = int(nt.counts["sell_count"].iloc[-1])
        parts = []
        if buy_stage > 0:
            parts.append(f"BUY {buy_stage}/9")
        if sell_stage > 0:
            parts.append(f"SHORT {sell_stage}/9")
        return "  |  ".join(parts) if parts else "—"
    except Exception:
        return "ERR"


def build_table(tickers: list, api, as_of: date, perfect: bool) -> list:
    rows = []
    for ticker in tickers:
        try:
            df = api.get(ticker, start="2015-01-01", end=str(as_of))
        except FileNotFoundError:
            rows.append({
                "ticker":    ticker,
                "name":      api.name(ticker),
                "williams":  "NO DATA",
                "nt_status": "NO DATA",
            })
            continue

        if len(df) < 60 or df.index[-1].date() != as_of:
            rows.append({
                "ticker":    ticker,
                "name":      api.name(ticker),
                "williams":  "NO DATA",
                "nt_status": "NO DATA",
            })
            continue

        w  = get_williams_status(df, as_of)
        nt = get_nt_status(df, perfect=perfect)

        rows.append({
            "ticker":    ticker,
            "name":      api.name(ticker),
            "williams":  w,
            "nt_status": nt,
        })
        log.info("  %-8s  %-14s  Williams=%-6s  NT=%s",
                 ticker, api.name(ticker), w, nt)

    return rows


def send_email(to, sender, password, subject, body):
    import smtplib
    from email.mime.text import MIMEText
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, to, msg.as_string())


def build_email(rows: list, as_of: date, perfect: bool) -> tuple:
    active = [r for r in rows if r["williams"] != "—" or r["nt_status"] != "—"]
    subject = (
        f"Portfolio Signals — {as_of}  "
        f"({sum(1 for r in rows if r['williams'] != '—' and r['williams'] != 'NO DATA')} Williams active, "
        f"{sum(1 for r in rows if r['nt_status'] not in ('—','NO DATA','ERR'))} NT active)"
    )

    lines = [
        f"Portfolio Signal Status — {as_of}",
        f"Nine Turns perfect filter: {'ON' if perfect else 'OFF'}",
        "=" * 70,
        f"{'Ticker':<8}  {'Name':<16}  {'Williams':<10}  {'Nine Turns':<20}",
        "-" * 70,
    ]
    for r in rows:
        lines.append(
            f"{r['ticker']:<8}  {r['name']:<16}  {r['williams']:<10}  {r['nt_status']:<20}"
        )
    lines.append("")
    lines.append(f"Total tickers: {len(rows)}")
    lines.append(f"Williams active: {sum(1 for r in rows if r['williams'] in ('BUY','SHORT'))}")
    lines.append(f"NT active (any stage): {sum(1 for r in rows if r['nt_status'] not in ('—','NO DATA','ERR'))}")

    return subject, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Show Williams + Nine Turns status for a list of tickers."
    )
    parser.add_argument("--root",    default="C:/Users/ttjia/OneDrive/Work/ashare/market_data")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated tickers, e.g. 600519,000001,601318")
    parser.add_argument("--file",    default="C:/Users/ttjia/OneDrive/Work/ashare/mypfo.txt",
                        help="Text file with tickers (one per line or comma-separated)")
    parser.add_argument("--as_of",   default=None,
                        help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--perfect", action="store_true",
                        help="Nine Turns: apply perfection filter")
    parser.add_argument("--email_to",   default="ttjiangtt@gmail.com")
    parser.add_argument("--email_from", default="ttjiangtt@gmail.com")
    parser.add_argument("--email_pass", default=None)
    args = parser.parse_args()

    email_to   = args.email_to   or os.environ.get("ASHARE_EMAIL_TO")
    email_from = args.email_from or os.environ.get("ASHARE_EMAIL_FROM")
    email_pass = args.email_pass or os.environ.get("ASHARE_EMAIL_PASS")

    if email_to and email_from and email_pass:
        log.info("Email configured → %s", email_to)
    else:
        log.info("No email configured — will print to console only")

    # Build ticker list
    tickers = []
    if args.tickers:
        tickers += [t.strip().zfill(6) for t in args.tickers.split(",") if t.strip()]
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
        for token in text.replace("\n", ",").split(","):
            t = token.strip()
            if t:
                tickers.append(t.zfill(6))
    tickers = list(dict.fromkeys(tickers))   # deduplicate, preserve order

    if not tickers:
        print("No tickers provided. Use --tickers 600519,000001 or --file tickers.txt")
        return

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    from data.local_api import LocalDataAPI
    api = LocalDataAPI(args.root)

    log.info("Checking %d tickers as of %s…", len(tickers), as_of)
    rows = build_table(tickers, api, as_of, args.perfect)

    # Print table
    df = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print(f"Portfolio Signal Status — {as_of}")
    print(f"{'='*70}")
    print(df[["ticker","name","williams","nt_status"]].to_string(index=False))
    print(f"\nWilliams active : {sum(1 for r in rows if r['williams'] in ('BUY','SHORT'))}/{len(rows)}")
    print(f"NT active       : {sum(1 for r in rows if r['nt_status'] not in ('—','NO DATA','ERR'))}/{len(rows)}")

    # Send email
    if email_to and email_from and email_pass:
        try:
            subject, body = build_email(rows, as_of, args.perfect)
            send_email(email_to, email_from, email_pass, subject, body)
            log.info("Email sent to %s", email_to)
        except Exception as e:
            log.error("Email failed: %s", e)


if __name__ == "__main__":
    main()
