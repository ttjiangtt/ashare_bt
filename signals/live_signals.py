"""
examples/live_signals.py
~~~~~~~~~~~~~~~~~~~~~~~~
Scans the local universe for entry signals as of a given date.
Supports Williams, Nine Turns, or both simultaneously.

Cross-reference columns:
  Williams email  → nt_stage  (current Nine Turns buy/sell count 1-9)
  Nine Turns email → williams (whether Williams signal is ON today)

Nine Turns mode shows stocks at any count stage, not just completed 9s.
Use --nt_min_stage / --nt_max_stage to filter which stages to show.

Usage:
    python examples/live_signals.py                         # williams only
    python examples/live_signals.py --signal nine_turns     # nine turns only
    python examples/live_signals.py --signal both           # both, two emails
    python examples/live_signals.py --signal nine_turns --nt_min_stage 7
    python examples/live_signals.py --backfill_from 2026-01-01
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LOG_COLS = ["run_date","ticker","name","side","signal_date","entry_price","consistency"]


# ── Signal helpers ────────────────────────────────────────────────────────────

def _build_williams(df):
    from signals.williams_signals import WilliamsSwings, WilliamsSignals
    return WilliamsSignals(WilliamsSwings(df).fit()).fit()

def _build_nine_turns(df, perfect=False):
    from signals.nine_turns import NineTurnsSignals
    return NineTurnsSignals(df, perfect=perfect).fit()


def get_nt_stage(df: pd.DataFrame, side: str) -> int:
    """
    Return the current Nine Turns count (1-9) on the last bar.
    side = 'BUY' uses buy_count, 'SHORT' uses sell_count.
    Returns 0 if count is 0 (no active setup).
    """
    try:
        nt = _build_nine_turns(df)
        col = "buy_count" if side == "BUY" else "sell_count"
        if nt.counts is not None and col in nt.counts.columns:
            return int(nt.counts[col].iloc[-1])
    except Exception:
        pass
    return 0


def has_williams_signal(df: pd.DataFrame, as_of: date, side: str) -> bool:
    """Return True if Williams fired a signal on as_of for the given side."""
    try:
        sg = _build_williams(df)
        if not sg.signals:
            return False
        last = sg.signals[-1]
        direction = 1 if side == "BUY" else -1
        return last.date.date() == as_of and last.direction == direction
    except Exception:
        return False


# ── Signal log helpers ────────────────────────────────────────────────────────

def log_path(signal_tag: str) -> Path:
    return Path(__file__).parent.parent / "williams" / signal_tag / "signal_log.csv"


def load_signal_log(signal_tag: str) -> pd.DataFrame:
    p = log_path(signal_tag)
    if p.exists():
        df = pd.read_csv(p, parse_dates=["run_date","signal_date"])
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        return df
    return pd.DataFrame(columns=LOG_COLS)


def append_signal_log(results: list, run_date: date, signal_tag: str) -> None:
    if not results:
        return
    p = log_path(signal_tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame([{
        "run_date":    str(run_date),
        "ticker":      r["ticker"],
        "name":        r["name"],
        "side":        r["side"],
        "signal_date": r["signal_date"],
        "entry_price": r["entry_price"],
        "consistency": r.get("consistency", float("nan")),
    } for r in results])
    rows.to_csv(p, mode="a", header=not p.exists(), index=False)


def build_stale_set(signal_log: pd.DataFrame, as_of: date, fresh_days: int) -> set:
    if signal_log.empty or fresh_days <= 0:
        return set()
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=fresh_days)
    recent = signal_log[
        (signal_log["run_date"] >= cutoff) &
        (signal_log["run_date"] <  pd.Timestamp(as_of))
    ]
    return set(zip(recent["ticker"], recent["side"]))


def load_consistency(signal_tag: str) -> dict:
    sweep_path = Path(__file__).parent.parent / "williams" / signal_tag / "sweep_by_year.csv"
    if not sweep_path.exists():
        log.warning("sweep_by_year.csv not found at %s — run markout_sweep.py --signal %s first",
                    sweep_path, signal_tag)
        return {}
    by_year = pd.read_csv(sweep_path)
    stats   = (by_year.groupby("ticker")["mean_5d"]
               .agg(years="count", mean="mean", std="std").reset_index())
    stats   = stats[stats["years"] >= 3].copy()
    stats["consistency"] = (stats["mean"] / stats["std"]).round(4)
    stats["ticker"] = stats["ticker"].astype(str).str.zfill(6)
    d = dict(zip(stats["ticker"], stats["consistency"]))
    log.info("Loaded %d consistency scores [%s]", len(d), signal_tag)
    return d


# ── Core scan functions ───────────────────────────────────────────────────────

def check_williams(ticker, api, as_of, side_filter, stale_set, consistency, min_c):
    """Check Williams signal for one ticker. Returns hit dict or None."""
    try:
        df = api.get(ticker, start="2015-01-01", end=str(as_of))
    except FileNotFoundError:
        return None
    if len(df) < 60 or df.index[-1].date() != as_of:
        return None
    try:
        sg = _build_williams(df)
    except Exception:
        return None
    if not sg.signals:
        return None
    last = sg.signals[-1]
    if last.date.date() != as_of:
        return None
    side = "BUY" if last.direction == 1 else "SHORT"
    if side_filter and side != side_filter.upper():
        return None
    if (ticker, side) in stale_set:
        return None
    c       = consistency.get(ticker, float("nan"))
    c_valid = (c == c)
    if min_c > float("-inf") and consistency:
        if not c_valid or c < min_c:
            return None
    # Cross-reference: Nine Turns stage
    nt_stage = get_nt_stage(df, side)
    return {
        "ticker":      ticker,
        "name":        api.name(ticker),
        "side":        side,
        "signal_date": str(last.date.date()),
        "entry_price": round(last.entry_close, 3),
        "consistency": c,
        "nt_stage":    nt_stage,   # cross-reference column
    }


def check_nine_turns(ticker, api, as_of, side_filter, stale_set, consistency, min_c,
                     perfect, nt_min_stage, nt_max_stage):
    """
    Check Nine Turns count for one ticker.
    Reports stocks at any stage between nt_min_stage and nt_max_stage.
    """
    try:
        df = api.get(ticker, start="2015-01-01", end=str(as_of))
    except FileNotFoundError:
        return None
    if len(df) < 60 or df.index[-1].date() != as_of:
        return None
    try:
        nt = _build_nine_turns(df, perfect=perfect)
    except Exception:
        return None

    results = []
    for col, side_label, direction in [("buy_count","BUY",1), ("sell_count","SHORT",-1)]:
        if side_filter and side_label != side_filter.upper():
            continue
        if nt.counts is None or col not in nt.counts.columns:
            continue
        stage = int(nt.counts[col].iloc[-1])
        if stage < nt_min_stage or stage > nt_max_stage:
            continue
        if (ticker, side_label) in stale_set:
            continue
        c       = consistency.get(ticker, float("nan"))
        c_valid = (c == c)
        if min_c > float("-inf") and consistency:
            if not c_valid or c < min_c:
                continue
        # Cross-reference: Williams signal
        w_on = has_williams_signal(df, as_of, side_label)
        # Only report as "signal fired" (for log) if stage == 9
        entry_price = round(float(df["close"].iloc[-1]), 3)
        results.append({
            "ticker":      ticker,
            "name":        api.name(ticker),
            "side":        side_label,
            "signal_date": str(as_of),
            "entry_price": entry_price,
            "consistency": c,
            "nt_stage":    stage,
            "williams":    "✓" if w_on else "—",
        })
    return results if results else None


# ── Scan one date ─────────────────────────────────────────────────────────────

def scan_date_williams(as_of, tickers, api, stale_set, consistency, min_c,
                       side_filter, silent=False):
    results = []
    t0      = time.time()
    if not silent:
        log.info("Scanning %d tickers [williams] as of %s…", len(tickers), as_of)
    for i, ticker in enumerate(tickers, 1):
        hit = check_williams(ticker, api, as_of, side_filter, stale_set, consistency, min_c)
        if hit:
            results.append(hit)
            if not silent:
                c = hit["consistency"]
                log.info("  ✓ %-8s  %-12s  %s  entry=%.2f  consistency=%s  nt_stage=%d",
                         hit["ticker"], hit["name"], hit["side"], hit["entry_price"],
                         f"{c:.3f}" if c==c else "n/a", hit["nt_stage"])
        if not silent and i % 100 == 0:
            log.info("[%d/%d]  signals: %d  (%.0f/min)",
                     i, len(tickers), len(results), i/(time.time()-t0)*60)
    return results


def scan_date_nine_turns(as_of, tickers, api, stale_set, consistency, min_c,
                         side_filter, perfect, nt_min_stage, nt_max_stage, silent=False):
    results = []
    t0      = time.time()
    if not silent:
        log.info("Scanning %d tickers [nine_turns stage %d-%d] as of %s…",
                 len(tickers), nt_min_stage, nt_max_stage, as_of)
    for i, ticker in enumerate(tickers, 1):
        hits = check_nine_turns(ticker, api, as_of, side_filter, stale_set, consistency,
                                min_c, perfect, nt_min_stage, nt_max_stage)
        if hits:
            for hit in hits:
                results.append(hit)
                if not silent:
                    c = hit["consistency"]
                    log.info("  ✓ %-8s  %-12s  %s  stage=%d  w=%s  consistency=%s",
                             hit["ticker"], hit["name"], hit["side"], hit["nt_stage"],
                             hit["williams"], f"{c:.3f}" if c==c else "n/a")
        if not silent and i % 100 == 0:
            log.info("[%d/%d]  signals: %d  (%.0f/min)",
                     i, len(tickers), len(results), i/(time.time()-t0)*60)
    return results


# ── Email ─────────────────────────────────────────────────────────────────────

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


def _streak_ytd(signal_log, run_date, r):
    if signal_log.empty:
        return 1, 1
    mask  = (signal_log["ticker"] == r["ticker"]) & (signal_log["side"] == r["side"])
    prior = signal_log[mask].copy()
    if prior.empty:
        return 1, 1
    dates  = sorted(prior["run_date"].dt.date.unique(), reverse=True)
    streak = 1
    prev   = pd.Timestamp(run_date).date()
    for d in dates:
        gap = (prev - d).days
        if 0 < gap <= 3:
            streak += 1; prev = d
        elif gap == 0:
            continue
        else:
            break
    ytd = int(signal_log[mask & (signal_log["run_date"].dt.year == pd.Timestamp(run_date).year)]
              ["run_date"].nunique())
    return streak, ytd


def build_williams_email(results, run_date, args, signal_log):
    n_buy   = sum(1 for r in results if r["side"] == "BUY")
    n_short = sum(1 for r in results if r["side"] == "SHORT")
    subject = (
        f"[Williams] Signals {run_date} — {n_buy} BUY, {n_short} SHORT "
        f"(consistency>={args.min_consistency}, fresh<={args.fresh_days}d)"
    )
    if not results:
        return subject, f"No Williams signals on {run_date}."
    lines = [
        f"Williams Entry Signals — {run_date}",
        f"Min consistency: {args.min_consistency}  |  Fresh: {args.fresh_days}d",
        "=" * 82,
    ]
    for side_label in ("BUY","SHORT"):
        subset = sorted([r for r in results if r["side"] == side_label],
                        key=lambda r: r["consistency"] if r["consistency"]==r["consistency"] else -999,
                        reverse=True)
        if not subset: continue
        lines.append(f"\n{side_label} ({len(subset)}):")
        lines.append(f"{'Ticker':<8}  {'Name':<14}  {'Date':<12}  {'Entry':>8}  "
                     f"{'Consist':>7}  {'Streak':>6}  {'YTD':>5}  {'NT Stage':>8}")
        lines.append("-" * 82)
        for r in subset:
            c     = r["consistency"]
            c_str = f"{c:.3f}" if c==c else "n/a"
            st, ytd = _streak_ytd(signal_log, run_date, r)
            nt = r.get("nt_stage", 0)
            nt_str = f"{nt}/9" if nt > 0 else "—"
            lines.append(f"{r['ticker']:<8}  {r['name']:<14}  {r['signal_date']:<12}  "
                         f"{r['entry_price']:>8.2f}  {c_str:>7}  {st:>6}  {ytd:>5}  {nt_str:>8}")
    return subject, "\n".join(lines)


def build_nine_turns_email(results, run_date, args, signal_log, nt_min_stage, nt_max_stage):
    n_buy   = sum(1 for r in results if r["side"] == "BUY")
    n_short = sum(1 for r in results if r["side"] == "SHORT")
    stage_info = f"stage {nt_min_stage}-{nt_max_stage}"
    subject = (
        f"[NineTurns {stage_info}] Signals {run_date} — {n_buy} BUY, {n_short} SHORT "
        f"(consistency>={args.min_consistency})"
    )
    if not results:
        return subject, f"No Nine Turns signals ({stage_info}) on {run_date}."
    lines = [
        f"Nine Turns Signals ({stage_info}) — {run_date}",
        f"Min consistency: {args.min_consistency}  |  Perfect filter: {getattr(args,'perfect',False)}",
        "=" * 82,
    ]
    for side_label in ("BUY","SHORT"):
        subset = sorted([r for r in results if r["side"] == side_label],
                        key=lambda r: (r["nt_stage"], r["consistency"] if r["consistency"]==r["consistency"] else -999),
                        reverse=True)
        if not subset: continue
        lines.append(f"\n{side_label} ({len(subset)}) — sorted by stage then consistency:")
        lines.append(f"{'Ticker':<8}  {'Name':<14}  {'Stage':>5}  {'Entry':>8}  "
                     f"{'Consist':>7}  {'Williams':>8}")
        lines.append("-" * 65)
        for r in subset:
            c     = r["consistency"]
            c_str = f"{c:.3f}" if c==c else "n/a"
            lines.append(f"{r['ticker']:<8}  {r['name']:<14}  {r['nt_stage']:>5}/9  "
                         f"{r['entry_price']:>8.2f}  {c_str:>7}  {r.get('williams','—'):>8}")
    return subject, "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    default="C:/Users/ttjia/OneDrive/Work/ashare/market_data")
    parser.add_argument("--signal",  default="williams",
                        choices=["williams","nine_turns","both"],
                        help="Signal type: williams, nine_turns, or both (default: williams)")
    parser.add_argument("--perfect", action="store_true", help="nine_turns: perfection filter")
    parser.add_argument("--side",    default="BUY")
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--min_consistency", default=0.5, type=float)
    parser.add_argument("--fresh_days",      default=0,   type=int,
                        help="Williams freshness filter (days)")
    parser.add_argument("--nt_min_stage",    default=1,   type=int,
                        help="Nine Turns: minimum stage to report (default 1)")
    parser.add_argument("--nt_max_stage",    default=9,   type=int,
                        help="Nine Turns: maximum stage to report (default 9)")
    parser.add_argument("--as_of",          default=None)
    parser.add_argument("--backfill_from",   default=None)
    parser.add_argument("--backfill_to",     default=None)
    parser.add_argument("--email_to",   default=None)
    parser.add_argument("--email_from", default=None)
    parser.add_argument("--email_pass", default=None)
    args = parser.parse_args()

    import os
    email_to   = args.email_to   or os.environ.get("ASHARE_EMAIL_TO")
    email_from = args.email_from or os.environ.get("ASHARE_EMAIL_FROM")
    email_pass = args.email_pass or os.environ.get("ASHARE_EMAIL_PASS")

    if email_to and email_from and email_pass:
        log.info("Email configured → %s", email_to)
    else:
        log.info("No email configured")

    from data.local_api import LocalDataAPI
    api = LocalDataAPI(args.root)

    tickers = (
        [t.strip().zfill(6) for t in args.tickers.split(",")]
        if args.tickers else api.list_tickers()
    )

    run_williams   = args.signal in ("williams", "both")
    run_nine_turns = args.signal in ("nine_turns", "both")

    nt_tag = "nine_turns_perfect" if args.perfect else "nine_turns"

    # Consistency scores per signal type
    w_consistency  = load_consistency("williams")    if run_williams   else {}
    nt_consistency = load_consistency(nt_tag)        if run_nine_turns else {}

    # Build run dates
    if args.backfill_from:
        start_d   = date.fromisoformat(args.backfill_from)
        end_d     = date.fromisoformat(args.backfill_to) if args.backfill_to else date.today()
        run_dates = []
        d = start_d
        while d <= end_d:
            if d.weekday() < 5:
                run_dates.append(d)
            d += timedelta(days=1)
        log.info("Backfill: %d trading days %s → %s", len(run_dates), start_d, end_d)
        is_backfill = True
    else:
        run_dates   = [date.fromisoformat(args.as_of) if args.as_of else date.today()]
        is_backfill = False

    w_log  = load_signal_log("williams") if run_williams   else pd.DataFrame(columns=LOG_COLS)
    nt_log = load_signal_log(nt_tag)     if run_nine_turns else pd.DataFrame(columns=LOG_COLS)

    w_all_results  = []
    nt_all_results = []

    for run_date in run_dates:
        if is_backfill:
            log.info("── %s ──", run_date)

        verbose = not is_backfill or len(run_dates) <= 5

        if run_williams:
            w_stale = build_stale_set(w_log, run_date, args.fresh_days)
            w_res   = scan_date_williams(
                run_date, tickers, api, w_stale, w_consistency,
                args.min_consistency, args.side, silent=not verbose,
            )
            append_signal_log(w_res, run_date, "williams")
            w_log = load_signal_log("williams")
            if not is_backfill:
                w_all_results = w_res
            elif is_backfill:
                log.info("  Williams: %d signals", len(w_res))

        if run_nine_turns:
            # Nine Turns freshness only applies at stage 9 (completion)
            nt_stale = build_stale_set(nt_log, run_date, args.fresh_days) if args.nt_max_stage == 9 else set()
            nt_res   = scan_date_nine_turns(
                run_date, tickers, api, nt_stale, nt_consistency,
                args.min_consistency, args.side, args.perfect,
                args.nt_min_stage, args.nt_max_stage, silent=not verbose,
            )
            # Only log completed 9s (not intermediate stages)
            completed = [r for r in nt_res if r["nt_stage"] == 9]
            append_signal_log(completed, run_date, nt_tag)
            nt_log = load_signal_log(nt_tag)
            if not is_backfill:
                nt_all_results = nt_res
            elif is_backfill:
                log.info("  Nine Turns: %d signals", len(nt_res))

    if is_backfill:
        log.info("Backfill complete.")
        return

    run_date = run_dates[0]

    # ── Print ─────────────────────────────────────────────────────────────────
    if run_williams:
        print("\n" + "=" * 65)
        print(f"WILLIAMS SIGNALS  —  {run_date}  (EOD)")
        print("=" * 65)
        if not w_all_results:
            print("No Williams signals.")
        else:
            df_w = pd.DataFrame(w_all_results)
            cols = ["ticker","name","signal_date","entry_price","consistency","nt_stage"]
            for sl in ("BUY","SHORT"):
                sub = df_w[df_w["side"]==sl]
                if sub.empty: continue
                print(f"\n── {sl} ({len(sub)}) ──")
                print(sub.sort_values("consistency",ascending=False)[cols].to_string(index=False))

    if run_nine_turns:
        print("\n" + "=" * 65)
        print(f"NINE TURNS SIGNALS (stage {args.nt_min_stage}-{args.nt_max_stage})  —  {run_date}  (EOD)")
        print("=" * 65)
        if not nt_all_results:
            print("No Nine Turns signals.")
        else:
            df_nt = pd.DataFrame(nt_all_results)
            cols  = ["ticker","name","nt_stage","entry_price","consistency","williams"]
            for sl in ("BUY","SHORT"):
                sub = df_nt[df_nt["side"]==sl]
                if sub.empty: continue
                print(f"\n── {sl} ({len(sub)}) ──")
                print(sub.sort_values(["nt_stage","consistency"],ascending=[False,False])[cols].to_string(index=False))

    # ── Email ─────────────────────────────────────────────────────────────────
    if email_to and email_from and email_pass:
        if run_williams and w_all_results is not None:
            try:
                subj, body = build_williams_email(w_all_results, run_date, args, w_log)
                send_email(email_to, email_from, email_pass, subj, body)
                log.info("Williams email sent to %s", email_to)
            except Exception as e:
                log.error("Williams email failed: %s", e)

        if run_nine_turns and nt_all_results is not None:
            try:
                subj, body = build_nine_turns_email(nt_all_results, run_date, args, nt_log,
                                                     args.nt_min_stage, args.nt_max_stage)
                send_email(email_to, email_from, email_pass, subj, body)
                log.info("Nine Turns email sent to %s", email_to)
            except Exception as e:
                log.error("Nine Turns email failed: %s", e)
    elif email_to:
        log.warning("email_to set but from/pass missing.")


if __name__ == "__main__":
    main()
