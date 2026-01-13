# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - Robust Version 4.2.2 (LEAGUE FIX + DEBUG + UI STATUS)
Flask API with:
- Poisson score model + Monte Carlo blend
- PSS (same league + home-only / away-only)
- H2H (same league if enough)
- Corner O/U with Poisson (8.5 / 9.5 / 10.5) + net pick
- Net picks for O/U 2.5 and BTTS with status: good/warn/bad (Android UI)
- Debug lines returned when payload.debug=true
"""

import re
import math
import time
import traceback
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import requests
from flask import Flask, request, jsonify

# ======================
# LOGGING
# ======================
logger = logging.getLogger("nowgoal")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ======================
# CONFIG
# ======================
MC_RUNS_DEFAULT = 10_000
RECENT_N = 10
H2H_N = 10
MAX_GOALS_FOR_MATRIX = 5

W_ST_BASE = 0.45
W_PSS_BASE = 0.30
W_H2H_BASE = 0.25

BLEND_ALPHA = 0.50

# Value thresholds (kept, but odds can be missing)
VALUE_MIN = 0.05
PROB_MIN = 0.55
KELLY_MIN = 0.02

# UI probability thresholds
P_GOOD = 0.62   # good
P_WARN = 0.56   # warn
# below warn => bad

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# ======================
# REGEX
# ======================
DATE_ANY_RE = re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2})\b")
SCORE_RE = re.compile(
    r"\b(\d{1,2})\s*-\s*(\d{1,2})(?!-\d{2,4})(?:\s*\((\d{1,2})\s*-\s*(\d{1,2})\))?\b"
)
CORNER_FT_RE = re.compile(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b")
CORNER_HT_RE = re.compile(r"\((\d{1,2})\s*-\s*(\d{1,2})\)")

FLOAT_RE = re.compile(r"(?<!\d)(\d+\.\d+)(?!\d)")
LEAGUE_CODE_RE = re.compile(r"\[([A-Z]{2,4}\s*[A-Z0-9]{1,4})-\d+\]")  # [GER D1-5] -> GER D1

# ======================
# UTILS
# ======================
def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def normalize_date(d: str) -> Optional[str]:
    if not d:
        return None
    d = d.strip()
    m = DATE_ANY_RE.search(d)
    if not m:
        return None
    val = m.group(1).replace("/", "-")
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
        yyyy, mm, dd = val.split("-")
        return f"{int(dd):02d}-{int(mm):02d}-{yyyy}"
    if re.match(r"^\d{1,2}-\d{1,2}-\d{4}$", val):
        dd, mm, yyyy = val.split("-")
        return f"{int(dd):02d}-{int(mm):02d}-{yyyy}"
    return None

def parse_date_key(date_str: str) -> Tuple[int, int, int]:
    if not date_str or not re.match(r"^\d{2}-\d{2}-\d{4}$", date_str):
        return (0, 0, 0)
    dd, mm, yyyy = date_str.split("-")
    return (int(yyyy), int(mm), int(dd))

def pick_status(p: float) -> Dict[str, str]:
    if p >= P_GOOD:
        return {"status": "good", "icon": "tick"}      # UI: green
    if p >= P_WARN:
        return {"status": "warn", "icon": "triangle"}  # UI: yellow
    return {"status": "bad", "icon": "stop"}           # UI: red

# ======================
# DATA CLASSES
# ======================
@dataclass
class MatchRow:
    league: str
    date: str
    home: str
    away: str
    ft_home: int
    ft_away: int
    ht_home: Optional[int] = None
    ht_away: Optional[int] = None
    corner_home: Optional[int] = None
    corner_away: Optional[int] = None
    corner_ht_home: Optional[int] = None
    corner_ht_away: Optional[int] = None

@dataclass
class SplitGFGA:
    matches: int
    gf: int
    ga: int
    @property
    def gf_pg(self) -> float:
        return self.gf / self.matches if self.matches else 0.0
    @property
    def ga_pg(self) -> float:
        return self.ga / self.matches if self.matches else 0.0

@dataclass
class StandRow:
    ft: str
    matches: Optional[int]
    win: Optional[int]
    draw: Optional[int]
    loss: Optional[int]
    scored: Optional[int]
    conceded: Optional[int]
    pts: Optional[int]
    rank: Optional[int]
    rate: Optional[str]

@dataclass
class TeamPrevStats:
    name: str
    gf_total: float = 0.0
    ga_total: float = 0.0
    n_total: int = 0
    gf_home: float = 0.0
    ga_home: float = 0.0
    n_home: int = 0
    gf_away: float = 0.0
    ga_away: float = 0.0
    n_away: int = 0
    clean_sheets: int = 0
    scored_matches: int = 0
    corners_for: float = 0.0
    corners_against: float = 0.0

# ======================
# HTML PARSE
# ======================
def strip_tags_keep_text(s: str) -> str:
    s = re.sub(r"<script\b.*?</script>", " ", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"<style\b.*?</style>", " ", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_tables_html(page_source: str) -> List[str]:
    return [m.group(0) for m in re.finditer(
        r"<table\b[^>]*>.*?</table>", page_source or "", flags=re.IGNORECASE | re.DOTALL
    )]

def extract_table_rows_from_html(table_html: str) -> List[List[str]]:
    """
    Hücreleri silme. Boş hücreler kolon hizası için gerekli.
    """
    rows: List[List[str]] = []
    trs = re.findall(r"<tr\b[^>]*>.*?</tr>", table_html or "", flags=re.IGNORECASE | re.DOTALL)

    for tr in trs:
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, flags=re.IGNORECASE | re.DOTALL)
        if not cells:
            continue

        cleaned = [strip_tags_keep_text(c) for c in cells]
        normalized = []
        for c in cleaned:
            c = (c or "").strip()
            if c in {"—", "-", "---", "----"}:
                c = ""
            normalized.append(c)

        if any(x for x in normalized):
            rows.append(normalized)

    return rows

def section_tables_by_marker(page_source: str, marker: str, max_tables: int = 3) -> List[str]:
    low = (page_source or "").lower()
    pos = low.find(marker.lower())
    if pos == -1:
        return []
    sub = page_source[pos:]
    tabs = extract_tables_html(sub)
    return tabs[:max_tables]

# ======================
# FETCH
# ======================
def safe_get(url: str, timeout: int = 25, retries: int = 2, referer: Optional[str] = None,
             debug_lines: Optional[List[str]] = None) -> str:
    headers = dict(HEADERS)
    if referer:
        headers["Referer"] = referer

    last_err = None
    for attempt in range(1, retries + 2):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            msg = f"FETCH OK [{attempt}/{retries+1}] status={r.status_code} bytes={len(r.text)} url={url}"
            logger.info(msg)
            if debug_lines is not None:
                debug_lines.append(msg)
            return r.text
        except Exception as e:
            last_err = e
            msg = f"FETCH FAIL [{attempt}/{retries+1}] url={url} err={e}"
            logger.warning(msg)
            if debug_lines is not None:
                debug_lines.append(msg)
            time.sleep(0.7)

    raise RuntimeError(f"Fetch failed: {url} ({last_err})")

def extract_match_id(url: str) -> str:
    m = re.search(r"(?:h2h-|/match/h2h-)(\d+)", url)
    if m:
        return m.group(1)
    nums = re.findall(r"\d{6,}", url)
    if not nums:
        raise ValueError("Match ID çıkaramadım")
    return nums[-1]

def extract_base_domain(url: str) -> str:
    m = re.match(r"^(https?://[^/]+)", url.strip())
    return m.group(1) if m else "https://live3.nowgoal26.com"

def build_h2h_url(url: str) -> str:
    match_id = extract_match_id(url)
    base = extract_base_domain(url)
    return f"{base}/match/h2h-{match_id}"

def build_oddscomp_url(url: str) -> str:
    match_id = extract_match_id(url)
    base = extract_base_domain(url)
    return f"{base}/oddscomp/{match_id}"

def parse_teams_from_title(html: str) -> Tuple[str, str]:
    m = re.search(r"<title>\s*(.*?)\s*</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = strip_tags_keep_text(m.group(1)) if m else ""
    mm = re.search(r"(.+?)\s+VS\s+(.+?)(?:\s+-|\s+\||$)", title, flags=re.IGNORECASE)
    if not mm:
        mm = re.search(r"(.+?)\s+vs\s+(.+?)(?:\s+-|\s+\||$)", title, flags=re.IGNORECASE)
    if not mm:
        return "", ""
    return mm.group(1).strip(), mm.group(2).strip()

def sort_matches_desc(matches: List[MatchRow]) -> List[MatchRow]:
    has_real_date = any(parse_date_key(m.date) != (0, 0, 0) for m in matches)
    if not has_real_date:
        return matches
    return sorted(matches, key=lambda x: parse_date_key(x.date), reverse=True)

def dedupe_matches(matches: List[MatchRow]) -> List[MatchRow]:
    seen = set()
    out = []
    for m in matches:
        key = (m.league, m.date, m.home, m.away, m.ft_home, m.ft_away, m.corner_home, m.corner_away)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out

def is_h2h_pair(m: MatchRow, home_team: str, away_team: str) -> bool:
    hk, ak = norm_key(home_team), norm_key(away_team)
    mh, ma = norm_key(m.home), norm_key(m.away)
    return (mh == hk and ma == ak) or (mh == ak and ma == hk)

# ======================
# LEAGUE EXTRACTION
# ======================
def extract_league_code_from_html(html: str) -> str:
    txt = strip_tags_keep_text(html)
    m = LEAGUE_CODE_RE.search(txt)
    if m:
        return m.group(1).strip()
    return ""

def guess_league_code_from_matches(matches: List[MatchRow]) -> str:
    # NowGoal often has league column like "GER D1"
    codes = [m.league.strip() for m in matches if m.league and len(m.league.strip()) <= 12]
    if not codes:
        return ""
    return Counter(codes).most_common(1)[0][0]

# ======================
# CORNER PARSE
# ======================
def parse_corner_cell(cell: str) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    '4-9(0-4)' -> FT=(4,9), HT=(0,4)
    """
    if not cell:
        return None, None
    txt = (cell or "").strip()
    if txt in {"", "-", "—", "---"}:
        return None, None

    ft_m = CORNER_FT_RE.search(txt)
    ht_m = CORNER_HT_RE.search(txt)

    ft = (int(ft_m.group(1)), int(ft_m.group(2))) if ft_m else None
    ht = (int(ht_m.group(1)), int(ht_m.group(2))) if ht_m else None
    return ft, ht

# ======================
# MATCH PARSE
# ======================
def parse_match_from_cells(cells: List[str]) -> Optional[MatchRow]:
    if not cells:
        return None

    def get(i: int) -> str:
        return (cells[i] or "").strip() if i < len(cells) else ""

    league = get(0) or "—"
    date_cell = get(1)
    home = get(2)
    score_cell = get(3)
    away = get(4)
    corner_cell = get(5)

    score_m = SCORE_RE.search(score_cell) if score_cell else None
    if home and away and score_m:
        ft_h = int(score_m.group(1))
        ft_a = int(score_m.group(2))
        ht_h = int(score_m.group(3)) if score_m.group(3) else None
        ht_a = int(score_m.group(4)) if score_m.group(4) else None
        date_val = normalize_date(date_cell) or ""

        ft_corner, ht_corner = parse_corner_cell(corner_cell)
        corner_home, corner_away = (ft_corner if ft_corner else (None, None))
        corner_ht_home, corner_ht_away = (ht_corner if ht_corner else (None, None))

        return MatchRow(
            league=league, date=date_val,
            home=home, away=away,
            ft_home=ft_h, ft_away=ft_a,
            ht_home=ht_h, ht_away=ht_a,
            corner_home=corner_home, corner_away=corner_away,
            corner_ht_home=corner_ht_home, corner_ht_away=corner_ht_away,
        )

    # fallback: find score anywhere
    score_idx = None
    score_m = None
    for i, c in enumerate(cells):
        m = SCORE_RE.search((c or "").strip())
        if m:
            score_idx = i
            score_m = m
            break
    if score_m is None or score_idx is None:
        return None

    ft_h = int(score_m.group(1))
    ft_a = int(score_m.group(2))
    ht_h = int(score_m.group(3)) if score_m.group(3) else None
    ht_a = int(score_m.group(4)) if score_m.group(4) else None

    home2 = next(((cells[i] or "").strip() for i in range(score_idx - 1, -1, -1) if (cells[i] or "").strip()), None)
    away2 = next(((cells[i] or "").strip() for i in range(score_idx + 1, len(cells)) if (cells[i] or "").strip()), None)
    if not home2 or not away2:
        return None

    date_val2 = ""
    for c in cells:
        d = normalize_date(c)
        if d:
            date_val2 = d
            break

    corner_home, corner_away = None, None
    corner_ht_home, corner_ht_away = None, None
    for i in range(score_idx + 1, min(score_idx + 10, len(cells))):
        ft_corner, ht_corner = parse_corner_cell(cells[i])
        if ft_corner:
            corner_home, corner_away = ft_corner
            if ht_corner:
                corner_ht_home, corner_ht_away = ht_corner
            break

    return MatchRow(
        league=league, date=date_val2,
        home=home2, away=away2,
        ft_home=ft_h, ft_away=ft_a,
        ht_home=ht_h, ht_away=ht_a,
        corner_home=corner_home, corner_away=corner_away,
        corner_ht_home=corner_ht_home, corner_ht_away=corner_ht_away,
    )

def parse_matches_from_table_html(table_html: str) -> List[MatchRow]:
    out: List[MatchRow] = []
    rows = extract_table_rows_from_html(table_html)
    for cells in rows:
        m = parse_match_from_cells(cells)
        if m:
            out.append(m)
    return sort_matches_desc(dedupe_matches(out))

# ======================
# STANDINGS
# ======================
def _to_int(x: str) -> Optional[int]:
    try:
        x = (x or "").strip()
        if x in {"", "-", "—", "---"}:
            return None
        return int(x)
    except Exception:
        return None

def parse_standings_table_rows(rows: List[List[str]]) -> List[StandRow]:
    wanted = {"Total", "Home", "Away", "Last 6", "Last6"}
    out: List[StandRow] = []

    for cells in rows:
        if not cells:
            continue
        head = (cells[0] or "").strip()
        if head not in wanted:
            continue
        label = "Last 6" if head == "Last6" else head

        def g(i): return (cells[i] if i < len(cells) else "") or ""

        r = StandRow(
            ft=label,
            matches=_to_int(g(1)),
            win=_to_int(g(2)),
            draw=_to_int(g(3)),
            loss=_to_int(g(4)),
            scored=_to_int(g(5)),
            conceded=_to_int(g(6)),
            pts=_to_int(g(7)),
            rank=_to_int(g(8)),
            rate=g(9).strip() if g(9) else None
        )
        if r.matches is not None and not (1 <= r.matches <= 80):
            continue
        if any(x.ft == r.ft for x in out):
            continue
        out.append(r)

    order = {"Total": 0, "Home": 1, "Away": 2, "Last 6": 3}
    out.sort(key=lambda x: order.get(x.ft, 99))
    return out

def extract_standings_for_team(page_source: str, team_name: str) -> List[StandRow]:
    team_key = norm_key(team_name)
    for tbl in extract_tables_html(page_source):
        text_low = strip_tags_keep_text(tbl).lower()
        required_keywords = ["matches", "win", "draw", "loss", "scored", "conceded"]
        if not all(k in text_low for k in required_keywords):
            continue
        if team_key and team_key not in norm_key(strip_tags_keep_text(tbl)):
            continue
        rows = extract_table_rows_from_html(tbl)
        parsed = parse_standings_table_rows(rows)
        if parsed:
            return parsed
    return []

def standings_to_splits(rows: List[StandRow]) -> Dict[str, Optional[SplitGFGA]]:
    mp: Dict[str, Optional[SplitGFGA]] = {"Total": None, "Home": None, "Away": None, "Last 6": None}
    for r in rows:
        if r.matches and r.scored is not None and r.conceded is not None:
            mp[r.ft] = SplitGFGA(r.matches, r.scored, r.conceded)
    return mp

# ======================
# ODDS (Bet365 Initial 1X2) - graceful
# ======================
def _extract_first_float(s: str) -> Optional[float]:
    if not s:
        return None
    m = FLOAT_RE.search(s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def extract_bet365_initial_1x2_from_oddscomp_html(odds_html: str) -> Optional[Dict[str, float]]:
    if not odds_html:
        return None
    tr_m = re.search(r"(<tr\b[^>]*>.*?Bet365.*?</tr>)", odds_html, flags=re.I | re.S)
    if not tr_m:
        return None

    tr_html = tr_m.group(1)
    tds = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr_html, flags=re.I | re.S)
    if not tds:
        return None

    vals = []
    for td in tds:
        txt = strip_tags_keep_text(td)
        v = _extract_first_float(txt)
        vals.append(v)

    # Common layout: 1X2 might be around positions 5-7 (depends)
    if len(vals) >= 8:
        o1, ox, o2 = vals[5], vals[6], vals[7]
        if all(v is not None for v in [o1, ox, o2]):
            if all(1.01 <= float(v) <= 200 for v in [o1, ox, o2]):
                return {"1": float(o1), "X": float(ox), "2": float(o2)}
    return None

def extract_bet365_initial_odds(url: str, debug_lines: Optional[List[str]] = None) -> Optional[Dict[str, float]]:
    odds_url = build_oddscomp_url(url)
    if debug_lines is not None:
        debug_lines.append(f"ODDSCOMP url={odds_url}")
    try:
        html = safe_get(odds_url, referer=extract_base_domain(url), debug_lines=debug_lines)
        odds = extract_bet365_initial_1x2_from_oddscomp_html(html)
        if odds is None and debug_lines is not None:
            debug_lines.append("ODDSCOMP: Odds bulunamadı (maç finished olabilir, sayfada --- olabilir).")
        return odds
    except Exception as e:
        if debug_lines is not None:
            debug_lines.append(f"ODDSCOMP error: {e}")
        return None

# ======================
# PREVIOUS & H2H
# ======================
def extract_previous_from_page(page_source: str) -> Tuple[List[str], List[str]]:
    markers = ["Previous Scores Statistics", "Previous Scores", "Recent Matches"]
    tabs = []
    for marker in markers:
        found_tabs = section_tables_by_marker(page_source, marker, max_tables=10)
        if found_tabs:
            tabs = found_tabs
            break

    if not tabs:
        # fallback: first match-like tables
        all_tables = extract_tables_html(page_source)
        for t in all_tables:
            matches = parse_matches_from_table_html(t)
            if matches and len(matches) >= 3:
                tabs.append(t)
            if len(tabs) >= 4:
                break

    if not tabs:
        return [], []

    match_tables: List[str] = []
    for t in tabs:
        ms = parse_matches_from_table_html(t)
        if ms:
            match_tables.append(t)
        if len(match_tables) >= 2:
            break

    if len(match_tables) == 0:
        return [], []
    if len(match_tables) == 1:
        return [match_tables[0]], []
    return [match_tables[0]], [match_tables[1]]

def extract_h2h_matches(page_source: str, home_team: str, away_team: str) -> List[MatchRow]:
    markers = ["Head to Head Statistics", "Head to Head", "H2H Statistics", "H2H", "VS Statistics"]
    for mk in markers:
        tabs = section_tables_by_marker(page_source, mk, max_tables=5)
        for t in tabs:
            cand = parse_matches_from_table_html(t)
            if not cand:
                continue
            pair_count = sum(1 for m in cand if is_h2h_pair(m, home_team, away_team))
            if pair_count >= 2:
                return cand

    # fallback: best pair table among all tables
    best_pair = 0
    best_list: List[MatchRow] = []
    for tbl in extract_tables_html(page_source):
        cand = parse_matches_from_table_html(tbl)
        if not cand:
            continue
        pair_count = sum(1 for m in cand if is_h2h_pair(m, home_team, away_team))
        if pair_count > best_pair:
            best_pair = pair_count
            best_list = cand
    return best_list

def filter_same_league_matches(matches: List[MatchRow], league_code: str) -> Tuple[List[MatchRow], bool]:
    """
    Returns (filtered, applied)
    applied=True only if league_code exists AND filtering actually reduces or changes set.
    """
    if not matches:
        return matches, False
    code_k = norm_key(league_code)
    if not code_k:
        return matches, False
    exact = [m for m in matches if norm_key(m.league) == code_k]
    if exact:
        changed = len(exact) != len(matches)
        return exact, changed
    return matches, False

def filter_team_home_only(matches: List[MatchRow], team: str) -> List[MatchRow]:
    tk = norm_key(team)
    return [m for m in matches if norm_key(m.home) == tk]

def filter_team_away_only(matches: List[MatchRow], team: str) -> List[MatchRow]:
    tk = norm_key(team)
    return [m for m in matches if norm_key(m.away) == tk]

# ======================
# PREV STATS
# ======================
def build_prev_stats(team: str, matches: List[MatchRow]) -> TeamPrevStats:
    tkey = norm_key(team)
    st = TeamPrevStats(name=team)
    if not matches:
        return st

    def team_gf_ga(m: MatchRow) -> Tuple[int, int, Optional[int], Optional[int]]:
        if norm_key(m.home) == tkey:
            return m.ft_home, m.ft_away, m.corner_home, m.corner_away
        return m.ft_away, m.ft_home, m.corner_away, m.corner_home

    gfs, gas, cfor, cagn = [], [], [], []
    clean_sheets = 0
    scored_matches = 0

    for m in matches:
        gf, ga, cf, ca = team_gf_ga(m)
        gfs.append(gf); gas.append(ga)
        if cf is not None: cfor.append(cf)
        if ca is not None: cagn.append(ca)
        if ga == 0: clean_sheets += 1
        if gf > 0: scored_matches += 1

    st.n_total = len(matches)
    st.gf_total = sum(gfs) / st.n_total if st.n_total else 0.0
    st.ga_total = sum(gas) / st.n_total if st.n_total else 0.0
    st.clean_sheets = clean_sheets
    st.scored_matches = scored_matches
    st.corners_for = (sum(cfor) / len(cfor)) if cfor else 0.0
    st.corners_against = (sum(cagn) / len(cagn)) if cagn else 0.0

    home_ms = [m for m in matches if norm_key(m.home) == tkey]
    away_ms = [m for m in matches if norm_key(m.away) == tkey]

    st.n_home = len(home_ms)
    if st.n_home:
        st.gf_home = sum(m.ft_home for m in home_ms) / st.n_home
        st.ga_home = sum(m.ft_away for m in home_ms) / st.n_home

    st.n_away = len(away_ms)
    if st.n_away:
        st.gf_away = sum(m.ft_away for m in away_ms) / st.n_away
        st.ga_away = sum(m.ft_home for m in away_ms) / st.n_away

    return st

# ======================
# POISSON
# ======================
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k > 170:
        return 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0.0

def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))

# ======================
# CORNERS (Poisson O/U) + NET
# ======================
def analyze_corners(home_prev: TeamPrevStats, away_prev: TeamPrevStats, h2h_matches: List[MatchRow]) -> Dict[str, Any]:
    h2h_total, h2h_home, h2h_away = [], [], []
    for m in h2h_matches[:H2H_N]:
        if m.corner_home is not None and m.corner_away is not None:
            h2h_total.append(m.corner_home + m.corner_away)
            h2h_home.append(m.corner_home)
            h2h_away.append(m.corner_away)

    h2h_total_avg = sum(h2h_total) / len(h2h_total) if h2h_total else 0.0
    h2h_home_avg = sum(h2h_home) / len(h2h_home) if h2h_home else 0.0
    h2h_away_avg = sum(h2h_away) / len(h2h_away) if h2h_away else 0.0

    # PSS corner means
    pss_home_for = home_prev.corners_for
    pss_home_against = home_prev.corners_against
    pss_away_for = away_prev.corners_for
    pss_away_against = away_prev.corners_against

    # Predict corners for each side
    if h2h_total_avg > 0:
        predicted_home = 0.6 * h2h_home_avg + 0.4 * ((pss_home_for + pss_away_against) / 2.0)
        predicted_away = 0.6 * h2h_away_avg + 0.4 * ((pss_away_for + pss_home_against) / 2.0)
    elif (pss_home_for > 0 or pss_away_for > 0):
        predicted_home = (pss_home_for + pss_away_against) / 2.0
        predicted_away = (pss_away_for + pss_home_against) / 2.0
    else:
        predicted_home = 0.0
        predicted_away = 0.0

    total_lam = max(0.01, predicted_home + predicted_away)

    preds = {}
    net = {}
    for line in [8.5, 9.5, 10.5]:
        # Over line => X >= floor(line)+1
        k = int(math.floor(line))
        p_le_k = poisson_cdf(k, total_lam)  # P(X <= k)
        p_over = 1.0 - p_le_k              # P(X >= k+1)
        p_under = 1.0 - p_over

        preds[f"O{line}"] = float(p_over)
        preds[f"U{line}"] = float(p_under)

        if p_over >= p_under:
            st = pick_status(float(p_over))
            net[str(line)] = {"pick": f"{line} OVER", "p": float(p_over), **st}
        else:
            st = pick_status(float(p_under))
            net[str(line)] = {"pick": f"{line} UNDER", "p": float(p_under), **st}

    data_points = len(h2h_total) + (1 if (pss_home_for > 0 or pss_away_for > 0) else 0)
    confidence = "high" if data_points >= 8 else ("medium" if data_points >= 4 else "low")

    return {
        "lambda_total": float(round(total_lam, 2)),
        "pred_home": float(round(predicted_home, 2)),
        "pred_away": float(round(predicted_away, 2)),
        "h2h_count": int(len(h2h_total)),
        "pss_available": bool(pss_home_for > 0 or pss_away_for > 0),
        "confidence": confidence,
        "predictions": preds,
        "net": net
    }

# ======================
# LAMBDA COMPUTATION
# ======================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in w.values())
    if s <= 1e-9:
        return {}
    return {k: max(0.0, v) / s for k, v in w.items()}

def compute_component_standings(st_home: Dict[str, Optional[SplitGFGA]],
                                st_away: Dict[str, Optional[SplitGFGA]]) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    hh = st_home.get("Home") or st_home.get("Total")
    aa = st_away.get("Away") or st_away.get("Total")
    if not hh or not aa or hh.matches < 3 or aa.matches < 3:
        return None
    lam_h = (hh.gf_pg + aa.ga_pg) / 2.0
    lam_a = (aa.gf_pg + hh.ga_pg) / 2.0
    meta = {
        "home_split": {"matches": hh.matches, "gf_pg": round(hh.gf_pg, 3), "ga_pg": round(hh.ga_pg, 3)},
        "away_split": {"matches": aa.matches, "gf_pg": round(aa.gf_pg, 3), "ga_pg": round(aa.ga_pg, 3)},
        "formula": "standing: (home_gfpg + away_gapg)/2 ; (away_gfpg + home_gapg)/2"
    }
    return lam_h, lam_a, meta

def compute_component_pss(home_prev: TeamPrevStats, away_prev: TeamPrevStats) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    if home_prev.n_total < 3 or away_prev.n_total < 3:
        return None
    h_gf, h_ga = home_prev.gf_total, home_prev.ga_total
    a_gf, a_ga = away_prev.gf_total, away_prev.ga_total
    lam_h = (h_gf + a_ga) / 2.0
    lam_a = (a_gf + h_ga) / 2.0
    meta = {
        "home_n": home_prev.n_total, "away_n": away_prev.n_total,
        "home_gf": round(h_gf, 3), "home_ga": round(h_ga, 3),
        "away_gf": round(a_gf, 3), "away_ga": round(a_ga, 3),
        "formula": "pss: (home_gf + away_ga)/2 ; (away_gf + home_ga)/2"
    }
    return lam_h, lam_a, meta

def compute_component_h2h(h2h_matches: List[MatchRow], home_team: str, away_team: str) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    if not h2h_matches or len(h2h_matches) < 3:
        return None
    hk, ak = norm_key(home_team), norm_key(away_team)
    used = h2h_matches[:H2H_N]
    hg, ag = [], []
    for m in used:
        if norm_key(m.home) == hk and norm_key(m.away) == ak:
            hg.append(m.ft_home); ag.append(m.ft_away)
        elif norm_key(m.home) == ak and norm_key(m.away) == hk:
            hg.append(m.ft_away); ag.append(m.ft_home)
    if len(hg) < 3:
        return None
    lam_h = sum(hg) / len(hg)
    lam_a = sum(ag) / len(ag)
    meta = {"matches": len(hg), "home_avg": round(lam_h, 3), "away_avg": round(lam_a, 3)}
    return lam_h, lam_a, meta

def clamp_lambda(lh: float, la: float) -> Tuple[float, float, List[str]]:
    warn = []
    def c(x: float, name: str) -> float:
        if x < 0.15:
            warn.append(f"{name} too low ({x:.2f}) -> 0.15")
            return 0.15
        if x > 3.80:
            warn.append(f"{name} too high ({x:.2f}) -> 3.80")
            return 3.80
        return x
    return c(lh, "lambda_home"), c(la, "lambda_away"), warn

def compute_lambdas(st_home_s: Dict[str, Optional[SplitGFGA]],
                    st_away_s: Dict[str, Optional[SplitGFGA]],
                    home_prev: TeamPrevStats,
                    away_prev: TeamPrevStats,
                    h2h_used: List[MatchRow],
                    home_team: str,
                    away_team: str) -> Tuple[float, float, Dict[str, Any]]:
    info: Dict[str, Any] = {"components": {}, "weights_used": {}, "warnings": []}
    comps: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}

    stc = compute_component_standings(st_home_s, st_away_s)
    if stc: comps["standing"] = stc

    pssc = compute_component_pss(home_prev, away_prev)
    if pssc: comps["pss"] = pssc

    h2c = compute_component_h2h(h2h_used, home_team, away_team)
    if h2c: comps["h2h"] = h2c

    w = {}
    if "standing" in comps: w["standing"] = W_ST_BASE
    if "pss" in comps:      w["pss"] = W_PSS_BASE
    if "h2h" in comps:      w["h2h"] = W_H2H_BASE

    w_norm = normalize_weights(w)
    info["weights_used"] = w_norm

    if not w_norm:
        info["warnings"].append("insufficient data -> default lambdas 1.20 / 1.20")
        lh, la = 1.20, 1.20
    else:
        lh = 0.0; la = 0.0
        for k, wk in w_norm.items():
            ch, ca, meta = comps[k]
            info["components"][k] = {"lam_home": round(ch, 4), "lam_away": round(ca, 4), "meta": meta}
            lh += wk * ch
            la += wk * ca

    lh, la, clamp_warn = clamp_lambda(lh, la)
    info["warnings"].extend(clamp_warn)
    return lh, la, info

# ======================
# POISSON MATRIX + MARKET PROBS
# ======================
def build_score_matrix(lh: float, la: float, max_g: int = 5) -> Dict[Tuple[int, int], float]:
    mat = {}
    for h in range(max_g + 1):
        ph = poisson_pmf(h, lh)
        for a in range(max_g + 1):
            mat[(h, a)] = ph * poisson_pmf(a, la)
    return mat

def market_probs_from_matrix(mat: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    p1 = sum(p for (h, a), p in mat.items() if h > a)
    px = sum(p for (h, a), p in mat.items() if h == a)
    p2 = sum(p for (h, a), p in mat.items() if h < a)
    btts = sum(p for (h, a), p in mat.items() if h >= 1 and a >= 1)

    out = {"1": p1, "X": px, "2": p2, "BTTS": btts, "NO_BTTS": 1.0 - btts}
    # totals
    for ln in [0.5, 1.5, 2.5, 3.5]:
        need = int(math.floor(ln) + 1)
        out[f"O{ln}"] = sum(p for (h, a), p in mat.items() if (h + a) >= need)
        out[f"U{ln}"] = 1.0 - out[f"O{ln}"]
    return out

def top_scores_from_matrix(mat: Dict[Tuple[int, int], float], top_n: int = 7) -> List[Tuple[str, float]]:
    items = sorted(mat.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(f"{h}-{a}", float(p)) for (h, a), p in items]

def monte_carlo(lh: float, la: float, n: int, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    hg = rng.poisson(lh, size=n)
    ag = rng.poisson(la, size=n)
    total = hg + ag

    def p(mask) -> float:
        return float(np.mean(mask))

    cnt = Counter(zip(hg.tolist(), ag.tolist()))
    top10 = cnt.most_common(10)
    top10_list = [(f"{h}-{a}", float(c / n)) for (h, a), c in top10]

    return {
        "p": {
            "1": p(hg > ag),
            "X": p(hg == ag),
            "2": p(hg < ag),
            "BTTS": p((hg >= 1) & (ag >= 1)),
            "NO_BTTS": p((hg == 0) | (ag == 0)),
            "O2.5": p(total >= 3),
            "U2.5": p(total <= 2),
        },
        "top10": top10_list
    }

def model_agreement(p_po: Dict[str, float], p_mc: Dict[str, float]) -> Tuple[float, str]:
    keys = ["1", "X", "2", "BTTS", "O2.5", "U2.5"]
    diffs = [abs(p_po.get(k, 0) - p_mc.get(k, 0)) for k in keys]
    d = max(diffs) if diffs else 0.0
    if d <= 0.03:
        return float(d), "excellent"
    if d <= 0.06:
        return float(d), "good"
    if d <= 0.10:
        return float(d), "medium"
    return float(d), "weak"

def blend_probs(p1: Dict[str, float], p2: Dict[str, float], alpha: float) -> Dict[str, float]:
    out = {}
    for k in set(list(p1.keys()) + list(p2.keys())):
        out[k] = float(alpha * p1.get(k, 0.0) + (1.0 - alpha) * p2.get(k, 0.0)) if (k in p1 and k in p2) else float(p1.get(k, p2.get(k, 0.0)))
    return out

# ======================
# NET PICKS
# ======================
def net_pick_ou25(blended: Dict[str, float]) -> Dict[str, Any]:
    p_o = float(blended.get("O2.5", 0.0))
    p_u = float(blended.get("U2.5", 0.0))
    if p_o >= p_u:
        st = pick_status(p_o)
        return {"market": "OU2.5", "pick": "OVER", "p": p_o, **st}
    st = pick_status(p_u)
    return {"market": "OU2.5", "pick": "UNDER", "p": p_u, **st}

def net_pick_btts(blended: Dict[str, float]) -> Dict[str, Any]:
    p_yes = float(blended.get("BTTS", 0.0))
    p_no = float(blended.get("NO_BTTS", 1.0 - p_yes))
    if p_yes >= p_no:
        st = pick_status(p_yes)
        return {"market": "BTTS", "pick": "YES", "p": p_yes, **st}
    st = pick_status(p_no)
    return {"market": "BTTS", "pick": "NO", "p": p_no, **st}

# ======================
# MAIN ANALYSIS
# ======================
def analyze_nowgoal(url: str, mc_runs: int = MC_RUNS_DEFAULT, debug: bool = False) -> Dict[str, Any]:
    debug_lines: Optional[List[str]] = [] if debug else None

    h2h_url = build_h2h_url(url)
    if debug_lines is not None:
        debug_lines.append(f"INPUT url={url}")
        debug_lines.append(f"H2H url={h2h_url}")
    logger.info(f"INPUT url={url}")
    logger.info(f"H2H url={h2h_url}")

    html = safe_get(h2h_url, referer=extract_base_domain(url), debug_lines=debug_lines)

    tables = extract_tables_html(html)
    if debug_lines is not None:
        debug_lines.append(f"HTML tables_total={len(tables)}")

    home_team, away_team = parse_teams_from_title(html)
    if debug_lines is not None:
        debug_lines.append(f"PARSED teams home='{home_team}' away='{away_team}'")
    if not home_team or not away_team:
        raise RuntimeError("Takım isimleri çıkarılamadı (title parse başarısız)")

    # Standings
    st_home_rows = extract_standings_for_team(html, home_team)
    st_away_rows = extract_standings_for_team(html, away_team)
    st_home = standings_to_splits(st_home_rows)
    st_away = standings_to_splits(st_away_rows)
    if debug_lines is not None:
        debug_lines.append(f"STANDINGS parsed home_rows={len(st_home_rows)} away_rows={len(st_away_rows)}")

    # League code: try from html, else guess later
    league_code = extract_league_code_from_html(html)
    if debug_lines is not None:
        debug_lines.append(f"LEAGUE_CODE initial='{league_code}'")

    # H2H
    h2h_all = extract_h2h_matches(html, home_team, away_team)
    h2h_pair = sort_matches_desc(dedupe_matches([m for m in h2h_all if is_h2h_pair(m, home_team, away_team)]))
    if not league_code:
        league_code = guess_league_code_from_matches(h2h_pair)  # fallback
        if debug_lines is not None:
            debug_lines.append(f"LEAGUE_CODE fallback_from_H2H='{league_code}'")

    h2h_filt, h2h_same_applied = filter_same_league_matches(h2h_pair, league_code)
    h2h_used = h2h_filt[:H2H_N] if len(h2h_filt) >= 3 else h2h_pair[:H2H_N]

    if debug_lines is not None:
        debug_lines.append(f"H2H pair_matches={len(h2h_pair)}")
        debug_lines.append(f"H2H same_league_applied={h2h_same_applied} used={len(h2h_used)}")
        debug_lines.append("H2H sample(top3)=" + " | ".join([f"{m.league} {m.home} {m.ft_home}-{m.ft_away} {m.away} corner={m.corner_home}-{m.corner_away}" for m in h2h_used[:3]]))

    # PSS
    prev_home_tabs, prev_away_tabs = extract_previous_from_page(html)
    prev_home_raw = parse_matches_from_table_html(prev_home_tabs[0]) if prev_home_tabs else []
    prev_away_raw = parse_matches_from_table_html(prev_away_tabs[0]) if prev_away_tabs else []
    if debug_lines is not None:
        debug_lines.append(f"PSS raw parsed: home_raw={len(prev_home_raw)} away_raw={len(prev_away_raw)}")

    if not league_code:
        league_code = guess_league_code_from_matches(prev_home_raw + prev_away_raw)
        if debug_lines is not None:
            debug_lines.append(f"LEAGUE_CODE fallback_from_PSS='{league_code}'")

    prev_home_f, pss_home_applied = filter_same_league_matches(prev_home_raw, league_code)
    prev_away_f, pss_away_applied = filter_same_league_matches(prev_away_raw, league_code)

    prev_home_sel = filter_team_home_only(prev_home_f, home_team)[:RECENT_N]
    prev_away_sel = filter_team_away_only(prev_away_f, away_team)[:RECENT_N]

    home_prev = build_prev_stats(home_team, prev_home_sel)
    away_prev = build_prev_stats(away_team, prev_away_sel)

    if debug_lines is not None:
        debug_lines.append(f"PSS same_applied home={pss_home_applied} away={pss_away_applied}")
        debug_lines.append(f"PSS selected: home_homeOnly={len(prev_home_sel)} away_awayOnly={len(prev_away_sel)}")
        debug_lines.append(f"PSS AVG home: gf={home_prev.gf_total:.2f} ga={home_prev.ga_total:.2f}")
        debug_lines.append(f"PSS AVG away: gf={away_prev.gf_total:.2f} ga={away_prev.ga_total:.2f}")
        debug_lines.append("PSS home sample(top3)=" + " | ".join([f"{m.league} {m.home} {m.ft_home}-{m.ft_away} {m.away} corner={m.corner_home}-{m.corner_away}" for m in prev_home_sel[:3]]))
        debug_lines.append("PSS away sample(top3)=" + " | ".join([f"{m.league} {m.home} {m.ft_home}-{m.ft_away} {m.away} corner={m.corner_home}-{m.corner_away}" for m in prev_away_sel[:3]]))

    # Lambdas
    lam_h, lam_a, lam_info = compute_lambdas(
        st_home_s=st_home,
        st_away_s=st_away,
        home_prev=home_prev,
        away_prev=away_prev,
        h2h_used=h2h_used,
        home_team=home_team,
        away_team=away_team
    )
    if debug_lines is not None:
        debug_lines.append(f"LAMBDAS final λ_home={lam_h:.3f} λ_away={lam_a:.3f} total={lam_h+lam_a:.3f}")
        debug_lines.append(f"LAMBDA weights={lam_info.get('weights_used', {})}")
        debug_lines.append(f"LAMBDA warnings={lam_info.get('warnings', [])}")

    # Models
    mat = build_score_matrix(lam_h, lam_a, max_g=MAX_GOALS_FOR_MATRIX)
    poisson_probs = market_probs_from_matrix(mat)
    top7 = top_scores_from_matrix(mat, top_n=7)

    mc = monte_carlo(lam_h, lam_a, n=max(10_000, int(mc_runs)), seed=42)
    diff, diff_label = model_agreement(poisson_probs, mc["p"])
    blended = blend_probs(poisson_probs, mc["p"], alpha=BLEND_ALPHA)

    if debug_lines is not None:
        debug_lines.append(f"MODEL_AGREE diff={diff:.3f} label={diff_label}")

    # Net picks
    net_ou = net_pick_ou25(blended)
    net_btts = net_pick_btts(blended)

    # Corners
    corners = analyze_corners(home_prev, away_prev, h2h_used)

    # Odds (optional)
    odds = extract_bet365_initial_odds(url, debug_lines=debug_lines)  # may be None, no crash

    data = {
        "teams": {"home": home_team, "away": away_team},
        "league_code": league_code,
        "lambda": {"home": float(lam_h), "away": float(lam_a), "total": float(lam_h + lam_a), "info": lam_info},
        "top_scores": [{"score": s, "p": float(p)} for s, p in top7],
        "blended_probs": {k: float(v) for k, v in blended.items()},
        "model_agreement": {"diff": float(diff), "label": diff_label},
        "net": {
            "score": top7[0][0] if top7 else "",
            "ou25": net_ou,
            "btts": net_btts,
            "corners": corners["net"],
        },
        "corners": corners,
        "odds_bet365_initial_1x2": odds,  # can be None
        "sources": {
            "standings_used": bool(st_home_rows and st_away_rows),
            "pss_home_used": int(len(prev_home_sel)),
            "pss_away_used": int(len(prev_away_sel)),
            "h2h_used": int(len(h2h_used)),
        }
    }
    if debug_lines is not None:
        data["debug_lines"] = debug_lines
    return data

# ======================
# FLASK APP
# ======================
app = Flask(__name__)

@app.get("/")
def root():
    return jsonify({"ok": True, "service": "nowgoal-analyzer-api", "version": "4.2.2-robust"})

@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "healthy"})

@app.post("/analiz_et")
def analiz_et_route():
    try:
        payload = request.get_json(silent=True) or {}
    except Exception as e:
        return jsonify({"ok": False, "error": f"Geçersiz JSON: {e}"}), 400

    url = (payload.get("url") or "").strip()
    debug = bool(payload.get("debug", False))
    mc_runs = payload.get("mc_runs", MC_RUNS_DEFAULT)

    if not url:
        return jsonify({"ok": False, "error": "URL boş olamaz"}), 400
    if not re.match(r'^https?://', url):
        return jsonify({"ok": False, "error": "Geçersiz URL formatı"}), 400

    try:
        try:
            mc_runs = int(mc_runs)
        except Exception:
            mc_runs = MC_RUNS_DEFAULT
        if mc_runs < 100:
            mc_runs = 100
        if mc_runs > 100_000:
            mc_runs = 100_000

        data = analyze_nowgoal(url, mc_runs=mc_runs, debug=debug)
        net = data["net"]

        # Android'e kolay: structured net picks
        return jsonify({
            "ok": True,
            "teams": data["teams"],
            "league_code": data["league_code"],
            "score_top": net["score"],
            "picks": {
                "ou25": net["ou25"],
                "btts": net["btts"],
                "corners": net["corners"],
            },
            "debug_lines": data.get("debug_lines") if debug else None
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Analiz hatası: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.post("/analyze")
def analyze_route():
    try:
        payload = request.get_json(silent=True) or {}
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid JSON: {e}"}), 400

    url = (payload.get("url") or "").strip()
    debug = bool(payload.get("debug", False))
    mc_runs = payload.get("mc_runs", MC_RUNS_DEFAULT)

    if not url:
        return jsonify({"ok": False, "error": "url required"}), 400
    if not re.match(r'^https?://', url):
        return jsonify({"ok": False, "error": "Invalid URL"}), 400

    try:
        mc_runs = int(mc_runs)
        if mc_runs < 100:
            mc_runs = 100
        if mc_runs > 100_000:
            mc_runs = 100_000
    except Exception:
        mc_runs = MC_RUNS_DEFAULT

    try:
        data = analyze_nowgoal(url, mc_runs=mc_runs, debug=debug)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("NowGoal Analyzer v4.2.2 (ROBUST)")
        print("Usage: python script.py serve")
        print("POST /analiz_et payload: {url:'...', debug:true, mc_runs:10000}")
        print("POST /analyze   payload: {url:'...', debug:true, mc_runs:10000}")
