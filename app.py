# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - PRODUCTION v5.2 (PSS_ONLY VBA STYLE)
- MODEL: %100 PSS (Standings/H2H display only)
- Recency weighting: first 5 matches 1.20, older 0.80 (goals + corners)
- HT corner support if present in corner cell like "8-2(3-1)"
- Android-friendly response + comprehensive text report
"""

import re
import math
import time
import traceback
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import numpy as np
import requests
from flask import Flask, request, jsonify

# =============================================================================
# CONFIG
# =============================================================================
MODEL_MODE = "PSS_ONLY"   # "PSS_ONLY" (VBA style) or "BLENDED" (not used here)

MC_RUNS_DEFAULT = 5000
RECENT_N = 10
H2H_N = 10

# VBA style recency weights
RECENT_WEIGHT_N = 5
W_RECENT = 1.20
W_OLD = 0.80

# If you want to keep Poisson vs MC blend:
BLEND_ALPHA = 0.50

VALUE_MIN = 0.05
PROB_MIN = 0.55
KELLY_MIN = 0.02

MAX_GOALS_FOR_MATRIX = 5
MAX_CORNERS_FOR_MATRIX = 20

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# =============================================================================
# LOGGING
# =============================================================================
def log_error(msg: str, exc: Exception = None):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    if exc:
        print(traceback.format_exc(), file=sys.stderr, flush=True)

def log_info(msg: str):
    print(f"[INFO] {msg}", file=sys.stdout, flush=True)

# =============================================================================
# REGEX
# =============================================================================
DATE_ANY_RE = re.compile(r"\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{2}-\d{2}")
SCORE_RE = re.compile(r"(\d{1,2})-(\d{1,2})(?:\((\d{1,2})-(\d{1,2})\))?")
CORNER_FT_RE = re.compile(r"(\d{1,2})-(\d{1,2})")
CORNER_HT_RE = re.compile(r"\((\d{1,2})-(\d{1,2})\)")

# =============================================================================
# DATA
# =============================================================================
@dataclass
class MatchRow:
    league: str
    date: str
    home: str
    away: str
    fthome: int
    ftaway: int
    hthome: Optional[int] = None
    htaway: Optional[int] = None
    cornerhome: Optional[int] = None
    corneraway: Optional[int] = None
    cornerhthome: Optional[int] = None
    cornerhtaway: Optional[int] = None

@dataclass
class SplitGFGA:
    matches: int
    gf: int
    ga: int

    @property
    def gfpg(self) -> float:
        return self.gf / self.matches if self.matches else 0.0

    @property
    def gapg(self) -> float:
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

    # weighted averages (VBA style)
    gftotal: float = 0.0
    gatotal: float = 0.0
    ntotal: int = 0

    gfhome: float = 0.0
    gahome: float = 0.0
    nhome: int = 0

    gfaway: float = 0.0
    gaaway: float = 0.0
    naway: int = 0

    cornersfor: float = 0.0
    cornersagainst: float = 0.0
    corners_n: int = 0

    cornersfor_ht: float = 0.0
    cornersagainst_ht: float = 0.0
    corners_ht_n: int = 0

    cleansheets: int = 0
    scoredmatches: int = 0

# =============================================================================
# HELPERS
# =============================================================================
def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def normalize_date(d: str) -> Optional[str]:
    if not d:
        return None
    d = d.strip()
    m = DATE_ANY_RE.search(d)
    if not m:
        return None
    val = m.group(0)
    if re.match(r"\d{4}-\d{2}-\d{2}", val):
        yyyy, mm, dd = val.split("-")
        return f"{int(dd):02d}-{int(mm):02d}-{yyyy}"
    if re.match(r"\d{1,2}-\d{1,2}-\d{4}", val):
        dd, mm, yyyy = val.split("-")
        return f"{int(dd):02d}-{int(mm):02d}-{yyyy}"
    return None

def parse_date_key(datestr: str) -> Tuple[int, int, int]:
    if not datestr or not re.match(r"\d{2}-\d{2}-\d{4}", datestr):
        return (0, 0, 0)
    dd, mm, yyyy = datestr.split("-")
    return (int(yyyy), int(mm), int(dd))

def strip_tags_keep_text(s: str) -> str:
    s = re.sub(r"<script.*?</script>", "", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"<style.*?</style>", "", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"<.*?>", "", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_tables_html(page_source: str) -> List[str]:
    return [m.group(0) for m in re.finditer(r"<table.*?</table>", page_source or "", flags=re.IGNORECASE | re.DOTALL)]

def extract_table_rows_from_html(table_html: str) -> List[List[str]]:
    rows: List[List[str]] = []
    trs = re.findall(r"<tr.*?</tr>", table_html or "", flags=re.IGNORECASE | re.DOTALL)
    for tr in trs:
        cells = re.findall(r"<t[dh].*?</t[dh]>", tr, flags=re.IGNORECASE | re.DOTALL)
        if not cells:
            continue
        cleaned = [strip_tags_keep_text(c) for c in cells]
        normalized = []
        for c in cleaned:
            c = (c or "").strip()
            if c in ("", "-"):
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

def safe_get(url: str, timeout: int = 20, retries: int = 2, referer: Optional[str] = None) -> str:
    last_err = None
    headers = dict(HEADERS)
    if referer:
        headers["Referer"] = referer
    for attempt in range(retries + 1):
        try:
            log_info(f"Fetching {url} (attempt {attempt + 1})")
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            log_info(f"Fetch OK: {url}")
            return r.text
        except requests.exceptions.Timeout as e:
            last_err = e
            log_error(f"Timeout: {url}", e)
            if attempt < retries:
                time.sleep(0.5)
        except Exception as e:
            last_err = e
            log_error(f"Fetch error: {url}", e)
            if attempt < retries:
                time.sleep(0.7)
    raise RuntimeError(f"Fetch failed after {retries + 1} attempts: {url} - {last_err}")

def extract_match_id(url: str) -> str:
    m = re.search(r"(?:h2h-|match/h2h-)(\d+)", url)
    if m:
        return m.group(1)
    nums = re.findall(r"\d{6,}", url)
    if not nums:
        raise ValueError("Match ID not found")
    return nums[-1]

def extract_base_domain(url: str) -> str:
    m = re.match(r"(https?://[^/]+)", url.strip())
    return m.group(1) if m else "https://live3.nowgoal26.com"

def build_h2h_url(url: str) -> str:
    match_id = extract_match_id(url)
    base = extract_base_domain(url)
    return f"{base}/match/h2h-{match_id}"

def parse_teams_from_title(html: str) -> Tuple[str, str]:
    og_match = re.search(
        r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE
    )
    if og_match:
        title_text = og_match.group(1).strip()
        log_info(f"og:title: {title_text}")
        vs_match = re.search(r"(.+?)\s+VS\s+(.+?)(?:\s*-|\s*$)", title_text, flags=re.IGNORECASE)
        if vs_match:
            return vs_match.group(1).strip(), vs_match.group(2).strip()

    title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, flags=re.IGNORECASE)
    if title_match:
        title_text = title_match.group(1).strip()
        log_info(f"title: {title_text}")
        vs_match = re.search(r"(.+?)\s+(?:VS|vs)\s+(.+?)(?:\s*-|\s*$)", title_text, flags=re.IGNORECASE)
        if vs_match:
            return vs_match.group(1).strip(), vs_match.group(2).strip()

    h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, flags=re.IGNORECASE)
    if h1_match:
        h1_text = h1_match.group(1).strip()
        vs_match = re.search(r"(.+?)\s+(?:vs|VS)\s+(.+?)(?:\s+Live|\s*$)", h1_text, flags=re.IGNORECASE)
        if vs_match:
            return vs_match.group(1).strip(), vs_match.group(2).strip()

    return "", ""

def sort_matches_desc(matches: List[MatchRow]) -> List[MatchRow]:
    # keep stable ordering when dates are missing
    indexed = [(i, m) for i, m in enumerate(matches)]
    has_real_date = any(parse_date_key(m.date) != (0, 0, 0) for _, m in indexed)
    if not has_real_date:
        return matches
    return [m for _, m in sorted(indexed, key=lambda im: (parse_date_key(im[1].date), -im[0]), reverse=True)]

def dedupe_matches(matches: List[MatchRow]) -> List[MatchRow]:
    seen = set()
    out = []
    for m in matches:
        key = (m.league, m.date, m.home, m.away, m.fthome, m.ftaway, m.cornerhome, m.corneraway, m.cornerhthome, m.cornerhtaway)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out

def is_h2h_pair(m: MatchRow, hometeam: str, awayteam: str) -> bool:
    hk, ak = norm_key(hometeam), norm_key(awayteam)
    mh, ma = norm_key(m.home), norm_key(m.away)
    return (mh == hk and ma == ak) or (mh == ak and ma == hk)

# =============================================================================
# PARSERS
# =============================================================================
def parse_corner_cell(cell: str) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    if not cell:
        return None, None
    txt = (cell or "").strip()
    if txt in ("", "-"):
        return None, None

    ftm = CORNER_FT_RE.search(txt)
    htm = CORNER_HT_RE.search(txt)

    ft = (int(ftm.group(1)), int(ftm.group(2))) if ftm else None
    ht = (int(htm.group(1)), int(htm.group(2))) if htm else None
    return ft, ht

def parse_match_from_cells(cells: List[str]) -> Optional[MatchRow]:
    if not cells:
        return None

    def get(i: int) -> str:
        return (cells[i] or "").strip() if i < len(cells) else ""

    # common NowGoal layout: league, date, home, score, away, corner
    league = get(0)
    datecell = get(1)
    home = get(2)
    scorecell = get(3)
    away = get(4)
    cornercell = get(5)

    scorem = SCORE_RE.search(scorecell) if scorecell else None
    if home and away and scorem:
        fth = int(scorem.group(1))
        fta = int(scorem.group(2))
        hth = int(scorem.group(3)) if scorem.group(3) else None
        hta = int(scorem.group(4)) if scorem.group(4) else None
        dateval = normalize_date(datecell) or ""

        ftcorner, htcorner = parse_corner_cell(cornercell)
        cornerhome, corneraway = ftcorner if ftcorner else (None, None)
        cornerhthome, cornerhtaway = htcorner if htcorner else (None, None)

        return MatchRow(
            league=league or "",
            date=dateval,
            home=home,
            away=away,
            fthome=fth,
            ftaway=fta,
            hthome=hth,
            htaway=hta,
            cornerhome=cornerhome,
            corneraway=corneraway,
            cornerhthome=cornerhthome,
            cornerhtaway=cornerhtaway,
        )

    # fallback: search any cell for score
    score_idx = None
    scorem = None
    for i, c in enumerate(cells):
        c0 = (c or "").strip()
        m = SCORE_RE.search(c0)
        if m:
            score_idx = i
            scorem = m
            break
    if scorem is None or score_idx is None:
        return None

    fth = int(scorem.group(1))
    fta = int(scorem.group(2))
    hth = int(scorem.group(3)) if scorem.group(3) else None
    hta = int(scorem.group(4)) if scorem.group(4) else None

    home2, away2 = None, None
    for i in range(score_idx - 1, -1, -1):
        if (cells[i] or "").strip():
            home2 = (cells[i] or "").strip()
            break
    for i in range(score_idx + 1, len(cells)):
        if (cells[i] or "").strip():
            away2 = (cells[i] or "").strip()
            break
    if not home2 or not away2:
        return None

    league2 = (cells[0] or "").strip()
    dateval2 = ""
    for c in cells:
        d = normalize_date(c)
        if d:
            dateval2 = d
            break

    cornerhome, corneraway = None, None
    cornerhthome, cornerhtaway = None, None
    for i in range(score_idx + 1, min(score_idx + 10, len(cells))):
        ftcorner, htcorner = parse_corner_cell(cells[i])
        if ftcorner:
            cornerhome, corneraway = ftcorner
        if htcorner:
            cornerhthome, cornerhtaway = htcorner
        if ftcorner or htcorner:
            break

    return MatchRow(
        league=league2,
        date=dateval2,
        home=home2,
        away=away2,
        fthome=fth,
        ftaway=fta,
        hthome=hth,
        htaway=hta,
        cornerhome=cornerhome,
        corneraway=corneraway,
        cornerhthome=cornerhthome,
        cornerhtaway=cornerhtaway,
    )

def parse_matches_from_table_html(table_html: str) -> List[MatchRow]:
    out: List[MatchRow] = []
    rows = extract_table_rows_from_html(table_html)
    for cells in rows:
        m = parse_match_from_cells(cells)
        if m:
            out.append(m)
    return sort_matches_desc(dedupe_matches(out))

# =============================================================================
# STANDINGS (DISPLAY ONLY)
# =============================================================================
def to_int(x: str) -> Optional[int]:
    try:
        x = (x or "").strip()
        if x in ("", "-"):
            return None
        return int(x)
    except Exception:
        return None

def parse_standings_table_rows(rows: List[List[str]]) -> List[StandRow]:
    wanted = ["Total", "Home", "Away", "Last 6", "Last6"]
    out: List[StandRow] = []
    for cells in rows:
        if not cells:
            continue
        head = (cells[0] or "").strip()
        if head not in wanted:
            continue
        label = "Last 6" if head == "Last6" else head

        def g(i):
            return cells[i] if i < len(cells) else ""

        r = StandRow(
            ft=label,
            matches=to_int(g(1)),
            win=to_int(g(2)),
            draw=to_int(g(3)),
            loss=to_int(g(4)),
            scored=to_int(g(5)),
            conceded=to_int(g(6)),
            pts=to_int(g(7)),
            rank=to_int(g(8)),
            rate=g(9).strip() if g(9) else None,
        )
        if r.matches is not None and not (1 <= r.matches <= 80):
            continue
        if any(x.ft == r.ft for x in out):
            continue
        out.append(r)

    order = {"Total": 0, "Home": 1, "Away": 2, "Last 6": 3}
    out.sort(key=lambda x: order.get(x.ft, 99))
    return out

def extract_standings_for_team(page_source: str, teamname: str) -> List[StandRow]:
    team_key = norm_key(teamname)
    for tbl in extract_tables_html(page_source):
        text_low = strip_tags_keep_text(tbl).lower()
        required = ["matches", "win", "draw", "loss", "scored", "conceded"]
        if not all(k in text_low for k in required):
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

# =============================================================================
# PREVIOUS / H2H EXTRACTION
# =============================================================================
def extract_previous_from_page(page_source: str) -> Tuple[str, str]:
    markers = ["Previous Scores Statistics", "Previous Scores", "Recent Matches"]
    tabs: List[str] = []
    for marker in markers:
        found = section_tables_by_marker(page_source, marker, max_tables=10)
        if found:
            tabs = found
            break

    if not tabs:
        all_tables = extract_tables_html(page_source)
        for t in all_tables:
            matches = parse_matches_from_table_html(t)
            if matches and len(matches) >= 3:
                tabs.append(t)
            if len(tabs) >= 4:
                break

    if not tabs:
        return "", ""

    match_tables: List[str] = []
    for t in tabs:
        ms = parse_matches_from_table_html(t)
        if ms:
            match_tables.append(t)
        if len(match_tables) >= 2:
            break

    if len(match_tables) == 0:
        return "", ""
    if len(match_tables) == 1:
        return match_tables[0], ""
    return match_tables[0], match_tables[1]

def extract_h2h_matches(page_source: str, hometeam: str, awayteam: str) -> List[MatchRow]:
    markers = ["Head to Head Statistics", "Head to Head", "H2H Statistics", "H2H", "VS Statistics"]
    for mk in markers:
        tabs = section_tables_by_marker(page_source, mk, max_tables=5)
        for t in tabs:
            cand = parse_matches_from_table_html(t)
            if not cand:
                continue
            pair_count = sum(1 for m in cand if is_h2h_pair(m, hometeam, awayteam))
            if pair_count >= 2:
                return cand

    best_pair = 0
    best_list: List[MatchRow] = []
    for tbl in extract_tables_html(page_source):
        cand = parse_matches_from_table_html(tbl)
        if not cand:
            continue
        pair_count = sum(1 for m in cand if is_h2h_pair(m, hometeam, awayteam))
        if pair_count > best_pair:
            best_pair = pair_count
            best_list = cand
    return best_list

def filter_same_league_matches(matches: List[MatchRow], leaguename: str) -> List[MatchRow]:
    if not leaguename:
        return matches
    lk = norm_key(leaguename)
    out = []
    for m in matches:
        ml = norm_key(m.league)
        if lk and (lk in ml or ml in lk):
            out.append(m)
    return out if out else matches

def filter_team_home_only(matches: List[MatchRow], team: str) -> List[MatchRow]:
    tk = norm_key(team)
    return [m for m in matches if norm_key(m.home) == tk]

def filter_team_away_only(matches: List[MatchRow], team: str) -> List[MatchRow]:
    tk = norm_key(team)
    return [m for m in matches if norm_key(m.away) == tk]

# =============================================================================
# VBA STYLE WEIGHTING
# =============================================================================
def weighted_mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    wsum = 0.0
    vsum = 0.0
    for i, v in enumerate(vals, start=1):
        w = W_RECENT if i <= RECENT_WEIGHT_N else W_OLD
        vsum += float(v) * w
        wsum += w
    return vsum / wsum if wsum > 1e-9 else 0.0

def build_prev_stats(team: str, matches: List[MatchRow]) -> TeamPrevStats:
    tkey = norm_key(team)
    st = TeamPrevStats(name=team)
    if not matches:
        return st

    def gf_ga_corner(m: MatchRow) -> Tuple[int, int, Optional[int], Optional[int], Optional[int], Optional[int]]:
        if norm_key(m.home) == tkey:
            return (m.fthome, m.ftaway, m.cornerhome, m.corneraway, m.cornerhthome, m.cornerhtaway)
        return (m.ftaway, m.fthome, m.corneraway, m.cornerhome, m.cornerhtaway, m.cornerhthome)

    gfs, gas = [], []
    cf, ca = [], []
    cf_ht, ca_ht = [], []
    cleansheets = 0
    scoredmatches = 0

    for m in matches:
        gf, ga, cfor, cag, cfor_ht, cag_ht = gf_ga_corner(m)
        gfs.append(gf)
        gas.append(ga)

        if ga == 0:
            cleansheets += 1
        if gf > 0:
            scoredmatches += 1

        if cfor is not None and cag is not None:
            cf.append(cfor)
            ca.append(cag)
        if cfor_ht is not None and cag_ht is not None:
            cf_ht.append(cfor_ht)
            ca_ht.append(cag_ht)

    st.ntotal = len(matches)
    st.gftotal = weighted_mean([float(x) for x in gfs])
    st.gatotal = weighted_mean([float(x) for x in gas])

    st.cleansheets = cleansheets
    st.scoredmatches = scoredmatches

    st.corners_n = len(cf)
    st.cornersfor = weighted_mean([float(x) for x in cf]) if cf else 0.0
    st.cornersagainst = weighted_mean([float(x) for x in ca]) if ca else 0.0

    st.corners_ht_n = len(cf_ht)
    st.cornersfor_ht = weighted_mean([float(x) for x in cf_ht]) if cf_ht else 0.0
    st.cornersagainst_ht = weighted_mean([float(x) for x in ca_ht]) if ca_ht else 0.0

    home_ms = [m for m in matches if norm_key(m.home) == tkey]
    away_ms = [m for m in matches if norm_key(m.away) == tkey]

    st.nhome = len(home_ms)
    if st.nhome:
        st.gfhome = weighted_mean([float(m.fthome) for m in home_ms])
        st.gahome = weighted_mean([float(m.ftaway) for m in home_ms])

    st.naway = len(away_ms)
    if st.naway:
        st.gfaway = weighted_mean([float(m.ftaway) for m in away_ms])
        st.gaaway = weighted_mean([float(m.fthome) for m in away_ms])

    return st

# =============================================================================
# POISSON / MATRICES
# =============================================================================
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k > 170:
        return 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except Exception:
        return 0.0

def build_score_matrix(lh: float, la: float, maxg: int = 5) -> Dict[Tuple[int, int], float]:
    mat = {}
    for h in range(maxg + 1):
        ph = poisson_pmf(h, lh)
        for a in range(maxg + 1):
            mat[(h, a)] = ph * poisson_pmf(a, la)
    return mat

def market_probs_from_matrix(mat: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    p1 = sum(p for (h, a), p in mat.items() if h > a)
    px = sum(p for (h, a), p in mat.items() if h == a)
    p2 = sum(p for (h, a), p in mat.items() if h < a)
    btts = sum(p for (h, a), p in mat.items() if h >= 1 and a >= 1)

    out = {"1": p1, "X": px, "2": p2, "BTTS": btts}
    for ln in [0.5, 1.5, 2.5, 3.5]:
        need = int(math.floor(ln)) + 1
        out[f"O{ln}"] = sum(p for (h, a), p in mat.items() if (h + a) >= need)
        out[f"U{ln}"] = 1.0 - out[f"O{ln}"]
    return out

def top_scores_from_matrix(mat: Dict[Tuple[int, int], float], topn: int = 7) -> List[Tuple[str, float]]:
    items = sorted(mat.items(), key=lambda x: x[1], reverse=True)[:topn]
    return [(f"{h}-{a}", p) for (h, a), p in items]

def monte_carlo(lh: float, la: float, n: int, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    hg = rng.poisson(lh, size=n)
    ag = rng.poisson(la, size=n)
    total = hg + ag

    def p(mask) -> float:
        return float(np.mean(mask))

    cnt = Counter(zip(hg.tolist(), ag.tolist()))
    top10 = cnt.most_common(10)
    top10_list = [f"{h}-{a}: {c / n * 100:.1f}%" for (h, a), c in top10]

    dist_total = Counter(total.tolist())
    total_bins: Dict[str, float] = {}
    for k in range(0, 5):
        total_bins[str(k)] = dist_total.get(k, 0) / n * 100.0
    total_bins["5+"] = sum(v for kk, v in dist_total.items() if kk >= 5) / n * 100.0

    return {
        "p": {
            "1": p(hg > ag),
            "X": p(hg == ag),
            "2": p(hg < ag),
            "BTTS": p((hg >= 1) & (ag >= 1)),
            "O2.5": p(total >= 3),
            "U2.5": p(total <= 2),
            "O3.5": p(total >= 4),
            "U3.5": p(total <= 3),
        },
        "TOTAL_DIST": total_bins,
        "TOP10": top10_list,
    }

def model_agreement(ppo: Dict[str, float], pmc: Dict[str, float]) -> Tuple[float, str]:
    keys = ["1", "X", "2", "BTTS", "O2.5", "U2.5"]
    diffs = [abs(ppo.get(k, 0) - pmc.get(k, 0)) for k in keys]
    d = max(diffs) if diffs else 0.0
    if d < 0.03:
        return d, "Mükemmel"
    if d < 0.06:
        return d, "İyi"
    if d < 0.10:
        return d, "Orta"
    return d, "Zayıf"

def blend_probs(p1: Dict[str, float], p2: Dict[str, float], alpha: float) -> Dict[str, float]:
    out = {}
    for k in set(list(p1.keys()) + list(p2.keys())):
        if k in p1 and k in p2:
            out[k] = alpha * p1[k] + (1.0 - alpha) * p2[k]
        else:
            out[k] = p1.get(k, p2.get(k, 0.0))
    return out

def clamp_lambda(lh: float, la: float) -> Tuple[float, float, List[str]]:
    warn = []
    def c(x: float, name: str) -> float:
        if x < 0.15:
            warn.append(f"{name} too low ({x:.2f}) → 0.15")
            return 0.15
        if x > 3.80:
            warn.append(f"{name} too high ({x:.2f}) → 3.80")
            return 3.80
        return x
    return c(lh, "home"), c(la, "away"), warn

# =============================================================================
# VBA STYLE LAMBDAS (%100 PSS)
# =============================================================================
def compute_lambdas_pss_only(homeprev: TeamPrevStats, awayprev: TeamPrevStats) -> Tuple[float, float, Dict[str, Any]]:
    info: Dict[str, Any] = {"mode": "PSS_ONLY", "weights": {"recent_n": RECENT_WEIGHT_N, "w_recent": W_RECENT, "w_old": W_OLD}, "warnings": []}

    # VBA logic: home lambda from HOME matches goals-for, away lambda from AWAY matches goals-for
    lh = homeprev.gfhome if homeprev.nhome >= 3 else (homeprev.gftotal if homeprev.ntotal >= 3 else 1.30)
    la = awayprev.gfaway if awayprev.naway >= 3 else (awayprev.gftotal if awayprev.ntotal >= 3 else 1.10)

    lh, la, clamp_warn = clamp_lambda(lh, la)
    if clamp_warn:
        info["warnings"].extend(clamp_warn)

    info["source"] = {
        "home_lambda_from": f"gfhome(weighted) n={homeprev.nhome}" if homeprev.nhome >= 3 else f"fallback gftotal/default n={homeprev.ntotal}",
        "away_lambda_from": f"gfaway(weighted) n={awayprev.naway}" if awayprev.naway >= 3 else f"fallback gftotal/default n={awayprev.ntotal}",
    }
    return lh, la, info

# =============================================================================
# CORNERS (VBA STYLE)
# =============================================================================
def build_corner_matrix(lh: float, la: float, maxc: int = 20) -> Dict[Tuple[int, int], float]:
    mat = {}
    for h in range(maxc + 1):
        ph = poisson_pmf(h, lh)
        for a in range(maxc + 1):
            mat[(h, a)] = ph * poisson_pmf(a, la)
    return mat

def corner_market_probs(mat: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    total_dist = {}
    for (h, a), p in mat.items():
        total = h + a
        total_dist[total] = total_dist.get(total, 0) + p

    probs = {}
    for line in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]:
        k = int(math.floor(line))
        punder = sum(total_dist.get(i, 0) for i in range(0, k + 1))
        pover = 1.0 - punder
        probs[f"O{line}"] = float(max(0.0, min(1.0, pover)))
        probs[f"U{line}"] = float(max(0.0, min(1.0, punder)))
    return probs

def most_likely_corner_score(mat: Dict[Tuple[int, int], float], topn: int = 5) -> List[Tuple[str, float]]:
    items = sorted(mat.items(), key=lambda x: x[1], reverse=True)[:topn]
    return [(f"{h}-{a}", p) for (h, a), p in items]

def analyze_corners_pss_only(homeprev: TeamPrevStats, awayprev: TeamPrevStats, h2hmatches: List[MatchRow]) -> Dict[str, Any]:
    # H2H averages (display only)
    h2h_total = []
    h2h_total_ht = []
    for m in h2hmatches[:H2H_N]:
        if m.cornerhome is not None and m.corneraway is not None:
            h2h_total.append(m.cornerhome + m.corneraway)
        if m.cornerhthome is not None and m.cornerhtaway is not None:
            h2h_total_ht.append(m.cornerhthome + m.cornerhtaway)

    h2h_total_avg = sum(h2h_total) / len(h2h_total) if h2h_total else 0.0
    h2h_total_ht_avg = sum(h2h_total_ht) / len(h2h_total_ht) if h2h_total_ht else 0.0

    # VBA Perplexity formula (PSS ONLY):
    # home corners = (home_for + away_against)/2
    # away corners = (away_for + home_against)/2
    if homeprev.corners_n > 0 and awayprev.corners_n > 0:
        pred_home = (homeprev.cornersfor + awayprev.cornersagainst) / 2.0
        pred_away = (awayprev.cornersfor + homeprev.cornersagainst) / 2.0
    else:
        pred_home = homeprev.cornersfor or 4.5
        pred_away = awayprev.cornersfor or 4.0

    # HT corners (if available)
    if homeprev.corners_ht_n > 0 and awayprev.corners_ht_n > 0:
        pred_home_ht = (homeprev.cornersfor_ht + awayprev.cornersagainst_ht) / 2.0
        pred_away_ht = (awayprev.cornersfor_ht + homeprev.cornersagainst_ht) / 2.0
    else:
        pred_home_ht = pred_home * 0.45
        pred_away_ht = pred_away * 0.45

    pred_home = max(0.01, float(pred_home))
    pred_away = max(0.01, float(pred_away))
    pred_home_ht = max(0.01, float(pred_home_ht))
    pred_away_ht = max(0.01, float(pred_away_ht))

    total_corners = pred_home + pred_away
    total_ht = pred_home_ht + pred_away_ht

    # FT markets
    corner_mat = build_corner_matrix(pred_home, pred_away, maxc=MAX_CORNERS_FOR_MATRIX)
    market_probs_ft = corner_market_probs(corner_mat)
    top_corner_scores = most_likely_corner_score(corner_mat, topn=5)

    # HT markets
    corner_mat_ht = build_corner_matrix(pred_home_ht, pred_away_ht, maxc=15)
    total_dist_ht = {}
    for (h, a), p in corner_mat_ht.items():
        total = h + a
        total_dist_ht[total] = total_dist_ht.get(total, 0) + p

    market_probs_ht = {}
    for line in [3.5, 4.5, 5.5, 6.5]:
        k = int(math.floor(line))
        punder = sum(total_dist_ht.get(i, 0) for i in range(0, k + 1))
        pover = 1.0 - punder
        market_probs_ht[f"HT_O{line}"] = float(max(0.0, min(1.0, pover)))
        market_probs_ht[f"HT_U{line}"] = float(max(0.0, min(1.0, punder)))

    top_corner_scores_ht = most_likely_corner_score(corner_mat_ht, topn=3)

    # Confidence (simple)
    dp = min(homeprev.corners_n, RECENT_N) + min(awayprev.corners_n, RECENT_N)
    dp_ht = min(homeprev.corners_ht_n, RECENT_N) + min(awayprev.corners_ht_n, RECENT_N)

    conf = "Yüksek" if dp >= 12 else ("Orta" if dp >= 6 else "Düşük")
    conf_ht = "Yüksek" if dp_ht >= 12 else ("Orta" if dp_ht >= 6 else "Düşük")

    return {
        "mode": "PSS_ONLY",
        "predicted_home_corners": round(pred_home, 1),
        "predicted_away_corners": round(pred_away, 1),
        "total_corners": round(total_corners, 1),
        "market_probs": market_probs_ft,
        "top_corner_scores": top_corner_scores,
        "confidence": conf,
        "first_half": {
            "predicted_home_ht": round(pred_home_ht, 1),
            "predicted_away_ht": round(pred_away_ht, 1),
            "total_ht": round(total_ht, 1),
            "predictions": market_probs_ht,
            "top_scores_ht": top_corner_scores_ht,
            "confidence_ht": conf_ht,
            "h2h_ht_avg_display": round(h2h_total_ht_avg, 1),
        },
        "h2h_avg_display": round(h2h_total_avg, 1),
        "h2h_data_count": len(h2h_total),
        "pss_data_available": bool(homeprev.corners_n > 0 and awayprev.corners_n > 0),
    }

# =============================================================================
# REPORT (VBA-LIKE)
# =============================================================================
def score_2_from_lambda(lh: float, la: float) -> str:
    return f"{round(lh)}-{round(la)}"

def format_report_vba_style(data: Dict[str, Any]) -> str:
    t = data["teams"]
    ds = data["datasources"]
    lam = data["lambda"]
    top7 = data["poisson"]["top7_scores"]
    blend = data["blended_probs"]
    corners = data.get("corner_analysis", {})
    ma = data.get("model_agreement", {})

    lines = []
    lines.append("=" * 55)
    lines.append("  FUTBOL MAC ANALIZI - %100 PSS MODEL")
    lines.append("  (STANDINGS & H2H SADECE GORUNUM - HESAPTA YOK)")
    lines.append("=" * 55)
    lines.append(f"MAC: {t['home']} vs {t['away']}")
    lines.append(f"LIG: {data.get('league','')}")
    lines.append("-" * 55)

    lines.append("0) VERI KAYNAK KONTROLU")
    lines.append(f"  Standings: {'VAR (gosterim)' if ds.get('standings_available') else 'YOK'}")
    lines.append(f"  H2H: {ds.get('h2h_matches',0)} mac (gosterim)")
    lines.append("  PSS (Same League):")
    lines.append(f"    HOME rol: {ds.get('home_prev_matches',0)} mac (ilk 5=1.20, diger=0.80)")
    lines.append(f"    AWAY rol: {ds.get('away_prev_matches',0)} mac (ilk 5=1.20, diger=0.80)")
    lines.append("-" * 55)

    lines.append("A) BEKLENEN GOLLER (xG) - %100 PSS")
    lines.append(f"  lambda_home = {lam['home']:.2f}")
    lines.append(f"  lambda_away = {lam['away']:.2f}")
    lines.append(f"  TOPLAM xG   = {lam['total']:.2f}")
    if lam.get("info", {}).get("warnings"):
        lines.append("  UYARI:")
        for w in lam["info"]["warnings"]:
            lines.append(f"    - {w}")
    lines.append("-" * 55)

    lines.append("B) EN OLASI SKORLAR (Poisson Matrix)")
    for i, (s, p) in enumerate(top7[:7], 1):
        lines.append(f"  {i}) {s}  %{p*100:.1f}")
    lines.append(f"  Skor-2: {data.get('score_2','N/A')}")
    lines.append("-" * 55)

    lines.append("C) MARKET OLASILIKLARI (Blend: Poisson + MC)")
    lines.append(f"  1: %{blend.get('1',0)*100:.1f} | X: %{blend.get('X',0)*100:.1f} | 2: %{blend.get('2',0)*100:.1f}")
    lines.append(f"  KG VAR: %{blend.get('BTTS',0)*100:.1f} | KG YOK: %{(1.0-blend.get('BTTS',0))*100:.1f}")
    for ln in ["0.5","1.5","2.5","3.5"]:
        lines.append(f"  U{ln}: %{blend.get('U'+ln,0)*100:.1f} | O{ln}: %{blend.get('O'+ln,0)*100:.1f}")
    if ma:
        lines.append(f"  Model Uyumu: {ma.get('label','N/A')} (max fark %{ma.get('diff',0)*100:.2f})")
    lines.append("-" * 55)

    if corners and corners.get("total_corners", 0) > 0:
        lines.append("D) KORNER (PSS ONLY + Poisson)")
        lines.append(f"  FT Toplam: {corners['total_corners']}  (Ev {corners['predicted_home_corners']} | Dep {corners['predicted_away_corners']})  Guven: {corners.get('confidence','')}")
        mp = corners.get("market_probs", {})
        for k in ["O9.5","O10.5","O11.5"]:
            if k in mp:
                lines.append(f"  {k}: %{mp[k]*100:.1f} | {k.replace('O','U')}: %{mp.get(k.replace('O','U'),0)*100:.1f}")
        tc = corners.get("top_corner_scores", [])
        if tc:
            lines.append("  En Olası Korner Skorları:")
            for i, (s, p) in enumerate(tc[:3], 1):
                lines.append(f"    {i}) {s}  %{p*100:.1f}")

        ht = corners.get("first_half", {})
        if ht.get("total_ht", 0) > 0:
            lines.append("  ILK YARI (HT) KORNER:")
            lines.append(f"    Toplam: {ht['total_ht']} (Ev {ht['predicted_home_ht']} | Dep {ht['predicted_away_ht']}) Guven: {ht.get('confidence_ht','')}")
            htp = ht.get("predictions", {})
            for k in ["HT_O4.5","HT_O5.5"]:
                if k in htp:
                    lines.append(f"    {k}: %{htp[k]*100:.1f} | {k.replace('O','U')}: %{htp.get(k.replace('O','U'),0)*100:.1f}")
        lines.append("-" * 55)

    lines.append("ANALIZ TAMAMLANDI")
    lines.append("=" * 55)
    return "\n".join(lines)

# =============================================================================
# MAIN
# =============================================================================
def analyze_nowgoal(url: str, mcruns: int = MC_RUNS_DEFAULT) -> Dict[str, Any]:
    log_info(f"Starting analysis: {url}")

    h2h_url = build_h2h_url(url)
    html = safe_get(h2h_url, referer=extract_base_domain(url))

    home_team, away_team = parse_teams_from_title(html)
    if not home_team or not away_team:
        raise RuntimeError("Team parse failed (og:title/title/h1)")

    log_info(f"Teams: {home_team} vs {away_team}")

    league_match = re.search(r'<span[^>]*class="?sclassLink"?[^>]*>([^<]+)</span>', html)
    league_name = strip_tags_keep_text(league_match.group(1)) if league_match else ""

    # Standings (display only)
    st_home_rows = extract_standings_for_team(html, home_team)
    st_away_rows = extract_standings_for_team(html, away_team)
    _ = standings_to_splits(st_home_rows)
    _ = standings_to_splits(st_away_rows)

    # Previous tables
    prev_home_tab, prev_away_tab = extract_previous_from_page(html)
    prev_home_raw = parse_matches_from_table_html(prev_home_tab) if prev_home_tab else []
    prev_away_raw = parse_matches_from_table_html(prev_away_tab) if prev_away_tab else []

    # H2H (display only)
    h2h_all = extract_h2h_matches(html, home_team, away_team)
    h2h_pair = [m for m in h2h_all if is_h2h_pair(m, home_team, away_team)]
    h2h_pair = sort_matches_desc(dedupe_matches(h2h_pair))
    h2h_used = h2h_pair[:H2H_N]

    # Same league filter for PSS (like you do in Excel)
    if league_name:
        prev_home_raw = filter_same_league_matches(prev_home_raw, league_name)
        prev_away_raw = filter_same_league_matches(prev_away_raw, league_name)

    # VBA style role filters
    prev_home_sel = filter_team_home_only(prev_home_raw, home_team)[:RECENT_N]
    prev_away_sel = filter_team_away_only(prev_away_raw, away_team)[:RECENT_N]

    home_prev_stats = build_prev_stats(home_team, prev_home_sel)
    away_prev_stats = build_prev_stats(away_team, prev_away_sel)

    # Lambdas (PSS only)
    lam_home, lam_away, lambda_info = compute_lambdas_pss_only(home_prev_stats, away_prev_stats)
    score_2 = score_2_from_lambda(lam_home, lam_away)

    # Poisson
    score_mat = build_score_matrix(lam_home, lam_away, maxg=MAX_GOALS_FOR_MATRIX)
    poisson_market = market_probs_from_matrix(score_mat)
    top7 = top_scores_from_matrix(score_mat, topn=7)

    # Monte Carlo
    mc = monte_carlo(lam_home, lam_away, n=mcruns, seed=42)
    diff, diff_label = model_agreement(poisson_market, mc["p"])
    blended = blend_probs(poisson_market, mc["p"], alpha=BLEND_ALPHA)

    # Corners (PSS only, HT supported)
    corner_analysis = analyze_corners_pss_only(home_prev_stats, away_prev_stats, h2h_used)

    value_block = {"used_odds": False, "decision": "Oran gerekli", "table": []}

    data = {
        "url": h2h_url,
        "teams": {"home": home_team, "away": away_team},
        "league": league_name,
        "lambda": {
            "home": lam_home,
            "away": lam_away,
            "total": lam_home + lam_away,
            "info": lambda_info,
        },
        "score_2": score_2,
        "poisson": {"market_probs": poisson_market, "top7_scores": top7},
        "mc": mc,
        "model_agreement": {"diff": diff, "label": diff_label},
        "blended_probs": blended,
        "corner_analysis": corner_analysis,
        "value_bets": value_block,
        "datasources": {
            "standings_available": bool(st_home_rows and st_away_rows),
            "h2h_matches": len(h2h_used),
            "home_prev_matches": len(prev_home_sel),
            "away_prev_matches": len(prev_away_sel),
            "pss_sameleague_used": bool(league_name),
        },
    }

    data["report_comprehensive"] = format_report_vba_style(data)
    log_info("Analysis complete")
    return data

# =============================================================================
# FLASK
# =============================================================================
app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({"ok": True, "service": "nowgoal-analyzer-api", "version": "5.2-pss-only", "status": "running"})

@app.route("/health")
def health():
    return jsonify({"ok": True, "status": "healthy", "timestamp": time.time()})

@app.route("/test")
def test():
    try:
        import numpy as _np
        import requests as _req
        return jsonify({"ok": True, "numpy_version": _np.__version__, "requests_version": _req.__version__, "test": "Libraries OK"})
    except Exception as e:
        log_error("Test failed", e)
        return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/analizet", methods=["POST"])
@app.route("/analiz_et", methods=["POST"])
def analizet_route():
    request_start = time.time()
    log_info("=== REQUEST STARTED: /analiz_et ===")

    try:
        payload = request.get_json(silent=True) or {}
        url = (payload.get("url") or "").strip()
        if not url:
            return jsonify({"ok": False, "error": "URL boş"}), 400
        if not re.match(r"https?://", url):
            return jsonify({"ok": False, "error": "Geçersiz URL"}), 400

        try:
            data = analyze_nowgoal(url, mcruns=MC_RUNS_DEFAULT)
        except requests.exceptions.Timeout as e:
            log_error("Timeout", e)
            return jsonify({"ok": False, "error": "Zaman aşımı", "detail": str(e)}), 504
        except requests.exceptions.RequestException as e:
            log_error("Request failed", e)
            return jsonify({"ok": False, "error": "Bağlantı hatası", "detail": str(e)}), 502
        except Exception as e:
            log_error("Analysis failed", e)
            return jsonify({"ok": False, "error": f"Analiz hatası: {str(e)}", "traceback": traceback.format_exc()}), 500

        top_skor = data["poisson"]["top7_scores"][0]
        blend = data["blended_probs"]
        corners = data["corner_analysis"]
        top_corner = corners["top_corner_scores"][0] if corners.get("top_corner_scores") else ("N/A", 0.0)

        response = {
            "ok": True,
            "model": "PSS_ONLY (VBA style)",
            "skor": f"{top_skor[0]}: {top_skor[1]*100:.1f}%",
            "skor_2": data["score_2"],
            "alt_ust": f"2.5 {'ÜST' if blend.get('O2.5', 0) > 0.5 else 'ALT'}: {max(blend.get('O2.5', 0), blend.get('U2.5', 0))*100:.1f}%",
            "btts": f"{'VAR' if blend.get('BTTS', 0) > 0.5 else 'YOK'}: {blend.get('BTTS', 0)*100:.1f}%",
            "korner": {
                "toplam": corners.get("total_corners", 0),
                "ev": corners.get("predicted_home_corners", 0),
                "deplasman": corners.get("predicted_away_corners", 0),
                "en_olasi": f"{top_corner[0]}: {top_corner[1]*100:.1f}%",
                "ilk_yari": corners.get("first_half", {}).get("total_ht", 0),
                "ilk_yari_ev": corners.get("first_half", {}).get("predicted_home_ht", 0),
                "ilk_yari_dep": corners.get("first_half", {}).get("predicted_away_ht", 0),
                "over_9_5": f"{corners.get('market_probs', {}).get('O9.5', 0)*100:.1f}%",
                "over_10_5": f"{corners.get('market_probs', {}).get('O10.5', 0)*100:.1f}%",
            },
            "karar": data["value_bets"].get("decision", ""),
            "guven": corners.get("confidence", ""),
            "detay": data["report_comprehensive"],
            "sure": f"{time.time() - request_start:.2f}s",
        }

        log_info(f"=== REQUEST COMPLETED in {time.time() - request_start:.2f}s ===")
        return jsonify(response)

    except Exception as e:
        log_error("Unhandled exception", e)
        return jsonify({"ok": False, "error": f"Beklenmeyen hata: {str(e)}", "traceback": traceback.format_exc()}), 500

@app.route("/analyze", methods=["POST"])
def analyze_route():
    try:
        payload = request.get_json(silent=True) or {}
        url = (payload.get("url") or "").strip()
        if not url:
            return jsonify({"ok": False, "error": "url required"}), 400
        data = analyze_nowgoal(url, mcruns=MC_RUNS_DEFAULT)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        log_error("Analyze route failed", e)
        return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "error": "Endpoint bulunamadı", "endpoints": ["/", "/health", "/test", "/analiz_et", "/analyze"]}), 404

@app.errorhandler(500)
def internal_error(e):
    log_error("500 error", e)
    return jsonify({"ok": False, "error": "Sunucu hatası"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    log_error("Unhandled exception", e)
    return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    log_info("=" * 70)
    log_info("NowGoal Analyzer v5.2 - PSS_ONLY (VBA Style)")
    log_info("=" * 70)
    log_info(f"Python: {sys.version}")
    try:
        import numpy
        log_info(f"NumPy: {numpy.__version__}")
    except ImportError:
        log_error("NumPy NOT INSTALLED!")
    try:
        import requests as req
        log_info(f"Requests: {req.__version__}")
    except ImportError:
        log_error("Requests NOT INSTALLED!")

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        log_info("Starting Flask server on 0.0.0.0:5000 ...")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    else:
        print("Usage: python app.py serve")
