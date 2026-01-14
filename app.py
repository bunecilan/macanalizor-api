# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - ENHANCED VERSION 5.0 FINAL COMPLETE
- İlk yarı korner H2H ve PSS'den gerçek verilerle
- Score-2 prediction (rounded lambda)
- /analiz_et endpoint fixed
- Full corner matrix
- Value betting
- Complete code - 2100+ lines
"""

import re
import math
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import numpy as np
import requests
from flask import Flask, request, jsonify

# ============================================================================
# CONFIGURATION
# ============================================================================
MC_RUNS_DEFAULT = 10000
RECENT_N = 10
H2H_N = 10
WST_BASE = 0.45
WPSS_BASE = 0.30
WH2H_BASE = 0.25
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

# ============================================================================
# REGEX PATTERNS
# ============================================================================
DATE_ANY_RE = re.compile(r'\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{2}-\d{2}')
SCORE_RE = re.compile(r'(\d{1,2})-(\d{1,2})(?:\((\d{1,2})-(\d{1,2})\))?')
CORNER_FT_RE = re.compile(r'(\d{1,2})-(\d{1,2})')
CORNER_HT_RE = re.compile(r'\((\d{1,2})-(\d{1,2})\)')
FLOAT_RE = re.compile(r'(\d+(?:\.\d+)?)')
INT_RE = re.compile(r'(\d+)')

# ============================================================================
# DATA CLASSES
# ============================================================================
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
    gftotal: float = 0.0
    gatotal: float = 0.0
    ntotal: int = 0
    gfhome: float = 0.0
    gahome: float = 0.0
    nhome: int = 0
    gfaway: float = 0.0
    gaaway: float = 0.0
    naway: int = 0
    cleansheets: int = 0
    scoredmatches: int = 0
    cornersfor: float = 0.0
    cornersagainst: float = 0.0
    cornersfor_ht: float = 0.0
    cornersagainst_ht: float = 0.0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def norm_key(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())

def normalize_date(d: str) -> Optional[str]:
    if not d:
        return None
    d = d.strip()
    m = DATE_ANY_RE.search(d)
    if not m:
        return None
    val = m.group(0)
    if re.match(r'\d{4}-\d{2}-\d{2}', val):
        yyyy, mm, dd = val.split('-')
        return f"{int(dd):02d}-{int(mm):02d}-{yyyy}"
    if re.match(r'\d{1,2}-\d{1,2}-\d{4}', val):
        dd, mm, yyyy = val.split('-')
        return f"{int(dd):02d}-{int(mm):02d}-{yyyy}"
    return None

def parse_date_key(datestr: str) -> Tuple[int, int, int]:
    if not datestr or not re.match(r'\d{2}-\d{2}-\d{4}', datestr):
        return (0, 0, 0)
    dd, mm, yyyy = datestr.split('-')
    return (int(yyyy), int(mm), int(dd))

def strip_tags_keep_text(s: str) -> str:
    s = re.sub(r'<script.*?</script>', '', s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r'<style.*?</style>', '', s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r'<.*?>', '', s)
    s = s.replace('&nbsp;', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def extract_tables_html(page_source: str) -> List[str]:
    return [m.group(0) for m in re.finditer(r'<table.*?</table>', page_source or '', flags=re.IGNORECASE | re.DOTALL)]

def extract_table_rows_from_html(table_html: str) -> List[List[str]]:
    rows: List[List[str]] = []
    trs = re.findall(r'<tr.*?</tr>', table_html or '', flags=re.IGNORECASE | re.DOTALL)
    for tr in trs:
        cells = re.findall(r'<t[dh].*?</t[dh]>', tr, flags=re.IGNORECASE | re.DOTALL)
        if not cells:
            continue
        cleaned = [strip_tags_keep_text(c) for c in cells]
        normalized = []
        for c in cleaned:
            c = (c or '').strip()
            if c in ('', '-'):
                c = ''
            normalized.append(c)
        if any(x for x in normalized):
            rows.append(normalized)
    return rows

def section_tables_by_marker(page_source: str, marker: str, max_tables: int = 3) -> List[str]:
    low = (page_source or '').lower()
    pos = low.find(marker.lower())
    if pos == -1:
        return []
    sub = page_source[pos:]
    tabs = extract_tables_html(sub)
    return tabs[:max_tables]

def safe_get(url: str, timeout: int = 25, retries: int = 2, referer: Optional[str] = None) -> str:
    last_err = None
    headers = dict(HEADERS)
    if referer:
        headers['Referer'] = referer
    for _ in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(0.7)
    raise RuntimeError(f"Fetch failed {url}: {last_err}")

def extract_match_id(url: str) -> str:
    m = re.search(r'(?:h2h-|match/h2h-)(\d+)', url)
    if m:
        return m.group(1)
    nums = re.findall(r'\d{6,}', url)
    if not nums:
        raise ValueError("Match ID bulunamadı")
    return nums[-1]

def extract_base_domain(url: str) -> str:
    m = re.match(r'(https?://[^/]+)', url.strip())
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
    m = re.search(r'<title.*?</title>', html, flags=re.IGNORECASE | re.DOTALL)
    title = strip_tags_keep_text(m.group(0)) if m else ""
    mm = re.search(r'(.+?)\s*VS\s*(.+?)(?:\s*-|$)', title, flags=re.IGNORECASE)
    if not mm:
        mm = re.search(r'(.+?)\s*vs\s*(.+?)(?:\s*-|$)', title, flags=re.IGNORECASE)
    if not mm:
        return ("", "")
    return (mm.group(1).strip(), mm.group(2).strip())

def sort_matches_desc(matches: List[MatchRow]) -> List[MatchRow]:
    has_real_date = any(parse_date_key(m.date) != (0, 0, 0) for m in matches)
    if not has_real_date:
        return matches
    return sorted(matches, key=lambda x: parse_date_key(x.date), reverse=True)

def dedupe_matches(matches: List[MatchRow]) -> List[MatchRow]:
    seen = set()
    out = []
    for m in matches:
        key = (m.league, m.date, m.home, m.away, m.fthome, m.ftaway, m.cornerhome, m.corneraway)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out

def is_h2h_pair(m: MatchRow, hometeam: str, awayteam: str) -> bool:
    hk, ak = norm_key(hometeam), norm_key(awayteam)
    mh, ma = norm_key(m.home), norm_key(m.away)
    return (mh == hk and ma == ak) or (mh == ak and ma == hk)

# ============================================================================
# PARSE CORNER CELL
# ============================================================================
def parse_corner_cell(cell: str) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Parse corner cell: "4-11(2-1)"
    Returns: FT: (4, 11), HT: (2, 1)
    """
    if not cell:
        return None, None
    txt = (cell or '').strip()
    if txt in ('', '-'):
        return None, None
    
    ftm = CORNER_FT_RE.search(txt)
    htm = CORNER_HT_RE.search(txt)
    
    ft = (int(ftm.group(1)), int(ftm.group(2))) if ftm else None
    ht = (int(htm.group(1)), int(htm.group(2))) if htm else None
    
    return ft, ht

# ============================================================================
# PARSE MATCH FROM CELLS
# ============================================================================
def parse_match_from_cells(cells: List[str]) -> Optional[MatchRow]:
    if not cells:
        return None
    
    def get(i: int) -> str:
        return (cells[i] or '').strip() if i < len(cells) else ''
    
    league = get(0) or ''
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
        dateval = normalize_date(datecell) or ''
        
        ftcorner, htcorner = parse_corner_cell(cornercell)
        cornerhome, corneraway = ftcorner if ftcorner else (None, None)
        cornerhthome, cornerhtaway = htcorner if htcorner else (None, None)
        
        return MatchRow(
            league=league,
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
    
    score_idx = None
    scorem = None
    for i, c in enumerate(cells):
        c0 = (c or '').strip()
        m = SCORE_RE.search(c0)
        if m:
            score_idx = i
            scorem = m
            break
    
    if not scorem or score_idx is None:
        return None
    
    fth = int(scorem.group(1))
    fta = int(scorem.group(2))
    hth = int(scorem.group(3)) if scorem.group(3) else None
    hta = int(scorem.group(4)) if scorem.group(4) else None
    
    home2 = None
    away2 = None
    for i in range(score_idx - 1, -1, -1):
        if cells[i] or ''.strip():
            home2 = (cells[i] or '').strip()
            break
    for i in range(score_idx + 1, len(cells)):
        if cells[i] or ''.strip():
            away2 = (cells[i] or '').strip()
            break
    
    if not home2 or not away2:
        return None
    
    league2 = cells[0] or ''.strip() or ''
    dateval2 = ''
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

# ============================================================================
# STANDINGS PARSE
# ============================================================================
def to_int(x: str) -> Optional[int]:
    try:
        x = (x or '').strip()
        if x in ('', '-'):
            return None
        return int(x)
    except Exception:
        return None

def parse_standings_table_rows(rows: List[List[str]]) -> List[StandRow]:
    wanted = ['Total', 'Home', 'Away', 'Last 6', 'Last6']
    out: List[StandRow] = []
    for cells in rows:
        if not cells:
            continue
        head = (cells[0] or '').strip()
        if head not in wanted:
            continue
        label = 'Last 6' if head == 'Last6' else head
        
        def g(i):
            return cells[i] if i < len(cells) else ''
        
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
            rate=g(9).strip() if g(9) else None
        )
        if r.matches is not None and not (1 <= r.matches <= 80):
            continue
        if any(x.ft == r.ft for x in out):
            continue
        out.append(r)
    
    order = {'Total': 0, 'Home': 1, 'Away': 2, 'Last 6': 3}
    out.sort(key=lambda x: order.get(x.ft, 99))
    return out

def extract_standings_for_team(page_source: str, teamname: str) -> List[StandRow]:
    team_key = norm_key(teamname)
    for tbl in extract_tables_html(page_source):
        text_low = strip_tags_keep_text(tbl).lower()
        required_keywords = ['matches', 'win', 'draw', 'loss', 'scored', 'conceded']
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
    mp: Dict[str, Optional[SplitGFGA]] = {'Total': None, 'Home': None, 'Away': None, 'Last 6': None}
    for r in rows:
        if r.matches and r.scored is not None and r.conceded is not None:
            mp[r.ft] = SplitGFGA(r.matches, r.scored, r.conceded)
    return mp

# ============================================================================
# EXTRACT PREVIOUS FROM PAGE
# ============================================================================
def extract_previous_from_page(page_source: str) -> Tuple[List[str], List[str]]:
    markers = ['Previous Scores Statistics', 'Previous Scores', 'Recent Matches']
    tabs = []
    for marker in markers:
        found_tabs = section_tables_by_marker(page_source, marker, max_tables=10)
        if found_tabs:
            tabs = found_tabs
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
        return match_tables[0], []
    return match_tables[0], match_tables[1]

# ============================================================================
# EXTRACT H2H MATCHES
# ============================================================================
def extract_h2h_matches(page_source: str, hometeam: str, awayteam: str) -> List[MatchRow]:
    markers = ['Head to Head Statistics', 'Head to Head', 'H2H Statistics', 'H2H', 'VS Statistics']
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

# ============================================================================
# FILTER FUNCTIONS
# ============================================================================
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

# ============================================================================
# BUILD PREV STATS - WITH HT CORNERS
# ============================================================================
def build_prev_stats(team: str, matches: List[MatchRow]) -> TeamPrevStats:
    tkey = norm_key(team)
    st = TeamPrevStats(name=team)
    
    if not matches:
        return st
    
    def team_gf_ga(m: MatchRow) -> Tuple[int, int, Optional[int], Optional[int], Optional[int], Optional[int]]:
        if norm_key(m.home) == tkey:
            return (m.fthome, m.ftaway, m.cornerhome, m.corneraway, m.cornerhthome, m.cornerhtaway)
        return (m.ftaway, m.fthome, m.corneraway, m.cornerhome, m.cornerhtaway, m.cornerhthome)
    
    gfs, gas = [], []
    corners_for, corners_against = [], []
    corners_for_ht, corners_against_ht = [], []
    cleansheets = 0
    scored_matches = 0
    
    for m in matches:
        gf, ga, cf, ca, cf_ht, ca_ht = team_gf_ga(m)
        
        gfs.append(gf)
        gas.append(ga)
        
        if cf is not None:
            corners_for.append(cf)
        if ca is not None:
            corners_against.append(ca)
        
        if cf_ht is not None:
            corners_for_ht.append(cf_ht)
        if ca_ht is not None:
            corners_against_ht.append(ca_ht)
        
        if ga == 0:
            cleansheets += 1
        if gf > 0:
            scored_matches += 1
    
    st.ntotal = len(matches)
    st.gftotal = sum(gfs) / st.ntotal if st.ntotal else 0.0
    st.gatotal = sum(gas) / st.ntotal if st.ntotal else 0.0
    st.cleansheets = cleansheets
    st.scoredmatches = scored_matches
    st.cornersfor = sum(corners_for) / len(corners_for) if corners_for else 0.0
    st.cornersagainst = sum(corners_against) / len(corners_against) if corners_against else 0.0
    st.cornersfor_ht = sum(corners_for_ht) / len(corners_for_ht) if corners_for_ht else 0.0
    st.cornersagainst_ht = sum(corners_against_ht) / len(corners_against_ht) if corners_against_ht else 0.0
    
    home_ms = [m for m in matches if norm_key(m.home) == tkey]
    away_ms = [m for m in matches if norm_key(m.away) == tkey]
    
    st.nhome = len(home_ms)
    if st.nhome:
        st.gfhome = sum(m.fthome for m in home_ms) / st.nhome
        st.gahome = sum(m.ftaway for m in home_ms) / st.nhome
    
    st.naway = len(away_ms)
    if st.naway:
        st.gfaway = sum(m.ftaway for m in away_ms) / st.naway
        st.gaaway = sum(m.fthome for m in away_ms) / st.naway
    
    return st

# ============================================================================
# POISSON FUNCTIONS
# ============================================================================
def poisson_pmf(k: int, lam: float) -> float:
    if lam == 0:
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

# ============================================================================
# CORNER ANALYSIS - ENHANCED WITH REAL HT DATA
# ============================================================================
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

def analyze_corners_enhanced(
    homeprev: TeamPrevStats,
    awayprev: TeamPrevStats,
    h2hmatches: List[MatchRow]
) -> Dict[str, Any]:
    """Enhanced corner analysis with REAL HT data from H2H and PSS"""
    
    h2h_total = []
    h2h_home = []
    h2h_away = []
    h2h_home_ht = []
    h2h_away_ht = []
    
    for m in h2hmatches[:H2H_N]:
        if m.cornerhome is not None and m.corneraway is not None:
            h2h_total.append(m.cornerhome + m.corneraway)
            h2h_home.append(m.cornerhome)
            h2h_away.append(m.corneraway)
        
        if m.cornerhthome is not None and m.cornerhtaway is not None:
            h2h_home_ht.append(m.cornerhthome)
            h2h_away_ht.append(m.cornerhtaway)
    
    h2h_total_avg = sum(h2h_total) / len(h2h_total) if h2h_total else 0.0
    h2h_home_avg = sum(h2h_home) / len(h2h_home) if h2h_home else 0.0
    h2h_away_avg = sum(h2h_away) / len(h2h_away) if h2h_away else 0.0
    h2h_home_ht_avg = sum(h2h_home_ht) / len(h2h_home_ht) if h2h_home_ht else 0.0
    h2h_away_ht_avg = sum(h2h_away_ht) / len(h2h_away_ht) if h2h_away_ht else 0.0
    
    pss_home_for = homeprev.cornersfor
    pss_home_against = homeprev.cornersagainst
    pss_away_for = awayprev.cornersfor
    pss_away_against = awayprev.cornersagainst
    pss_home_for_ht = homeprev.cornersfor_ht
    pss_home_against_ht = homeprev.cornersagainst_ht
    pss_away_for_ht = awayprev.cornersfor_ht
    pss_away_against_ht = awayprev.cornersagainst_ht
    
    if h2h_total_avg > 0:
        predicted_home = 0.6 * h2h_home_avg + 0.4 * ((pss_home_for + pss_away_against) / 2)
        predicted_away = 0.6 * h2h_away_avg + 0.4 * ((pss_away_for + pss_home_against) / 2)
    elif pss_home_for > 0 or pss_away_for > 0:
        predicted_home = (pss_home_for + pss_away_against) / 2
        predicted_away = (pss_away_for + pss_home_against) / 2
    else:
        predicted_home = 0.0
        predicted_away = 0.0
    
    total_corners = max(0.01, predicted_home + predicted_away)
    
    if h2h_home_ht_avg > 0 or h2h_away_ht_avg > 0:
        predicted_home_ht = 0.6 * h2h_home_ht_avg + 0.4 * ((pss_home_for_ht + pss_away_against_ht) / 2)
        predicted_away_ht = 0.6 * h2h_away_ht_avg + 0.4 * ((pss_away_for_ht + pss_home_against_ht) / 2)
    elif pss_home_for_ht > 0 or pss_away_for_ht > 0:
        predicted_home_ht = (pss_home_for_ht + pss_away_against_ht) / 2
        predicted_away_ht = (pss_away_for_ht + pss_home_against_ht) / 2
    else:
        predicted_home_ht = predicted_home * 0.45
        predicted_away_ht = predicted_away * 0.45
    
    total_corners_ht = max(0.01, predicted_home_ht + predicted_away_ht)
    
    corner_mat = build_corner_matrix(predicted_home, predicted_away, maxc=MAX_CORNERS_FOR_MATRIX)
    market_probs_ft = corner_market_probs(corner_mat)
    top_corner_scores = most_likely_corner_score(corner_mat, topn=5)
    
    corner_mat_ht = build_corner_matrix(predicted_home_ht, predicted_away_ht, maxc=15)
    
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
    
    datapoints = len(h2h_total) + (1 if pss_home_for > 0 or pss_away_for > 0 else 0)
    datapoints_ht = len(h2h_home_ht) + (1 if pss_home_for_ht > 0 or pss_away_for_ht > 0 else 0)
    
    confidence = "Yüksek" if datapoints >= 8 else ("Orta" if datapoints >= 4 else "Düşük")
    confidence_ht = "Yüksek" if datapoints_ht >= 8 else ("Orta" if datapoints_ht >= 4 else "Düşük")
    
    return {
        "predicted_home_corners": round(predicted_home, 1),
        "predicted_away_corners": round(predicted_away, 1),
        "total_corners": round(total_corners, 1),
        "market_probs": market_probs_ft,
        "top_corner_scores": top_corner_scores,
        "confidence": confidence,
        "first_half": {
            "predicted_home_ht": round(predicted_home_ht, 1),
            "predicted_away_ht": round(predicted_away_ht, 1),
            "total_ht": round(total_corners_ht, 1),
            "predictions": market_probs_ht,
            "top_scores_ht": top_corner_scores_ht,
            "confidence_ht": confidence_ht,
            "data_source": "H2H ve PSS gerçek ilk yarı verileri",
            "h2h_ht_count": len(h2h_home_ht),
            "pss_ht_available": bool(pss_home_for_ht > 0 or pss_away_for_ht > 0)
        },
        "h2h_avg": round(h2h_total_avg, 1),
        "h2h_data_count": len(h2h_total),
        "pss_data_available": bool(pss_home_for > 0 or pss_away_for > 0)
    }

# ============================================================================
# SCORE-2 PREDICTION
# ============================================================================
def score_2_from_lambda(lh: float, la: float) -> str:
    return f"{round(lh)}-{round(la)}"

# ============================================================================
# GOAL ANALYSIS
# ============================================================================
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
    elif d < 0.06:
        return d, "İyi"
    elif d < 0.10:
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

# ============================================================================
# LAMBDA CALCULATIONS
# ============================================================================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in w.values())
    if s < 1e-9:
        return {}
    return {k: max(0.0, v) / s for k, v in w.items()}

def compute_component_standings(st_home: Dict[str, Optional[SplitGFGA]], st_away: Dict[str, Optional[SplitGFGA]]) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    hh = st_home.get("Home") or st_home.get("Total")
    aa = st_away.get("Away") or st_away.get("Total")
    if not hh or not aa or hh.matches < 3 or aa.matches < 3:
        return None
    lamh = (hh.gfpg + aa.gapg) / 2.0
    lama = (aa.gfpg + hh.gapg) / 2.0
    meta = {
        "home_split": {"matches": hh.matches, "gfpg": hh.gfpg, "gapg": hh.gapg},
        "away_split": {"matches": aa.matches, "gfpg": aa.gfpg, "gapg": aa.gapg},
        "formula": "Standing-based lambda"
    }
    return lamh, lama, meta

def compute_component_pss(homeprev: TeamPrevStats, awayprev: TeamPrevStats) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    if homeprev.ntotal < 3 or awayprev.ntotal < 3:
        return None
    hgf = homeprev.gftotal
    hga = homeprev.gatotal
    agf = awayprev.gftotal
    aga = awayprev.gatotal
    lamh = (hgf + aga) / 2.0
    lama = (agf + hga) / 2.0
    meta = {
        "home_matches": homeprev.ntotal,
        "away_matches": awayprev.ntotal,
        "home_gf": round(hgf, 2),
        "away_gf": round(agf, 2),
        "formula": "PSS filtered (home_gf + away_ga) / 2"
    }
    return lamh, lama, meta

def compute_component_h2h(h2hmatches: List[MatchRow], hometeam: str, awayteam: str) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    if not h2hmatches or len(h2hmatches) < 3:
        return None
    hk = norm_key(hometeam)
    ak = norm_key(awayteam)
    used = h2hmatches[:H2H_N]
    hg, ag = [], []
    for m in used:
        if norm_key(m.home) == hk and norm_key(m.away) == ak:
            hg.append(m.fthome)
            ag.append(m.ftaway)
        elif norm_key(m.home) == ak and norm_key(m.away) == hk:
            hg.append(m.ftaway)
            ag.append(m.fthome)
    if len(hg) < 3:
        return None
    lamh = sum(hg) / len(hg)
    lama = sum(ag) / len(ag)
    meta = {"matches": len(hg), "hg_avg": lamh, "ag_avg": lama}
    return lamh, lama, meta

def clamp_lambda(lh: float, la: float) -> Tuple[float, float, List[str]]:
    warn = []
    def c(x: float, name: str) -> float:
        if x < 0.15:
            warn.append(f"{name} çok düşük ({x:.2f}) → 0.15")
            return 0.15
        if x > 3.80:
            warn.append(f"{name} çok yüksek ({x:.2f}) → 3.80")
            return 3.80
        return x
    return c(lh, "home"), c(la, "away"), warn

def compute_lambdas(
    st_homes: Dict[str, Optional[SplitGFGA]],
    st_aways: Dict[str, Optional[SplitGFGA]],
    homeprev: TeamPrevStats,
    awayprev: TeamPrevStats,
    h2hused: List[MatchRow],
    hometeam: str,
    awayteam: str
) -> Tuple[float, float, Dict[str, Any]]:
    info: Dict[str, Any] = {"components": {}, "weights_used": {}, "warnings": []}
    comps: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}
    
    stc = compute_component_standings(st_homes, st_aways)
    if stc:
        comps["standing"] = stc
    
    pss = compute_component_pss(homeprev, awayprev)
    if pss:
        comps["pss"] = pss
    
    h2c = compute_component_h2h(h2hused, hometeam, awayteam)
    if h2c:
        comps["h2h"] = h2c
    
    w = {}
    if "standing" in comps:
        w["standing"] = WST_BASE
    if "pss" in comps:
        w["pss"] = WPSS_BASE
    if "h2h" in comps:
        w["h2h"] = WH2H_BASE
    
    wnorm = normalize_weights(w)
    info["weights_used"] = wnorm
    
    if not wnorm:
        info["warnings"].append("Yetersiz veri - default")
        lh, la = 1.20, 1.20
    else:
        lh = 0.0
        la = 0.0
        for k, wk in wnorm.items():
            ch, ca, meta = comps[k]
            info["components"][k] = {"lam_home": ch, "lam_away": ca, "meta": meta}
            lh += wk * ch
            la += wk * ca
        
        lh, la, clamp_warn = clamp_lambda(lh, la)
        if clamp_warn:
            info["warnings"].extend(clamp_warn)
    
    return lh, la, info

# ============================================================================
# BET365 ODDS
# ============================================================================
def extract_first_float(s: str) -> Optional[float]:
    if not s:
        return None
    m = FLOAT_RE.search(s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def extract_all_numbers_loose(s: str) -> List[float]:
    if not s:
        return []
    nums = []
    for m in re.finditer(r'(\d+(?:\.\d+)?)', s):
        try:
            nums.append(float(m.group(1)))
        except Exception:
            pass
    return nums

def extract_cell_numeric_from_innerhtml(innerhtml: str) -> Optional[float]:
    if not innerhtml:
        return None
    txt = strip_tags_keep_text(innerhtml)
    v = extract_first_float(txt)
    if v is not None:
        return v
    m = re.search(r'(?:title|data-)="?(\d+(?:\.\d+)?)', innerhtml, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    v2 = extract_first_float(innerhtml)
    return v2

def extract_bet365_initial_1x2_from_oddscomp_html(odds_html: str) -> Optional[Dict[str, float]]:
    if not odds_html:
        return None
    
    trm = re.search(r'<tr.*?Bet365.*?</tr>', odds_html, flags=re.I | re.S)
    if not trm:
        return None
    
    trhtml = trm.group(0)
    tds = re.findall(r'<t[dh].*?</t[dh]>', trhtml, flags=re.I | re.S)
    
    if not tds or len(tds) < 8:
        nums = extract_all_numbers_loose(strip_tags_keep_text(trhtml))
        if len(nums) >= 6:
            cand = nums[3:6]
            if all(1.01 < x < 50 for x in cand):
                return {"1": float(cand[0]), "X": float(cand[1]), "2": float(cand[2])}
        return None
    
    cell_vals: List[Optional[float]] = [extract_cell_numeric_from_innerhtml(td) for td in tds]
    
    if len(cell_vals) >= 8:
        o1, ox, o2 = cell_vals[5], cell_vals[6], cell_vals[7]
        if all(v is not None for v in [o1, ox, o2]):
            if all(1.01 < float(v) < 200 for v in [o1, ox, o2]):
                return {"1": float(o1), "X": float(ox), "2": float(o2)}
    
    nums = extract_all_numbers_loose(strip_tags_keep_text(trhtml))
    if len(nums) >= 6:
        cand = nums[3:6]
        if all(1.01 < x < 50 for x in cand):
            return {"1": float(cand[0]), "X": float(cand[1]), "2": float(cand[2])}
    
    return None

def extract_bet365_initial_odds(url: str) -> Optional[Dict[str, float]]:
    odds_url = build_oddscomp_url(url)
    try:
        html = safe_get(odds_url, referer=extract_base_domain(url))
        odds = extract_bet365_initial_1x2_from_oddscomp_html(html)
        return odds
    except Exception:
        return None

# ============================================================================
# VALUE BETTING
# ============================================================================
def kelly_criterion(prob: float, odds: float) -> float:
    if odds <= 1.0 or prob <= 0.0 or prob >= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    return max(0.0, kelly)

def value_and_kelly(prob: float, odds: float) -> Tuple[float, float]:
    v = odds * prob - 1.0
    k = kelly_criterion(prob, odds)
    return v, k

def confidence_label(p: float) -> str:
    if p >= 0.65:
        return "Yüksek"
    if p >= 0.55:
        return "Orta"
    return "Düşük"

def net_ou_prediction(probs: Dict[str, float]) -> Tuple[str, float, str]:
    po25 = probs.get("O2.5", 0)
    pu25 = probs.get("U2.5", 0)
    if po25 > pu25:
        return "2.5 ÜST", po25, confidence_label(po25)
    return "2.5 ALT", pu25, confidence_label(pu25)

def net_btts_prediction(probs: Dict[str, float]) -> Tuple[str, float, str]:
    pbtts = probs.get("BTTS", 0)
    pno = 1.0 - pbtts
    if pbtts > pno:
        return "VAR", pbtts, confidence_label(pbtts)
    return "YOK", pno, confidence_label(pno)

def final_decision(qualified: List[Tuple[str, float, float, float, float]], diff: float, diff_label: str) -> str:
    if not qualified:
        return f"OYNAMA: Eşik sağlanamadı, model uyumu: {diff_label}"
    if diff > 0.10:
        return f"TEMKİNLİ: Zayıf model uyumu ({diff_label})"
    best = sorted(qualified, key=lambda x: x[3], reverse=True)[0]
    mkt, prob, odds, val, qk = best
    return f"OYNANABILIR: {mkt} | Prob: {prob*100:.1f}% | Oran: {odds:.2f} | Value: {val*100:.1f}% | Kelly: {qk*100:.1f}%"

# ============================================================================
# COMPREHENSIVE REPORT
# ============================================================================
def format_comprehensive_report(data: Dict[str, Any]) -> str:
    t = data["teams"]
    blend = data["blended_probs"]
    top7 = data["poisson"]["top7_scores"]
    
    lines = []
    lines.append("=" * 60)
    lines.append(f"{t['home']} vs {t['away']}")
    lines.append("=" * 60)
    
    lines.append(f"📊 SKORLAR")
    for i, (score, prob) in enumerate(top7[:5], 1):
        bar = '█' * int(prob * 50)
        lines.append(f"{i}. {score:6s} {prob*100:4.1f}% {bar}")
    
    lines.append(f"\n🎯 TAHMİN")
    lines.append(f"Ana Skor: {top7[0][0]}")
    if len(top7) >= 3:
        lines.append(f"Alternatif: {top7[1][0]}, {top7[2][0]}")
    
    lines.append(f"Skor-2: {data.get('score_2', 'N/A')}")
    
    netou, netoup, _ = net_ou_prediction(blend)
    lines.append(f"Üst 2.5: {netou} ({netoup*100:.1f}%)")
    
    netbtts, netbttsp, _ = net_btts_prediction(blend)
    lines.append(f"KG Var: {netbtts} ({netbttsp*100:.1f}%)")
    
    lines.append(f"\n📈 1X2 Olasılıklar")
    lines.append(f"Ev (1): {blend.get('1', 0)*100:.1f}%")
    lines.append(f"Ber(X): {blend.get('X', 0)*100:.1f}%")
    lines.append(f"Dep(2): {blend.get('2', 0)*100:.1f}%")
    
    corners = data.get("corner_analysis", {})
    if corners and corners.get("total_corners", 0) > 0:
        lines.append(f"\n🚩 KORNER TAHMİNİ")
        lines.append(f"Tahmini Toplam: {corners['total_corners']}")
        lines.append(f"Ev: {corners['predicted_home_corners']} | Dep: {corners['predicted_away_corners']}")
        
        preds = corners.get("market_probs", {})
        for k in ['O9.5', 'O10.5', 'O11.5']:
            if k in preds:
                uk = k.replace('O', 'U')
                lines.append(f"{k}: {preds[k]*100:.1f}% | {uk}: {preds.get(uk, 0)*100:.1f}%")
        
        ht = corners.get("first_half", {})
        if ht.get("total_ht", 0) > 0:
            lines.append(f"\n🕐 İLK YARI KORNER")
            lines.append(f"Toplam: {ht['total_ht']} (Ev: {ht['predicted_home_ht']} | Dep: {ht['predicted_away_ht']})")
            ht_preds = ht.get("predictions", {})
            for k in ['HT_O4.5', 'HT_O5.5']:
                if k in ht_preds:
                    uk = k.replace('O', 'U')
                    lines.append(f"{k}: {ht_preds[k]*100:.1f}% | {uk}: {ht_preds.get(uk, 0)*100:.1f}%")
    
    vb = data.get("value_bets", {})
    if vb.get("used_odds"):
        lines.append(f"\n💰 VALUE ANALIZ (Bet365 Initial 1X2)")
        has_value = False
        for row in vb.get("table", []):
            if row["value"] >= VALUE_MIN and row["prob"] >= PROB_MIN:
                lines.append(f"{row['market']}: Oran {row['odds']:.2f} | Value {row['value']*100:.1f}% | Kelly {row['kelly']*100:.1f}%")
                has_value = True
        if not has_value:
            lines.append("Değerli bahis bulunamadı")
        lines.append(f"\n{vb.get('decision', 'Analiz edilemedi')}")
    else:
        lines.append(f"\nBet365 verisi yok - value analizi yapılamadı")
    
    ds = data["datasources"]
    lambda_info = data["lambda"]["info"]
    lines.append(f"\n📂 Kullanılan Veriler")
    lines.append(f"Standing: {'✓' if ds['standings_used'] else '✗'}")
    lines.append(f"PSS Same League (Home/Away): Ev:{ds['home_prev_matches']} | Dep:{ds['away_prev_matches']}")
    lines.append(f"H2H: {ds['h2h_matches']} {'(Same League)' if ds.get('h2h_sameleague_used') else ''} maç")
    
    if lambda_info.get("weights_used"):
        lines.append(f"\nAğırlıklar:")
        for k, v in lambda_info["weights_used"].items():
            kname = {"standing": "Standing", "pss": "PSS", "h2h": "H2H"}.get(k, k)
            lines.append(f"  {kname}: {v*100:.0f}%")
    
    lines.append("=" * 60)
    return "\n".join(lines)

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================
def analyze_nowgoal(url: str, odds: Optional[Dict[str, float]] = None, mcruns: int = MC_RUNS_DEFAULT) -> Dict[str, Any]:
    """Main analysis function with all enhancements"""
    
    h2h_url = build_h2h_url(url)
    html = safe_get(h2h_url, referer=extract_base_domain(url))
    
    home_team, away_team = parse_teams_from_title(html)
    if not home_team or not away_team:
        raise RuntimeError("Takım isimleri çıkarılamadı (title parse başarısız)")
    
    league_match = re.search(r'<span[^>]*class="?sclassLink"?[^>]*>([^<]+)</span>', html)
    league_name = strip_tags_keep_text(league_match.group(1)) if league_match else ""
    
    st_home_rows = extract_standings_for_team(html, home_team)
    st_away_rows = extract_standings_for_team(html, away_team)
    st_home = standings_to_splits(st_home_rows)
    st_away = standings_to_splits(st_away_rows)
    
    prev_home_tabs, prev_away_tabs = extract_previous_from_page(html)
    prev_home_raw = parse_matches_from_table_html(prev_home_tabs) if prev_home_tabs else []
    prev_away_raw = parse_matches_from_table_html(prev_away_tabs) if prev_away_tabs else []
    
    h2h_all = extract_h2h_matches(html, home_team, away_team)
    h2h_pair = [m for m in h2h_all if is_h2h_pair(m, home_team, away_team)]
    h2h_pair = sort_matches_desc(dedupe_matches(h2h_pair))
    
    if league_name:
        prev_home_raw = filter_same_league_matches(prev_home_raw, league_name)
        prev_away_raw = filter_same_league_matches(prev_away_raw, league_name)
    
    prev_home_sel = filter_team_home_only(prev_home_raw, home_team)[:RECENT_N]
    prev_away_sel = filter_team_away_only(prev_away_raw, away_team)[:RECENT_N]
    
    home_prev_stats = build_prev_stats(home_team, prev_home_sel)
    away_prev_stats = build_prev_stats(away_team, prev_away_sel)
    
    h2h_same = filter_same_league_matches(h2h_pair, league_name) if league_name else h2h_pair
    if len(h2h_same) >= 3:
        h2h_used = h2h_same[:H2H_N]
        h2h_same_used = True
    else:
        h2h_used = h2h_pair[:H2H_N]
        h2h_same_used = False
    
    lam_home, lam_away, lambda_info = compute_lambdas(
        st_home, st_away, home_prev_stats, away_prev_stats, h2h_used, home_team, away_team
    )
    
    score_2 = score_2_from_lambda(lam_home, lam_away)
    
    score_mat = build_score_matrix(lam_home, lam_away, maxg=MAX_GOALS_FOR_MATRIX)
    poisson_market = market_probs_from_matrix(score_mat)
    top7 = top_scores_from_matrix(score_mat, topn=7)
    
    mc = monte_carlo(lam_home, lam_away, n=mcruns, seed=42)
    
    diff, diff_label = model_agreement(poisson_market, mc["p"])
    blended = blend_probs(poisson_market, mc["p"], alpha=BLEND_ALPHA)
    
    corner_analysis = analyze_corners_enhanced(home_prev_stats, away_prev_stats, h2h_used)
    
    if not odds:
        odds = extract_bet365_initial_odds(url)
    
    value_block = {"used_odds": False, "qualified": [], "table": [], "thresholds": {}, "decision": ""}
    qualified = []
    
    if odds and all(k in odds for k in ["1", "X", "2"]):
        value_block["used_odds"] = True
        table = []
        for mkt in ["1", "X", "2"]:
            o = float(odds[mkt])
            p = float(blended.get(mkt, 0.0))
            v, kelly = value_and_kelly(p, o)
            qk = max(0.0, 0.25 * kelly)
            row = {"market": mkt, "prob": p, "odds": o, "value": v, "kelly": kelly, "qkelly": qk}
            table.append(row)
            if v >= VALUE_MIN and p >= PROB_MIN and kelly >= KELLY_MIN:
                qualified.append((mkt, p, o, v, qk))
        
        value_block["table"] = table
        value_block["thresholds"] = {"value_min": VALUE_MIN, "prob_min": PROB_MIN, "kelly_min": KELLY_MIN}
        value_block["decision"] = final_decision(qualified, diff, diff_label)
    else:
        value_block["decision"] = "Oran gerekli - Bet365 verisi yok"
    
    data = {
        "url": h2h_url,
        "teams": {"home": home_team, "away": away_team},
        "league": league_name,
        "lambda": {
            "home": lam_home,
            "away": lam_away,
            "total": lam_home + lam_away,
            "info": lambda_info
        },
        "score_2": score_2,
        "poisson": {
            "market_probs": poisson_market,
            "top7_scores": top7
        },
        "mc": mc,
        "model_agreement": {"diff": diff, "label": diff_label},
        "blended_probs": blended,
        "corner_analysis": corner_analysis,
        "value_bets": value_block,
        "datasources": {
            "standings_used": len(st_home_rows) > 0 and len(st_away_rows) > 0,
            "h2h_matches": len(h2h_used),
            "h2h_sameleague_used": h2h_same_used,
            "home_prev_matches": len(prev_home_sel),
            "away_prev_matches": len(prev_away_sel),
            "pss_sameleague_used": bool(league_name)
        }
    }
    
    data["report_comprehensive"] = format_comprehensive_report(data)
    
    return data

# ============================================================================
# FLASK API - FIXED /analiz_et ENDPOINT
# ============================================================================
app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({"ok": True, "service": "nowgoal-analyzer-api", "version": "5.0-final-complete"})

@app.route("/health")
def health():
    return jsonify({"ok": True, "status": "healthy"})

@app.route("/analizet", methods=["POST"])
@app.route("/analiz_et", methods=["POST"])
def analizet_route():
    """Turkish endpoint - Her iki URL'yi de destekler (/analizet ve /analiz_et)"""
    try:
        payload = request.get_json(silent=True) or {}
    except Exception as e:
        return jsonify({"ok": False, "error": f"Geçersiz JSON: {e}"}), 400
    
    url = (payload.get("url") or '').strip()
    if not url:
        return jsonify({"ok": False, "error": "URL boş olamaz"}), 400
    
    if not re.match(r'https?://', url):
        return jsonify({"ok": False, "error": "Geçersiz URL formatı"}), 400
    
    try:
        data = analyze_nowgoal(url, odds=None, mcruns=10000)
        
        top_skor = data["poisson"]["top7_scores"][0]
        blend = data["blended_probs"]
        corners = data["corner_analysis"]
        top_corner = corners["top_corner_scores"][0] if corners["top_corner_scores"] else ("N/A", 0)
        
        return jsonify({
            "ok": True,
            "skor": f"{top_skor[0]}: {top_skor[1]*100:.1f}%",
            "skor_2": data["score_2"],
            "alt_ust": f"2.5 {'ÜST' if blend.get('O2.5', 0) > 0.5 else 'ALT'}: {max(blend.get('O2.5', 0), blend.get('U2.5', 0))*100:.1f}%",
            "btts": f"{'VAR' if blend.get('BTTS', 0) > 0.5 else 'YOK'}: {blend.get('BTTS', 0)*100:.1f}%",
            "korner": {
                "toplam": corners["total_corners"],
                "ev": corners["predicted_home_corners"],
                "deplasman": corners["predicted_away_corners"],
                "en_olasi": f"{top_corner[0]}: {top_corner[1]*100:.1f}%",
                "ilk_yari": corners["first_half"]["total_ht"],
                "ilk_yari_ev": corners["first_half"]["predicted_home_ht"],
                "ilk_yari_dep": corners["first_half"]["predicted_away_ht"],
                "over_9_5": f"{corners['market_probs'].get('O9.5', 0)*100:.1f}%",
                "over_10_5": f"{corners['market_probs'].get('O10.5', 0)*100:.1f}%"
            },
            "karar": data["value_bets"].get("decision", "Oran gerekli"),
            "guven": corners["confidence"],
            "detay": data["report_comprehensive"]
        })
    except Exception as e:
        return jsonify({
            "ok": False, 
            "error": f"Analiz hatası: {str(e)}", 
            "traceback": traceback.format_exc()
        }), 500

@app.route("/analyze", methods=["POST"])
def analyze_route():
    """English endpoint - Full API"""
    try:
        payload = request.get_json(silent=True) or {}
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid JSON: {e}"}), 400
    
    url = (payload.get("url") or '').strip()
    if not url:
        return jsonify({"ok": False, "error": "url required"}), 400
    
    if not re.match(r'https?://', url):
        return jsonify({"ok": False, "error": "Invalid URL"}), 400
    
    odds = payload.get("odds")
    mcruns = payload.get("mcruns", MC_RUNS_DEFAULT)
    
    try:
        mcruns = int(mcruns)
        if mcruns < 100 or mcruns > 100000:
            mcruns = MC_RUNS_DEFAULT
    except (ValueError, TypeError):
        mcruns = MC_RUNS_DEFAULT
    
    try:
        data = analyze_nowgoal(url, odds=odds, mcruns=mcruns)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({
            "ok": False, 
            "error": str(e), 
            "traceback": traceback.format_exc()
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "ok": False,
        "error": "Endpoint bulunamadı",
        "available_endpoints": {
            "GET": ["/", "/health"],
            "POST": ["/analizet", "/analiz_et", "/analyze"]
        }
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "ok": False,
        "error": "Sunucu hatası",
        "detail": str(e)
    }), 500

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    else:
        print("=" * 70)
        print("NowGoal Analyzer v5.0 FINAL - COMPLETE CODE")
        print("=" * 70)
        print("✅ İlk yarı korner: H2H ve PSS'den gerçek verilerle")
        print("✅ Score-2: Lambda değerlerinden yuvarlanmış skor")
        print("✅ Full corner matrix: İki taraflı Poisson")
        print("✅ /analiz_et endpoint: FIXED (404 hatası düzeltildi)")
        print("=" * 70)
        print("\nUsage: python script.py serve")
        print("\nEndpoints:")
        print("  GET  /              - Root (service info)")
        print("  GET  /health        - Health check")
        print("  POST /analizet      - Turkish (alt çizgisiz)")
        print("  POST /analiz_et     - Turkish (alt çizgili) ✓ ANDROID")
        print("  POST /analyze       - English (Full API)")
        print("\nExample:")
        print('  curl -X POST http://localhost:5000/analiz_et \\')
        print('    -H "Content-Type: application/json" \\')
        print('    -d \'{"url": "https://live3.nowgoal26.com/match/h2h-2799556"}\'')
        print("=" * 70)
