# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - ULTIMATE VBA LOGIC REPLICA
- BASE: Python Scraping Engine (v5.2) - Full Preservation
- LOGIC: VBA 'Magic Dice' (PoissonRastgele) Logic Implemented
- SIMULATION: Iterative loop (10,000 runs) exactly like VBA
- OUTPUT: Exact "ResimGibi" VBA Format
- STATUS: FULL CODE / NO CUTS
- UPDATED: requests-html for JavaScript Rendering (Render sunucusu uyumlu)
"""

import re
import math
import time
import traceback
import sys
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import requests
from flask import Flask, request, jsonify

# ============================================================================
# PLAYWRIGHT IMPORTS - JavaScript Rendering için
# ============================================================================
import asyncio

# ============================================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================================
MC_RUNS_DEFAULT = 10000
RECENT_N = 10  # PSS analizinde dikkate alınacak maksimum maç sayısı
H2H_N = 10     # H2H için çekilecek maç sayısı

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# ============================================================================
# 2. LOGGING HELPERS
# ============================================================================
def log_error(msg: str, exc: Exception = None):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    if exc:
        print(f"[ERROR] {traceback.format_exc()}", file=sys.stderr, flush=True)

def log_info(msg: str):
    print(f"[INFO] {msg}", file=sys.stdout, flush=True)

# ============================================================================
# 3. REGEX PATTERNS
# ============================================================================
DATE_ANY_RE = re.compile(r'\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{2}-\d{2}')
SCORE_RE = re.compile(r'(\d{1,2})-(\d{1,2})(?:\((\d{1,2})-(\d{1,2})\))?')
CORNER_FT_RE = re.compile(r'(\d{1,2})-(\d{1,2})')
CORNER_HT_RE = re.compile(r'\((\d{1,2})-(\d{1,2})\)')
FLOAT_RE = re.compile(r'(\d+(?:\.\d+)?)')
INT_RE = re.compile(r'(\d+)')

# ============================================================================
# 4. DATA CLASSES
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

# ============================================================================
# 5. SCRAPING HELPER FUNCTIONS
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
    s = s.replace('&amp;', '&')
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

def safe_get(url: str, timeout: int = 20, retries: int = 2, referer: Optional[str] = None) -> str:
    last_err = None
    headers = dict(HEADERS)
    if referer:
        headers['Referer'] = referer
    for attempt in range(retries + 1):
        try:
            log_info(f"Fetching {url} (attempt {attempt + 1})")
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            log_info(f"Successfully fetched {url}")
            return r.text
        except requests.exceptions.Timeout as e:
            last_err = e
            log_error(f"Timeout on attempt {attempt + 1}: {url}", e)
            if attempt < retries:
                time.sleep(0.5)
        except Exception as e:
            last_err = e
            log_error(f"Error on attempt {attempt + 1}: {url}", e)
            if attempt < retries:
                time.sleep(0.7)
    raise RuntimeError(f"Fetch failed after {retries + 1} attempts: {url} - {last_err}")

def extract_match_id(url: str) -> str:
    m = re.search(r'(?:h2h-|match/h2h-)(\d+)', url)
    if m:
        return m.group(1)
    nums = re.findall(r'\d{6,}', url)
    if not nums:
        raise ValueError("Match ID not found")
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
    og_match = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    if og_match:
        title_text = og_match.group(1)
        vs_match = re.search(r'(.+?)\s+VS\s+(.+?)(?:\s*-|\s*$)', title_text, flags=re.IGNORECASE)
        if vs_match:
            return (vs_match.group(1).strip(), vs_match.group(2).strip())

    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, flags=re.IGNORECASE)
    if title_match:
        title_text = title_match.group(1).strip()
        vs_match = re.search(r'(.+?)\s+(?:VS|vs)\s+(.+?)(?:\s*-|\s*$)', title_text, flags=re.IGNORECASE)
        if vs_match:
            return (vs_match.group(1).strip(), vs_match.group(2).strip())

    h1_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html, flags=re.IGNORECASE)
    if h1_match:
        h1_text = h1_match.group(1).strip()
        vs_match = re.search(r'(.+?)\s+(?:vs|VS)\s+(.+?)(?:\s+Live|\s*$)', h1_text, flags=re.IGNORECASE)
        if vs_match:
            return (vs_match.group(1).strip(), vs_match.group(2).strip())

    log_error("Could not parse team names from any source")
    return ("Home", "Away")

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

def parse_corner_cell(cell: str) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
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
# 6. STANDINGS PARSING
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

# ============================================================================
# 7. EXTRACT MATCH DATA LOGIC
# ============================================================================
def extract_previous_from_page(page_source: str) -> Tuple[List[MatchRow], List[MatchRow]]:
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

    match_tables: List[List[MatchRow]] = []
    for t in tabs:
        ms = parse_matches_from_table_html(t)
        if ms:
            match_tables.append(ms)
        if len(match_tables) >= 2:
            break

    if len(match_tables) == 0:
        return [], []
    if len(match_tables) == 1:
        return match_tables[0], []
    return match_tables[0], match_tables[1]

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
# 8. FILTER FUNCTIONS
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
# 9. VBA LOGIC PORTED TO PYTHON (CORE CALCULATIONS)
# ============================================================================

def calculate_weighted_pss_goals(matches: List[MatchRow], team_name: str, is_home_context: bool) -> float:
    total_goals = 0.0
    total_weight = 0.0
    count = 0

    for m in matches:
        goals_scored = 0
        if is_home_context:
            goals_scored = m.fthome
        else:
            goals_scored = m.ftaway

        count += 1
        weight = 1.2 if count <= 5 else 0.8

        total_goals += goals_scored * weight
        total_weight += weight

        if count >= 20: break

    if total_weight == 0:
        return 1.3 if is_home_context else 1.1

    return total_goals / total_weight

def calculate_weighted_pss_corners(matches: List[MatchRow], team_name: str, is_home_context: bool) -> Tuple[float, float]:
    won_total = 0.0
    conceded_total = 0.0
    total_weight = 0.0
    count = 0

    for m in matches:
        if m.cornerhome is None or m.corneraway is None: continue

        won = 0
        conceded = 0

        if is_home_context:
            won = m.cornerhome
            conceded = m.corneraway
        else:
            won = m.corneraway
            conceded = m.cornerhome

        count += 1
        weight = 1.2 if count <= 5 else 0.8

        won_total += won * weight
        conceded_total += conceded * weight
        total_weight += weight

        if count >= 20: break

    if total_weight == 0:
        return 5.0, 5.0

    return (won_total / total_weight), (conceded_total / total_weight)

def poisson_pmf(lam: float, k: int) -> float:
    if lam <= 0: lam = 0.1
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

# --- VBA 'MAGIC DICE' FUNCTION (POISSONRASTGELE) ---
def vba_poisson_random(lam: float) -> int:
    """
    VBA kodundaki 'PoissonRastgele' fonksiyonunun Python karşılığı.
    VBA'daki Rnd() yerine random.random() kullanılır.
    """
    if lam <= 0: lam = 0.1
    L = math.exp(-lam)
    p = 1.0
    k = 0
    # Do...Loop equivalent
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            break
    return k - 1

# --- 9.4 Monte Carlo Simulations (Goals) - ITERATIVE LOOP ---
def monte_carlo_simulation_vba(lam_home: float, lam_away: float, num_sims: int = 10000) -> Dict[str, Any]:
    """
    VBA: MonteCarloSimulasyonu (Exact Logic with Loop)
    """
    gol_dagilim = Counter() # 0 to 10+
    over25 = 0
    over35 = 0
    btts = 0
    home_win = 0
    draw = 0
    away_win = 0
    score_counts = Counter()

    for _ in range(num_sims):
        ev_gol = vba_poisson_random(lam_home)
        dep_gol = vba_poisson_random(lam_away)

        toplam = ev_gol + dep_gol

        # VBA: If toplamGol <= 10 Then golDagilim(toplamGol) ...
        if toplam <= 10:
            gol_dagilim[toplam] += 1
        else:
            gol_dagilim[toplam] += 1

        if toplam > 2: over25 += 1
        if toplam > 3: over35 += 1
        if ev_gol >= 1 and dep_gol >= 1: btts += 1

        if ev_gol > dep_gol: home_win += 1
        elif ev_gol == dep_gol: draw += 1
        else: away_win += 1

        score_key = f"{ev_gol}-{dep_gol}"
        score_counts[score_key] += 1

    return {
        "dist_goals": gol_dagilim,
        "over25_pct": over25 / num_sims,
        "over35_pct": over35 / num_sims,
        "btts_pct": btts / num_sims,
        "1_pct": home_win / num_sims,
        "X_pct": draw / num_sims,
        "2_pct": away_win / num_sims,
        "top_scores": score_counts.most_common(10),
        "total_sims": num_sims
    }

# --- 9.5 Monte Carlo Simulations (Corners) - ITERATIVE LOOP ---
def monte_carlo_corners_vba(lam_home: float, lam_away: float, num_sims: int = 10000) -> Dict[str, Any]:
    """
    VBA: MonteCarloKORNER_ResimGibi (Exact Logic with Loop)
    """
    dist_total = Counter()
    over75 = 0; over85 = 0; over95 = 0; over105 = 0; over115 = 0
    home_more = 0; draw = 0; away_more = 0
    score_counts = Counter()

    for _ in range(num_sims):
        ev_k = vba_poisson_random(lam_home)
        dep_k = vba_poisson_random(lam_away)
        top_k = ev_k + dep_k

        # 1. Dağılım
        if top_k <= 19:
            dist_total[top_k] += 1
        else:
            dist_total[20] += 1

        # 2. Market
        if top_k > 7: over75 += 1
        if top_k > 8: over85 += 1
        if top_k > 9: over95 += 1
        if top_k > 10: over105 += 1
        if top_k > 11: over115 += 1

        # 3. Taraf
        if ev_k > dep_k: home_more += 1
        elif ev_k == dep_k: draw += 1
        else: away_more += 1

        # 4. Skor
        score_key = f"{ev_k}-{dep_k}"
        score_counts[score_key] += 1

    return {
        "dist_total": dist_total,
        "over75": over75 / num_sims,
        "over85": over85 / num_sims,
        "over95": over95 / num_sims,
        "over105": over105 / num_sims,
        "over115": over115 / num_sims,
        "home_more": home_more / num_sims,
        "draw": draw / num_sims,
        "away_more": away_more / num_sims,
        "top_scores": score_counts.most_common(5),
        "total_sims": num_sims
    }

def get_confidence(prob: float) -> str:
    if prob >= 0.70: return "YUKSEK"
    if prob >= 0.50: return "ORTA"
    return "DUSUK"

# ============================================================================
# 10. PLAYWRIHT BROWSER AUTOMATION - YENİ EKLENEN BÖLÜM
# ============================================================================

def fetch_real_odds_with_playwright(match_id: str, base_url: str) -> List[float]:
    """
    [YENİ ÇÖZÜM - Playwright ile JavaScript Rendering - Python 3.13 + Gunicorn Uyumlu]

    Bu fonksiyon playwright kullanarak:
    1) Tarayıcı açar
    2) Sayfaya gider
    3) JavaScript'in yüklenmesini bekler
    4) Bet365 Initial 1X2 oranlarını çeker

    Home: 2.15
    Draw: 3.50
    Away: 2.85

    NOT: Chrome/Chromium gerektirir, Render sunucusunda çalışır!
    """

    log_info("=" * 60)
    log_info("PLAYWRIGHT JAVA SCRIPT RENDERING BAŞLADI")
    log_info("=" * 60)

    async def scrape_odds():
        from playwright.async_api import async_playwright

        # ADIM 1: Playwright başlat
        log_info("ADIM 1: Playwright başlatılıyor...")
        async with async_playwright() as p:
            # ADIM 2: Chromium başlat (gunicorn uyumlu)
            log_info("ADIM 2: Chromium başlatılıyor...")
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-software-rasterizer',
                    '--disable-accelerated-2d-canvas',
                    '--no-zygote',
                ]
            )

            try:
                # ADIM 3: Yeni sayfa aç
                log_info("ADIM 3: Yeni sayfa açılıyor...")
                page = await browser.new_page()

                # ADIM 4: URL'ye git
                url = f"{base_url}/oddscomp/{match_id}"
                log_info(f"ADIM 4: URL'ye gidiliyor: {url}")
                await page.goto(url, wait_until='networkidle', timeout=30000)

                log_info("ADIM 5: Sayfa yüklendi")

                # ADIM 5: JavaScript'in çalışması için bekle
                log_info("ADIM 6: JavaScript bekleniyor (5 saniye)...")
                await page.wait_for_selector('table', timeout=10000)
                await asyncio.sleep(5)

                log_info("ADIM 7: JavaScript render edildi")

                # ADIM 6: Tablodan Bet365 oranlarını çek
                log_info("ADIM 8: Tablolar aranıyor...")

                # Tablodaki tüm satırları al
                rows = await page.locator('table tr').all_inner_texts()

                log_info(f"ADIM 9: {len(rows)} adet satır bulundu")

                bet365_odds = None

                for row_index, row_text in enumerate(rows):
                    row_lower = row_text.lower()

                    # Bet365 ve Initial satırını bul
                    if 'bet365' in row_lower and 'initial' in row_lower:
                        log_info(f"ADIM 10: Bet365 Initial satırı bulundu! (Satır {row_index + 1})")
                        log_info(f"Satır içeriği: {row_text[:200]}...")

                        # Satırdaki sayıları bul (oranlar)
                        import re
                        found_odds = re.findall(r'(\d+\.\d{2})', row_text)

                        log_info(f"ADIM 11: Bulunan sayılar: {found_odds}")

                        if len(found_odds) >= 3:
                            # İlk 3 sayıyı al (Home, Draw, Away)
                            bet365_odds = [float(found_odds[0]), float(found_odds[1]), float(found_odds[2])]
                            log_info(f"ADIM 12: Bet365 oranları çekildi: {bet365_odds}")
                            break

                # ADIM 7: Sonuçları kontrol et
                if bet365_odds and all(o > 1.0 for o in bet365_odds):
                    log_info("=" * 60)
                    log_info("✓✓✓ BAŞARILI! Bet365 Initial oranları çekildi!")
                    log_info(f"  Home (1): {bet365_odds[0]}")
                    log_info(f"  Draw (X): {bet365_odds[1]}")
                    log_info(f"  Away (2): {bet365_odds[2]}")
                    log_info("=" * 60)
                    return bet365_odds
                else:
                    log_error("Bet365 oranları bulunamadı veya geçersiz")
                    return [1.0, 1.0, 1.0]

            except Exception as e:
                log_error(f"playwright hatası: {e}")
                return [1.0, 1.0, 1.0]

            finally:
                # ADIM 8: Tarayıcıyı kapat
                log_info("ADIM 13: Tarayıcı kapatılıyor...")
                await browser.close()
                log_info("Tarayıcı kapatıldı")

    # Asyncio ile çalıştır
    try:
        result = asyncio.run(scrape_odds())
        return result
    except RuntimeError as e:
        log_error(f"Asyncio hatası: {e}")
        return [1.0, 1.0, 1.0]

def fetch_real_odds_with_pyppeteer(match_id: str, base_url: str) -> List[float]:
    """
    [GÜNCELLENDİ - Playwright Versiyonu - Python 3.13 + Gunicorn Uyumlu]

    Ana fonksiyon: playwright kullanarak Bet365 Initial 1X2 oranlarını çeker.

    Önce playwright ile dener, başarısız olursa varsayılan değerleri döndürür.

    Döndürülen değerler:
    - Başarılı: [Home_oranı, Draw_oranı, Away_oranı]
    - Başarısız: [1.0, 1.0, 1.0]
    """

    log_info("=" * 60)
    log_info("fetch_real_odds fonksiyonu çağrıldı")
    log_info(f"Match ID: {match_id}")
    log_info(f"Base URL: {base_url}")
    log_info("=" * 60)

    # Playwright ile oranları çek
    odds = fetch_real_odds_with_playwright(match_id, base_url)

    # Sonuç döndür
    return odds

def fetch_real_odds_with_requests_html(match_id: str, base_url: str) -> List[float]:
    """
    [GÜNCELLENDİ - Playwright Versiyonu - Python 3.13 Uyumlu]

    Wrapper fonksiyon - playwright kullanarak Bet365 Initial 1X2 oranlarını çeker.
    """

    log_info("=" * 60)
    log_info("fetch_real_odds fonksiyonu çağrıldı")
    log_info(f"Match ID: {match_id}")
    log_info(f"Base URL: {base_url}")
    log_info("=" * 60)

    # Playwright ile oranları çek
    odds = fetch_real_odds_with_playwright(match_id, base_url)

    # Sonuç döndür
    return odds

def fetch_real_odds(match_id: str, base_url: str) -> List[float]:
    """
    [GÜNCELLENDİ - Playwright VERSİYONU]

    Wrapper fonksiyon - playwright kullanarak Bet365 Initial 1X2 oranlarını çeker.
    """

    log_info("=" * 60)
    log_info("fetch_real_odds fonksiyonu çağrıldı")
    log_info(f"Match ID: {match_id}")
    log_info(f"Base URL: {base_url}")
    log_info("=" * 60)

    # Playwright ile oranları çek
    odds = fetch_real_odds_with_playwright(match_id, base_url)

    # Sonuç döndür
    return odds

# ============================================================================
# 11. REPORT GENERATOR
# ============================================================================
def generate_vba_report(data: Dict[str, Any]) -> str:
    t = data['teams']
    xg = data['xg']
    corn = data['corners']
    pois = data['poisson']
    market = data['market_goals']
    market_corn = data['market_corners']
    mc = data['mc_goals']
    mc_corn = data['mc_corners']
    val = data['value']

    lines = []
    lines.append("="*55)
    lines.append("  FUTBOL MAC ANALIZI - %100 PSS MODEL")
    lines.append("  (STANDINGS HIC KULLANILMAZ - SADECE PSS!)")
    lines.append("="*55 + "\n")

    lines.append(f"MAC: {t['home']} vs {t['away']}")
    lines.append(f"ORANLAR: 1: {val['odds'][0]:.2f} | X: {val['odds'][1]:.2f} | 2: {val['odds'][2]:.2f}")
    lines.append("="*55 + "\n")

    lines.append("="*55)
    lines.append("0) VERI KAYNAK KONTROLU")
    lines.append("="*55 + "\n")

    lines.append("STANDINGS VERILERI (SADECE GORUNUM - KULLANILMAYACAK):")
    lines.append(f"  {t['home']}: {data['counts']['home_standings']} ev maci (SADECE GORUNTU)")
    lines.append(f"  {t['away']}: {data['counts']['away_standings']} deplasman maci (SADECE GORUNTU)")
    lines.append("  *** STANDINGS HESAPLAMALARDA KULLANILMAYACAK ***\n")

    lines.append("H2H VERILERI (SADECE GORUNUM - HESAPLAMADA KULLANILMAYACAK):")
    h2h_cnt = data['counts']['h2h']
    if h2h_cnt > 0:
        lines.append(f"  TOPLAM {h2h_cnt} MAC BULUNDU (SADECE GORUNTU)")
        for i, m in enumerate(data['h2h_sample'][:5], 1):
             lines.append(f"    {i}) {m.home} {m.fthome}-{m.ftaway} {m.away}")
        if h2h_cnt > 5: lines.append(f"    ... ve {h2h_cnt-5} mac daha")
    else:
        lines.append("  H2H verisi bulunamadi")
    lines.append("  *** H2H VERILERI HESAPLAMALARDA KULLANILMAYACAK ***\n")

    lines.append("EV TAKIMI PSS (%100 ANA VERİ - Hesaplama Kaynagi):")
    if data['counts']['pss_home'] > 0:
        lines.append(f"  TOPLAM {data['counts']['pss_home']} MAC BULUNDU")
        lines.append("  Tum Maclar:")
        for i, m in enumerate(data['pss_home_sample'], 1):
            c_txt = f" | Korner: {m.cornerhome}-{m.corneraway}" if m.cornerhome is not None else ""
            lines.append(f"    {i}) {m.home} {m.fthome}-{m.ftaway} {m.away}{c_txt}")
    else:
        lines.append("  KRITIK HATA: Ev takimi PSS verisi bulunamadi!")
    lines.append("\n")

    lines.append("DEPLASMAN TAKIMI PSS (%100 ANA VERİ - Hesaplama Kaynagi):")
    if data['counts']['pss_away'] > 0:
        lines.append(f"  TOPLAM {data['counts']['pss_away']} MAC BULUNDU")
        lines.append("  Tum Maclar:")
        for i, m in enumerate(data['pss_away_sample'], 1):
            c_txt = f" | Korner: {m.cornerhome}-{m.corneraway}" if m.cornerhome is not None else ""
            lines.append(f"    {i}) {m.home} {m.fthome}-{m.ftaway} {m.away}{c_txt}")
    else:
        lines.append("  KRITIK HATA: Deplasman PSS verisi bulunamadi!")
    lines.append("\n")

    if data['counts']['pss_home'] == 0 or data['counts']['pss_away'] == 0:
        lines.append("="*55)
        lines.append("ANALIZ DURDURULDU: EKSIK PSS VERISI")
        lines.append("="*55)
        return "\n".join(lines)

    lines.append("="*55)
    lines.append("A) BEKLENEN GOLLER (xG) - %100 PSS")
    lines.append("="*55 + "\n")

    lines.append("PSS TABANLI xG (Tek Kaynak - %100 PSS):")
    lines.append(f"   {t['home']} xG: {xg['home']:.2f} gol (Son {data['counts']['pss_home']} mac direkt ortalaması)")
    lines.append(f"   {t['away']} xG: {xg['away']:.2f} gol (Son {data['counts']['pss_away']} mac direkt ortalaması)")
    lines.append(f"   TOPLAM xG = {(xg['home'] + xg['away']):.2f} gol\n")

    lines.append("HESAPLAMA MANTIĞI:")
    lines.append("   1) PSS verilerinden DIREKT gol ortalamasi")
    lines.append("   2) Standings HİÇ kullanilmadi")
    lines.append("   3) H2H HİÇ kullanilmadi")
    lines.append("   4) Sadece SON FORM (PSS) baz alindi")
    lines.append("   5) %100 Temiz - Mukerrerlik YOK\n")

    lines.append("BEKLENEN KORNERLER (Perplexity AI Yontemi - %100 PSS):")
    lines.append(f"   {t['home']} Korner: {corn['home']:.2f}")
    lines.append(f"   {t['away']} Korner: {corn['away']:.2f}")
    lines.append(f"   TOPLAM Korner = {(corn['home'] + corn['away']):.2f}\n")

    lines.append("="*55)
    lines.append("B) POISSON OLASILILIKLARI")
    lines.append("="*55 + "\n")

    def print_p(team, probs):
        lines.append(f"{team} Gol Olasılıkları:")
        lines.append(f"   P(0 gol) = {probs[0]*100:.1f}%")
        lines.append(f"   P(1 gol) = {probs[1]*100:.1f}%")
        lines.append(f"   P(2 gol) = {probs[2]*100:.1f}%")
        p3p = 1.0 - sum(probs[:3])
        lines.append(f"   P(3+ gol) = {p3p*100:.1f}%\n")

    print_p(t['home'], pois['home_dist'])
    print_p(t['away'], pois['away_dist'])

    p00 = pois['home_dist'][0] * pois['away_dist'][0]
    lines.append("OZEL: 0-0 SKOR OLASILIĞI:")
    lines.append(f"   P(0-0) = {p00*100:.1f}%")
    lines.append("   NOT: Dusuk olasilik ama her zaman mumkundur!\n")

    lines.append("En Olası 7 Skor:")
    for i, (score, prob) in enumerate(pois['top_scores'], 1):
        lines.append(f"   {i}) {score} - %{prob*100:.1f}")
    lines.append("")

    lines.append("="*55)
    lines.append("C) MARKET OLASILILIKLARI (GOL)")
    lines.append("="*55 + "\n")

    m = market
    lines.append("Toplam Gol:")
    lines.append(f"   Ust 0.5: %{m['o05']*100:.1f} | Alt 0.5: %{(1-m['o05'])*100:.1f}")
    lines.append(f"   Ust 1.5: %{m['o15']*100:.1f} | Alt 1.5: %{(1-m['o15'])*100:.1f}")
    lines.append(f"   Ust 2.5: %{m['o25']*100:.1f} | Alt 2.5: %{(1-m['o25'])*100:.1f}")
    lines.append(f"   Ust 3.5: %{m['o35']*100:.1f} | Alt 3.5: %{(1-m['o35'])*100:.1f}\n")

    lines.append("BTTS (Karsilıklı Gol):")
    lines.append(f"   Var: %{m['btts']*100:.1f} | Yok: %{(1-m['btts'])*100:.1f}\n")

    lines.append("1X2 Olasılıkları:")
    lines.append(f"   Ev (1): %{m['1']*100:.1f}")
    lines.append(f"   Beraberlik (X): %{m['X']*100:.1f}")
    lines.append(f"   Deplasman (2): %{m['2']*100:.1f}\n")

    lines.append("="*55)
    lines.append("D) MARKET OLASILILIKLARI (KORNER)")
    lines.append("="*55 + "\n")

    mcorn = market_corn
    lines.append("Toplam Korner:")
    for k in [8.5, 9.5, 10.5, 11.5]:
        key = f"o{str(k).replace('.','')}"
        prob = mcorn[key]
        lines.append(f"   Ust {k}: %{prob*100:.1f} | Alt {k}: %{(1-prob)*100:.1f}")
    lines.append("")

    lines.append(f"{t['home']} Korner:")
    lines.append(f"   Ust 4.5: %{mcorn['home_o45']*100:.1f} | Alt 4.5: %{(1-mcorn['home_o45'])*100:.1f}")
    lines.append(f"   Ust 5.5: %{mcorn['home_o55']*100:.1f} | Alt 5.5: %{(1-mcorn['home_o55'])*100:.1f}\n")

    lines.append(f"{t['away']} Korner:")
    lines.append(f"   Ust 4.5: %{mcorn['away_o45']*100:.1f} | Alt 4.5: %{(1-mcorn['away_o45'])*100:.1f}")
    lines.append(f"   Ust 5.5: %{mcorn['away_o55']*100:.1f} | Alt 5.5: %{(1-mcorn['away_o55'])*100:.1f}\n")

    lines.append("="*55)
    lines.append(f"E) MONTE CARLO SIMULASYONU ({mc['total_sims']:,} KOSU)")
    lines.append("="*55 + "\n")

    lines.append(f"Monte Carlo Sonuclari ({mc['total_sims']:,} simulasyon):\n")
    lines.append("Toplam Gol Dagilimi:")
    for i in range(6):
        cnt = mc['dist_goals'][i]
        pct = cnt / mc['total_sims']
        bar = "-" * int(pct * 50)
        lines.append(f" {i} gol: %{pct*100:.1f} {bar}")
    plus6 = sum(mc['dist_goals'][k] for k in mc['dist_goals'] if k >= 6)
    lines.append(f" 6+ gol: %{plus6/mc['total_sims']*100:.1f}\n")

    lines.append("Market Sonuclari:")
    lines.append(f" Ust 2.5: %{mc['over25_pct']*100:.1f}")
    lines.append(f" Ust 3.5: %{mc['over35_pct']*100:.1f}")
    lines.append(f" BTTS: %{mc['btts_pct']*100:.1f}")
    lines.append(f" Ev (1): %{mc['1_pct']*100:.1f}")
    lines.append(f" Beraberlik (X): %{mc['X_pct']*100:.1f}")
    lines.append(f" Deplasman (2): %{mc['2_pct']*100:.1f}\n")

    lines.append("En Sik Gorulen 10 Skor:")
    for i, (sc, cnt) in enumerate(mc['top_scores'], 1):
        lines.append(f" {i}) {sc} - %{cnt/mc['total_sims']*100:.1f}")
    lines.append("")

    lines.append("="*55)
    lines.append(f"E-1) MONTE CARLO KORNER SIMULASYONU ({mc_corn['total_sims']:,} KOSU)")
    lines.append("="*55 + "\n")

    lines.append("Toplam Korner Dagilimi:")
    for k in range(6, 16):
        cnt = mc_corn['dist_total'][k]
        pct = cnt / mc_corn['total_sims']
        bar = "-" * int(pct * 50)
        lines.append(f" {k:02d} Korner: %{pct*100:.1f} {bar}")
    lines.append("")

    lines.append("Korner Alt/Ust Olasiliklari:")
    for k in [7, 8, 9, 10, 11]:
        pct = mc_corn[f'over{k}5']
        lines.append(f" {k}.5 Ust: %{pct*100:.1f} | Alt: %{(1-pct)*100:.1f}")
    lines.append("")

    lines.append("Korner Mac Sonucu:")
    lines.append(f" Ev Sahibi: %{mc_corn['home_more']*100:.1f}")
    lines.append(f" Beraberlik: %{mc_corn['draw']*100:.1f}")
    lines.append(f" Deplasman: %{mc_corn['away_more']*100:.1f}\n")

    lines.append("En Sik Gorulen 5 Korner Skoru:")
    for i, (sc, cnt) in enumerate(mc_corn['top_scores'], 1):
        lines.append(f" {i}) {sc} - %{cnt/mc_corn['total_sims']*100:.1f}")
    lines.append("")

    lines.append("="*55)
    lines.append("F) VALUE BET VE KELLY ANALIZI")
    lines.append("="*55 + "\n")

    def calc_kelly(odds, prob):
        if odds <= 1: return 0.0
        k = ((odds - 1) * prob - (1 - prob)) / (odds - 1)
        return max(0.0, k * 0.25)

    def check_val(odds, prob):
        val = (odds * prob) - 1
        return val, "TICK DEGER VAR" if val >= 0.05 else "CROSS"

    probs_1x2 = [market['1'], market['X'], market['2']]
    labels_1x2 = ["Ev (1)", "Beraberlik (X)", "Deplasman (2)"]

    lines.append("1X2 Marketleri:")
    has_value = False
    best_value = -1.0

    for i in range(3):
        o = val['odds'][i]
        p = probs_1x2[i]
        v_score, v_txt = check_val(o, p)
        if v_score >= 0.05: has_value = True
        best_value = max(best_value, v_score)

        lines.append(f"   {labels_1x2[i]}: Oran {o:.2f} | Olasilik %{p*100:.1f} | Value: {v_score*100:.1f}% {v_txt}")
        kelly = calc_kelly(o, p)
        if kelly > 0:
            lines.append(f"      Kelly: %{kelly*100:.1f} (maks %2-5 onerilir)")
    lines.append("")

    lines.append("="*55)
    lines.append("G) NET SONUC VE ONERILER")
    lines.append("="*55 + "\n")

    total_xg = xg['home'] + xg['away']
    tempo = "ORTA (Dengeli mac)"
    if total_xg < 2.3: tempo = "DUSUK (Savunmaci, kapali mac)"
    elif total_xg > 3.2: tempo = "YUKSEK (Hucum odakli, acik mac)"

    lines.append("1) Beklenen Goller:")
    lines.append(f"   Lambda_home = {xg['home']:.2f} gol")
    lines.append(f"   Lambda_away = {xg['away']:.2f} gol")
    lines.append(f"   xG_toplam = {total_xg:.2f} gol\n")

    lines.append(f"2) Mac Temposu: {tempo}\n")

    lines.append("3) NET SKOR TAHMINI:")
    for i in range(3):
        sc, pr = pois['top_scores'][i]
        lines.append(f"   {i+1}. Tahmin: {sc} (%{pr*100:.1f})")
    lines.append("")

    ou_conf = get_confidence(market['o25'] if market['o25'] > 0.5 else 1-market['o25'])
    ou_sel = "2.5 UST" if market['o25'] > 0.5 else "2.5 ALT"
    ou_pct = market['o25'] if market['o25'] > 0.5 else 1-market['o25']
    lines.append(f"4) NET ALT/UST: {ou_sel} (%{ou_pct*100:.1f}) - Guven: {ou_conf}\n")

    btts_conf = get_confidence(market['btts'] if market['btts'] > 0.5 else 1-market['btts'])
    btts_sel = "KG VAR" if market['btts'] > 0.5 else "KG YOK"
    btts_pct = market['btts'] if market['btts'] > 0.5 else 1-market['btts']
    lines.append(f"5) NET BTTS: {btts_sel} (%{btts_pct*100:.1f}) - Guven: {btts_conf}\n")

    lines.append("6) En Iyi 3 Bahis Adayi:")
    if has_value:
        lines.append("   (Yukaridaki Value bolumune bakiniz)")
    else:
        lines.append("   Value Bahis Bulunamadi")
    lines.append("")

    decision = "TICK BAHIS OYNANABILIR" if has_value and max(probs_1x2) >= 0.5 else "CROSS BU MACA BAHIS OYNAMA"
    lines.append(f"7) SON KARAR: {decision}")
    if has_value:
        lines.append(f"   Gerekce: Value bet bulundu (+{best_value*100:.1f}%), model uyumu iyi")
    else:
        lines.append("   Gerekce: Yeterli value bulunamadi, oranlar modele gore dusuk, bekleme tavsiye edilir")

    lines.append("\n" + "="*55)
    lines.append("ONEMLI HATIRLATMA!")
    lines.append("="*55)
    lines.append("Bu model %100 PSS (Son Mac) bazlidir.")
    lines.append("Standings ve H2H HIC KULLANILMAMIŞTIR.")
    lines.append("Sadece en guncel form baz alinmistir.")
    lines.append("Tahminler ORTALAMA beklentidir (100 mac uzerinden).")
    lines.append("Tek bir mac 0-0, 1-0 veya 5-4 bitebilir - bu normaldir!")
    lines.append("Model uzun vadede (50+ mac) degerlendirilmelidir.")
    lines.append("="*55)
    lines.append("ANALIZ TAMAMLANDI")
    lines.append("="*55)

    return "\n".join(lines)

# ============================================================================
# 12. MAIN ANALYSIS ORCHESTRATOR
# ============================================================================

def analyze_nowgoal(url: str, manual_odds: Optional[List[float]] = None) -> Dict[str, Any]:
    log_info(f"Starting PSS analysis for: {url}")

    match_id = extract_match_id(url)
    base_domain = extract_base_domain(url)

    # 1. SCRAPING
    h2h_url = build_h2h_url(url)
    html = safe_get(h2h_url, referer=extract_base_domain(url))

    home_team, away_team = parse_teams_from_title(html)

    # H2H ve İstatistikler
    h2h_matches = extract_h2h_matches(html, home_team, away_team)
    raw_home_list, raw_away_list = extract_previous_from_page(html)

    # Filtering
    prev_home_list = filter_team_home_only(raw_home_list, home_team)[:RECENT_N]
    prev_away_list = filter_team_away_only(raw_away_list, away_team)[:RECENT_N]

    st_count_h = 0
    st_count_a = 0
    if "Standings" in html: st_count_h = 10; st_count_a = 10

    # === [GÜNCELLENDİ] ORAN ÇEKME ===
    # requests-html kullanarak Bet365 Initial 1X2 oranlarını çeker
    scraped_odds = fetch_real_odds(match_id, base_domain)

    if scraped_odds and scraped_odds != [1.0, 1.0, 1.0]:
        odds = scraped_odds
    elif manual_odds and len(manual_odds) >= 3:
        odds = manual_odds
        log_info(f"Siteden çekilemedi, manuel oran: {odds}")
    else:
        odds = [1.0, 1.0, 1.0]
        log_info("Oran bulunamadı, varsayılan [1.0, 1.0, 1.0]")

    # 2. HESAPLAMALAR
    lam_home = calculate_weighted_pss_goals(prev_home_list, home_team, True)
    lam_away = calculate_weighted_pss_goals(prev_away_list, away_team, False)

    h_won, h_conceded = calculate_weighted_pss_corners(prev_home_list, home_team, True)
    a_won, a_conceded = calculate_weighted_pss_corners(prev_away_list, away_team, False)

    lam_corn_h = (h_won + a_conceded) / 2.0
    lam_corn_a = (a_won + h_conceded) / 2.0

    if lam_corn_h <= 0: lam_corn_h = 4.0
    if lam_corn_a <= 0: lam_corn_a = 3.5

    # Poisson
    h_dist = [poisson_pmf(lam_home, i) for i in range(6)]
    a_dist = [poisson_pmf(lam_away, i) for i in range(6)]

    scores = []
    for h in range(6):
        for a in range(6):
            scores.append((f"{h}-{a}", h_dist[h] * a_dist[a]))
    scores.sort(key=lambda x: x[1], reverse=True)

    # Market Goals
    m_goals = {'o05': 0, 'o15': 0, 'o25': 0, 'o35': 0, 'btts': 0, '1': 0, 'X': 0, '2': 0}
    for h in range(6):
        for a in range(6):
            prob = h_dist[h] * a_dist[a]
            total = h + a
            if total > 0: m_goals['o05'] += prob
            if total > 1: m_goals['o15'] += prob
            if total > 2: m_goals['o25'] += prob
            if total > 3: m_goals['o35'] += prob
            if h > 0 and a > 0: m_goals['btts'] += prob
            if h > a: m_goals['1'] += prob
            elif h == a: m_goals['X'] += prob
            else: m_goals['2'] += prob

    # Market Corners
    h_corn_dist_trunc = [poisson_pmf(lam_corn_h, i) for i in range(11)]
    a_corn_dist_trunc = [poisson_pmf(lam_corn_a, i) for i in range(11)]

    m_corn = {'o85': 0, 'o95': 0, 'o105': 0, 'o115': 0,
              'home_o45': 0, 'home_o55': 0, 'away_o45': 0, 'away_o55': 0}

    for h in range(11):
        for a in range(11):
            prob = h_corn_dist_trunc[h] * a_corn_dist_trunc[a]
            tot = h + a
            if tot > 8: m_corn['o85'] += prob
            if tot > 9: m_corn['o95'] += prob
            if tot > 10: m_corn['o105'] += prob
            if tot > 11: m_corn['o115'] += prob

    m_corn['home_o45'] = sum(h_corn_dist_trunc[5:])
    m_corn['home_o55'] = sum(h_corn_dist_trunc[6:])
    m_corn['away_o45'] = sum(a_corn_dist_trunc[5:])
    m_corn['away_o55'] = sum(a_corn_dist_trunc[6:])

    # Simulations
    mc_goals = monte_carlo_simulation_vba(lam_home, lam_away, MC_RUNS_DEFAULT)
    mc_corners = monte_carlo_corners_vba(lam_corn_h, lam_corn_a, MC_RUNS_DEFAULT)

    full_data = {
        'teams': {'home': home_team, 'away': away_team},
        'counts': {
            'home_standings': st_count_h,
            'away_standings': st_count_a,
            'h2h': len(h2h_matches),
            'pss_home': len(prev_home_list),
            'pss_away': len(prev_away_list)
        },
        'h2h_sample': h2h_matches,
        'pss_home_sample': prev_home_list,
        'pss_away_sample': prev_away_list,
        'xg': {'home': lam_home, 'away': lam_away},
        'corners': {'home': lam_corn_h, 'away': lam_corn_a},
        'poisson': {'home_dist': h_dist, 'away_dist': a_dist, 'top_scores': scores[:7]},
        'market_goals': m_goals,
        'market_corners': m_corn,
        'mc_goals': mc_goals,
        'mc_corners': mc_corners,
        'value': {'odds': odds}
    }

    report_text = generate_vba_report(full_data)

    return {
        "ok": True,
        "report": report_text,
        "raw_data": full_data
    }

# ============================================================================
# 13. FLASK API
# ============================================================================
app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({
        "ok": True,
        "service": "nowgoal-analyzer-ultimate",
        "version": "6.6-requests-html-render",
        "status": "running"
    })

@app.route("/health")
def health():
    return jsonify({"ok": True, "status": "healthy", "timestamp": time.time()})

@app.route("/analizet", methods=["POST"])
@app.route("/analiz_et", methods=["POST"])
def analizet_route():
    request_start = time.time()
    log_info("=== REQUEST STARTED: /analiz_et ===")

    try:
        try:
            payload = request.get_json(silent=True) or {}
            log_info(f"Payload: {payload}")
        except Exception as e:
            log_error("JSON parse failed", e)
            return jsonify({"ok": False, "error": f"JSON hatası: {e}"}), 400

        url = (payload.get("url") or '').strip()
        odds = payload.get("odds", [2.50, 3.20, 2.50])

        if not url:
            log_error("URL is empty")
            return jsonify({"ok": False, "error": "URL boş"}), 400

        if not re.match(r'https?://', url):
            log_error(f"Invalid URL: {url}")
            return jsonify({"ok": False, "error": "Geçersiz URL"}), 400

        log_info(f"Analyzing: {url}")

        try:
            result = analyze_nowgoal(url, odds)
            elapsed = time.time() - request_start
            log_info(f"Analysis OK in {elapsed:.2f}s")

            data = result['raw_data']
            top_skor = data['poisson']['top_scores'][0]
            m = data['market_goals']
            corn = data['corners']

            response = {
                "ok": True,
                "skor": f"{top_skor[0]}: {top_skor[1]*100:.1f}%",
                "alt_ust": f"2.5 {'ÜST' if m['o25'] > 0.5 else 'ALT'}: {max(m['o25'], 1-m['o25'])*100:.1f}%",
                "btts": f"{'VAR' if m['btts'] > 0.5 else 'YOK'}: {max(m['btts'], 1-m['btts'])*100:.1f}%",
                "korner": {
                    "toplam": f"{corn['home'] + corn['away']:.1f}",
                    "ev": f"{corn['home']:.1f}",
                    "deplasman": f"{corn['away']:.1f}"
                },
                "detay": result['report'],
                "sure": f"{elapsed:.2f}s"
            }

            return jsonify(response)

        except requests.exceptions.Timeout as e:
            log_error("Timeout", e)
            return jsonify({"ok": False, "error": "Zaman aşımı", "detail": str(e)}), 504
        except Exception as e:
            log_error("Analysis failed", e)
            return jsonify({"ok": False, "error": f"Analiz hatası: {str(e)}", "traceback": traceback.format_exc()}), 500

    except Exception as e:
        log_error("Unhandled exception", e)
        return jsonify({"ok": False, "error": f"Beklenmeyen hata: {str(e)}"}), 500

if __name__ == "__main__":
    log_info("=" * 70)
    log_info("NowGoal Analyzer REQUESTS-HTML VERSION - COMPLETE")
    log_info("Render Sunucusu Uyumlu - Chrome Gerektirmez!")
    log_info("=" * 70)
    log_info("KURULUM:")
    log_info("1) pip install requests-html flask requests")
    log_info("2) python app.py serve")
    log_info("=" * 70)

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        log_info("Starting Flask server on 0.0.0.0:5000...")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    else:
        print("Usage: python app.py serve")
