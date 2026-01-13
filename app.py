# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - Enhanced Version 4.3 (FINAL)
Flask API with Corner Analysis & Enhanced Value Betting

FIX PACK (v4.3 - FINAL):
1) PSS Home/Away ve Same League filtresi düzeltildi:
   - ÖNCE Home/Away ayırımı yapılır
   - SONRA Same League filtresi uygulanır (min 3 maç)
2) H2H için Same League filtresi KALDIRILDI
3) PSS minimum 3 maç kontrolü eklendi
4) H2H minimum 3 maç kontrolü eklendi
5) Standing ZORUNLU (yoksa ValueError)
6) Ağırlıklar güncellendi: Standing 50%, PSS 30%, H2H 20%
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

# ======================
# CONFIG
# ======================
MC_RUNS_DEFAULT = 10_000
RECENT_N = 10
H2H_N = 10

# AĞIRLIKLAR - Standing güçlendirildi
W_ST_BASE = 0.50   # 0.45 → 0.50
W_PSS_BASE = 0.30  # Aynı
W_H2H_BASE = 0.20  # 0.25 → 0.20

BLEND_ALPHA = 0.50
VALUE_MIN = 0.05
PROB_MIN = 0.55
KELLY_MIN = 0.02
MAX_GOALS_FOR_MATRIX = 5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# ======================
# REGEX
# ======================
DATE_ANY_RE = re.compile(r"\b(\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{2}-\d{2})\b")
SCORE_RE = re.compile(
    r"\b(\d{1,2})\s*-\s*(\d{1,2})(?!-\d{2,4})(?:\s*\((\d{1,2})\s*-\s*(\d{1,2})\))?\b"
)

CORNER_FT_RE = re.compile(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b")
CORNER_HT_RE = re.compile(r"\((\d{1,2})\s*-\s*(\d{1,2})\)")

FLOAT_RE = re.compile(r"(?<!\d)(\d+\.\d+)(?!\d)")
INT_RE = re.compile(r"(?<!\d)(\d+)(?!\d)")

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
    val = m.group(1)
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
            if c in {"—", "-"}:
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
def safe_get(url: str, timeout: int = 25, retries: int = 2, referer: Optional[str] = None) -> str:
    last_err = None
    headers = dict(HEADERS)
    if referer:
        headers["Referer"] = referer

    for _ in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            return r.text
        except Exception as e:
            last_err = e
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
# CORNER PARSE
# ======================
def parse_corner_cell(cell: str) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    if not cell:
        return None, None
    txt = (cell or "").strip()
    if txt in {"", "-", "—"}:
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
            league=league,
            date=date_val,
            home=home,
            away=away,
            ft_home=ft_h,
            ft_away=ft_a,
            ht_home=ht_h,
            ht_away=ht_a,
            corner_home=corner_home,
            corner_away=corner_away,
            corner_ht_home=corner_ht_home,
            corner_ht_away=corner_ht_away,
        )

    # fallback
    score_idx = None
    score_m = None
    for i, c in enumerate(cells):
        c0 = (c or "").strip()
        m = SCORE_RE.search(c0)
        if m:
            score_idx = i
            score_m = m
            break
    if not score_m or score_idx is None:
        return None

    ft_h = int(score_m.group(1))
    ft_a = int(score_m.group(2))
    ht_h = int(score_m.group(3)) if score_m.group(3) else None
    ht_a = int(score_m.group(4)) if score_m.group(4) else None

    home2 = None
    away2 = None

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

    league2 = (cells[0] or "").strip() or "—"

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
        league=league2,
        date=date_val2,
        home=home2,
        away=away2,
        ft_home=ft_h,
        ft_away=ft_a,
        ht_home=ht_h,
        ht_away=ht_a,
        corner_home=corner_home,
        corner_away=corner_away,
        corner_ht_home=corner_ht_home,
        corner_ht_away=corner_ht_away,
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
        if x in {"", "-", "—"}:
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
# ODDS
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

def _extract_all_numbers_loose(s: str) -> List[float]:
    if not s:
        return []
    nums = []
    for m in re.finditer(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", s):
        try:
            nums.append(float(m.group(1)))
        except Exception:
            pass
    return nums

def _extract_cell_numeric_from_inner_html(inner_html: str) -> Optional[float]:
    if not inner_html:
        return None
    txt = strip_tags_keep_text(inner_html)
    v = _extract_first_float(txt)
    if v is not None:
        return v

    m = re.search(r'(?:title|data-[a-z0-9_-]+)\s*=\s*"(\d+\.\d+)"', inner_html, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None

    v2 = _extract_first_float(inner_html)
    return v2

def extract_bet365_initial_1x2_from_oddscomp_html(odds_html: str) -> Optional[Dict[str, float]]:
    if not odds_html:
        return None

    tr_m = re.search(r"(<tr\b[^>]*>.*?Bet365.*?</tr>)", odds_html, flags=re.I | re.S)
    if not tr_m:
        return None

    tr_html = tr_m.group(1)
    tds = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr_html, flags=re.I | re.S)
    if not tds or len(tds) < 8:
        nums = _extract_all_numbers_loose(strip_tags_keep_text(tr_html))
        if len(nums) >= 6:
            cand = nums[3:6]
            if all(1.01 <= x <= 50 for x in cand):
                return {"1": float(cand[0]), "X": float(cand[1]), "2": float(cand[2])}
        return None

    cell_vals: List[Optional[float]] = [_extract_cell_numeric_from_inner_html(td) for td in tds]

    if len(cell_vals) >= 8:
        o1, ox, o2 = cell_vals[5], cell_vals[6], cell_vals[7]
        if all(v is not None for v in [o1, ox, o2]):
            if all(1.01 <= float(v) <= 200 for v in [o1, ox, o2]):
                return {"1": float(o1), "X": float(ox), "2": float(o2)}

    nums = _extract_all_numbers_loose(strip_tags_keep_text(tr_html))
    if len(nums) >= 6:
        cand = nums[3:6]
        if all(1.01 <= x <= 50 for x in cand):
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

def filter_same_league_matches(matches: List[MatchRow], league_name: str, min_matches: int = 3) -> List[MatchRow]:
    """Same League filtresi - minimum maç kontrolü ile"""
    if not league_name or not matches:
        return matches

    lk = norm_key(league_name)
    if not lk:
        return matches

    filtered = []
    for m in matches:
        ml = norm_key(m.league)
        if ml and (lk in ml or ml in lk or ml == lk):
            filtered.append(m)

    # En az min_matches varsa filtreli döndür
    if len(filtered) >= min_matches:
        return filtered

    # Yoksa orijinal döndür
    return matches

def filter_team_home_only(matches: List[MatchRow], team: str) -> List[MatchRow]:
    """Takımın sadece EV SAHİBİ olduğu maçları filtreler"""
    tk = norm_key(team)
    return [m for m in matches if norm_key(m.home) == tk]

def filter_team_away_only(matches: List[MatchRow], team: str) -> List[MatchRow]:
    """Takımın sadece DEPLASMAN olduğu maçları filtreler"""
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

    gfs, gas, corners_for, corners_against = [], [], [], []
    clean_sheets = 0
    scored_matches = 0

    for m in matches:
        gf, ga, cf, ca = team_gf_ga(m)
        gfs.append(gf)
        gas.append(ga)
        if cf is not None:
            corners_for.append(cf)
        if ca is not None:
            corners_against.append(ca)
        if ga == 0:
            clean_sheets += 1
        if gf > 0:
            scored_matches += 1

    st.n_total = len(matches)
    st.gf_total = sum(gfs) / st.n_total if st.n_total else 0.0
    st.ga_total = sum(gas) / st.n_total if st.n_total else 0.0
    st.clean_sheets = clean_sheets
    st.scored_matches = scored_matches
    st.corners_for = sum(corners_for) / len(corners_for) if corners_for else 0.0
    st.corners_against = sum(corners_against) / len(corners_against) if corners_against else 0.0

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
# POISSON HELPERS
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
# CORNER ANALYSIS
# ======================
def analyze_corners(home_prev: TeamPrevStats, away_prev: TeamPrevStats, h2h_matches: List[MatchRow]) -> Dict[str, Any]:
    h2h_total = []
    h2h_home = []
    h2h_away = []

    for m in h2h_matches[:H2H_N]:
        if m.corner_home is not None and m.corner_away is not None:
            h2h_total.append(m.corner_home + m.corner_away)
            h2h_home.append(m.corner_home)
            h2h_away.append(m.corner_away)

    h2h_total_avg = sum(h2h_total) / len(h2h_total) if h2h_total else 0.0
    h2h_home_avg = sum(h2h_home) / len(h2h_home) if h2h_home else 0.0
    h2h_away_avg = sum(h2h_away) / len(h2h_away) if h2h_away else 0.0

    pss_home_for = home_prev.corners_for
    pss_home_against = home_prev.corners_against
    pss_away_for = away_prev.corners_for
    pss_away_against = away_prev.corners_against

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

    predictions = {}
    for line in [8.5, 9.5, 10.5, 11.5]:
        k = int(math.floor(line))
        p_under_or_eq = poisson_cdf(k, total_corners)
        p_over = 1.0 - p_under_or_eq
        predictions[f"O{line}"] = float(max(0.0, min(1.0, p_over)))
        predictions[f"U{line}"] = float(1.0 - predictions[f"O{line}"])

    data_points = len(h2h_total) + (1 if (pss_home_for > 0 or pss_away_for > 0) else 0)
    if data_points >= 8:
        confidence = "Yüksek"
    elif data_points >= 4:
        confidence = "Orta"
    else:
        confidence = "Düşük"

    return {
        "predicted_home_corners": round(predicted_home, 1),
        "predicted_away_corners": round(predicted_away, 1),
        "total_corners": round(total_corners, 1),
        "h2h_avg": round(h2h_total_avg, 1),
        "h2h_data_count": len(h2h_total),
        "pss_data_available": bool(pss_home_for > 0 or pss_away_for > 0),
        "predictions": predictions,
        "confidence": confidence
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
        "home_split": {"matches": hh.matches, "gf_pg": hh.gf_pg, "ga_pg": hh.ga_pg},
        "away_split": {"matches": aa.matches, "gf_pg": aa.gf_pg, "ga_pg": aa.ga_pg},
        "formula": "Standing-based lambda",
    }
    return lam_h, lam_a, meta

def compute_component_pss(home_prev: TeamPrevStats, away_prev: TeamPrevStats) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """PSS bileşeni - EN AZ 3 MAÇ GEREKLİ"""

    # Minimum 3 maç kontrolü
    if home_prev.n_total < 3 or away_prev.n_total < 3:
        return None

    h_gf = home_prev.gf_total
    h_ga = home_prev.ga_total
    a_gf = away_prev.gf_total
    a_ga = away_prev.ga_total

    lam_h = (h_gf + a_ga) / 2.0
    lam_a = (a_gf + h_ga) / 2.0

    meta = {
        "home_matches": home_prev.n_total,
        "away_matches": away_prev.n_total,
        "home_gf": round(h_gf, 2),
        "away_gf": round(a_gf, 2),
        "formula": "PSS (filtered): (home_gf + away_ga) / 2"
    }
    return lam_h, lam_a, meta

def compute_component_h2h(h2h_matches: List[MatchRow], home_team: str, away_team: str) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """H2H bileşeni - EN AZ 3 MAÇ GEREKLİ"""

    # Minimum 3 maç kontrolü
    if not h2h_matches or len(h2h_matches) < 3:
        return None

    hk = norm_key(home_team)
    ak = norm_key(away_team)
    used = h2h_matches[:H2H_N]
    hg, ag = [], []
    for m in used:
        if norm_key(m.home) == hk and norm_key(m.away) == ak:
            hg.append(m.ft_home); ag.append(m.ft_away)
        elif norm_key(m.home) == ak and norm_key(m.away) == hk:
            hg.append(m.ft_away); ag.append(m.ft_home)

    # Eşleşen maç sayısı kontrolü
    if len(hg) < 3:
        return None

    lam_h = sum(hg) / len(hg)
    lam_a = sum(ag) / len(ag)
    meta = {"matches": len(hg), "hg_avg": lam_h, "ag_avg": lam_a}
    return lam_h, lam_a, meta

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
    return c(lh, "λ_home"), c(la, "λ_away"), warn

def compute_lambdas(st_home_s: Dict[str, Optional[SplitGFGA]],
                    st_away_s: Dict[str, Optional[SplitGFGA]],
                    home_prev: TeamPrevStats,
                    away_prev: TeamPrevStats,
                    h2h_used: List[MatchRow],
                    home_team: str,
                    away_team: str) -> Tuple[float, float, Dict[str, Any]]:
    """
    Lambda hesaplama - STANDING ZORUNLU!
    Standing yoksa ValueError fırlatır.
    """
    info: Dict[str, Any] = {"components": {}, "weights_used": {}, "warnings": []}
    comps: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}

    # ===== STANDING - ZORUNLU KONTROL =====
    stc = compute_component_standings(st_home_s, st_away_s)
    if not stc:
        raise ValueError(
            "❌ ANALİZ YAPILAMIYOR\n\n"
            f"Sebep: Standing (Puan Durumu) verisi bulunamadı.\n"
            f"Takımlar: {home_team} vs {away_team}\n\n"
            "Her iki takım için de en az 3 maçlık puan durumu verisi gereklidir.\n"
            "Lütfen maç sayfasını kontrol edin veya başka bir maç deneyin."
        )

    comps["standing"] = stc
    info["standing_mandatory"] = True

    # ===== PSS - İSTEĞE BAĞLI (min 3 maç) =====
    pss = compute_component_pss(home_prev, away_prev)
    if pss:
        comps["pss"] = pss
    else:
        info["warnings"].append(
            f"PSS atlandı: Home={home_prev.n_total} maç, Away={away_prev.n_total} maç (minimum 3 gerekli)"
        )

    # ===== H2H - İSTEĞE BAĞLI (min 3 maç) =====
    h2c = compute_component_h2h(h2h_used, home_team, away_team)
    if h2c:
        comps["h2h"] = h2c
    else:
        info["warnings"].append(
            f"H2H atlandı: {len(h2h_used)} maç (minimum 3 gerekli)"
        )

    # ===== AĞIRLIK HESAPLAMA =====
    w = {}
    if "standing" in comps:
        w["standing"] = W_ST_BASE  # 0.50
    if "pss" in comps:
        w["pss"] = W_PSS_BASE      # 0.30
    if "h2h" in comps:
        w["h2h"] = W_H2H_BASE      # 0.20

    w_norm = normalize_weights(w)
    info["weights_used"] = w_norm

    # ===== LAMBDA HESAPLAMA =====
    lh = 0.0
    la = 0.0
    for k, wk in w_norm.items():
        ch, ca, meta = comps[k]
        info["components"][k] = {"lam_home": ch, "lam_away": ca, "meta": meta}
        lh += wk * ch
        la += wk * ca

    lh, la, clamp_warn = clamp_lambda(lh, la)
    if clamp_warn:
        info["warnings"].extend(clamp_warn)
    return lh, la, info

# ======================
# POISSON & MC
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

    out = {"1": p1, "X": px, "2": p2, "BTTS": btts}

    for ln in [0.5, 1.5, 2.5, 3.5]:
        need = int(math.floor(ln) + 1)
        out[f"O{ln}"] = sum(p for (h, a), p in mat.items() if (h + a) >= need)
        out[f"U{ln}"] = 1.0 - out[f"O{ln}"]

    return out

def monte_carlo(lh: float, la: float, n: int, seed: Optional[int] = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    hg = rng.poisson(lh, size=n)
    ag = rng.poisson(la, size=n)
    total = hg + ag

    def p(mask) -> float:
        return float(np.mean(mask))

    cnt = Counter(zip(hg.tolist(), ag.tolist()))
    top10 = cnt.most_common(10)
    top10_list = [(f"{h}-{a}", c / n * 100.0) for (h, a), c in top10]

    dist_total = Counter(total.tolist())
    total_bins: Dict[str, float] = {}
    for k in range(0, 5):
        total_bins[str(k)] = dist_total.get(k, 0) / n * 100.0
    total_bins["5+"] = sum(v for kk, v in dist_total.items() if kk >= 5) / n * 100.0

    out = {
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
    return out

def model_agreement(p_po: Dict[str, float], p_mc: Dict[str, float]) -> Tuple[float, str]:
    keys = ["1", "X", "2", "BTTS", "O2.5", "U2.5"]
    diffs = [abs(p_po.get(k, 0) - p_mc.get(k, 0)) for k in keys]
    d = max(diffs) if diffs else 0.0
    if d <= 0.03:
        return d, "Mükemmel"
    elif d <= 0.06:
        return d, "İyi"
    elif d <= 0.10:
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

# ======================
# KELLY & VALUE
# ======================
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

# ======================
# REPORTING
# ======================
def top_scores_from_matrix(mat: Dict[Tuple[int, int], float], top_n: int = 7) -> List[Tuple[str, float]]:
    items = sorted(mat.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(f"{h}-{a}", p) for (h, a), p in items]

def net_ou_prediction(probs: Dict[str, float]) -> Tuple[str, float, str]:
    p_o25 = probs.get("O2.5", 0)
    p_u25 = probs.get("U2.5", 0)
    if p_o25 >= p_u25:
        return "2.5 ÜST", p_o25, confidence_label(p_o25)
    return "2.5 ALT", p_u25, confidence_label(p_u25)

def net_btts_prediction(probs: Dict[str, float]) -> Tuple[str, float, str]:
    p_btts = probs.get("BTTS", 0)
    p_no = 1.0 - p_btts
    if p_btts >= p_no:
        return "VAR", p_btts, confidence_label(p_btts)
    return "YOK", p_no, confidence_label(p_no)

def final_decision(qualified: List[Tuple[str, float, float, float, float]], diff: float, diff_label: str) -> str:
    if not qualified:
        return f"OYNAMA (Eşik sağlanmadı, model uyumu: {diff_label})"
    if diff > 0.10:
        return f"TEMKİNLİ (Zayıf model uyumu: {diff_label})"
    best = sorted(qualified, key=lambda x: x[3], reverse=True)[0]
    mkt, prob, odds, val, qk = best
    return f"OYNANABİLİR → {mkt} (Prob: %{prob*100:.1f}, Oran: {odds:.2f}, Value: %{val*100:+.1f}, Kelly: %{qk*100:.1f})"

def format_comprehensive_report(data: Dict[str, Any]) -> str:
    t = data["teams"]
    blend = data["blended_probs"]
    top7 = data["poisson"]["top7_scores"]

    lines = []
    lines.append("=" * 60)
    lines.append(f"  {t['home']} vs {t['away']}")
    lines.append("=" * 60)

    lines.append(f"\nOLASI SKORLAR:")
    for i, (score, prob) in enumerate(top7[:5], 1):
        bar = "█" * int(prob * 50)
        lines.append(f"  {i}. {score:6s} %{prob*100:4.1f}  {bar}")

    lines.append(f"\nNET TAHMİN:")
    lines.append(f"  Ana Skor: {top7[0][0]}")
    if len(top7) >= 3:
        lines.append(f"  Alt Skor: {top7[1][0]}, {top7[2][0]}")

    net_ou, net_ou_p, _ = net_ou_prediction(blend)
    lines.append(f"\nAlt/Üst 2.5: {net_ou} (%{net_ou_p*100:.1f})")

    net_btts, net_btts_p, _ = net_btts_prediction(blend)
    lines.append(f"KG Var: {net_btts} (%{net_btts_p*100:.1f})")

    lines.append(f"\n1X2 Olasılıkları:")
    lines.append(f"  Ev (1): %{blend.get('1', 0)*100:.1f}")
    lines.append(f"  Ber(X): %{blend.get('X', 0)*100:.1f}")
    lines.append(f"  Dep(2): %{blend.get('2', 0)*100:.1f}")

    corners = data.get("corner_analysis", {})
    if corners and corners.get("total_corners", 0) > 0:
        lines.append(f"\nKorner Tahmini (Toplam λ): {corners['total_corners']}")
        lines.append(f"  (Ev: {corners['predicted_home_corners']} | Dep: {corners['predicted_away_corners']})")
        preds = corners.get("predictions", {})
        for k in ["O8.5","O9.5","O10.5","O11.5"]:
            lines.append(f"  {k}: %{preds.get(k,0)*100:.1f}  |  U{k[1:]}: %{preds.get('U'+k[1:],0)*100:.1f}")

    vb = data.get("value_bets", {})
    if vb.get("used_odds"):
        lines.append(f"\nBAHİS ANALİZİ (Bet365 Initial 1X2):")
        has_value = False
        for row in vb.get("table", []):
            if row["value"] >= VALUE_MIN and row["prob"] >= PROB_MIN:
                lines.append(f"  ✅ {row['market']}: Oran {row['odds']:.2f} | Value %{row['value']*100:+.1f} | Kelly %{row['kelly']*100:.1f}")
                has_value = True
        if not has_value:
            lines.append("  ⚠️  Değerli bahis bulunamadı")
        lines.append(f"\nKARAR: {vb.get('decision', 'Analiz edilemedi')}")
    else:
        lines.append("\nOran verisi yok - value analizi yapılamadı")

    ds = data["data_sources"]
    lambda_info = data["lambda"]["info"]

    lines.append(f"\nKullanılan Veriler:")
    lines.append(f"  Standing: ✓ (ZORUNLU)")
    lines.append(f"  PSS (Same League + Home/Away): {'✓' if ds.get('pss_sufficient') else '✗'} (Ev:{ds['home_prev_matches']} | Dep:{ds['away_prev_matches']})")
    lines.append(f"  H2H: {'✓' if ds.get('h2h_sufficient') else '✗'} ({ds['h2h_matches']} maç)")

    if lambda_info.get("weights_used"):
        lines.append(f"\nAğırlıklar:")
        for k, v in lambda_info["weights_used"].items():
            k_name = {"standing": "Standing", "pss": "PSS", "h2h": "H2H"}.get(k, k)
            lines.append(f"  {k_name}: %{v*100:.0f}")

    if lambda_info.get("warnings"):
        lines.append(f"\nUyarılar:")
        for w in lambda_info["warnings"]:
            lines.append(f"  ⚠️ {w}")

    lines.append("=" * 60)
    return "\n".join(lines)

# ======================
# MAIN ANALYSIS
# ======================
def analyze_nowgoal(url: str, odds: Optional[Dict[str, float]] = None, mc_runs: int = MC_RUNS_DEFAULT) -> Dict[str, Any]:
    """
    FINAL VERSION - Tüm düzeltmeler uygulanmış
    """
    h2h_url = build_h2h_url(url)
    html = safe_get(h2h_url, referer=extract_base_domain(url))

    home_team, away_team = parse_teams_from_title(html)
    if not home_team or not away_team:
        raise RuntimeError("Takım isimleri çıkarılamadı")

    # Lig ismini çıkar
    league_match = re.search(r'<span\s+class="?sclassLink"?>.*?</span>', html)
    league_name = strip_tags_keep_text(league_match.group(1)) if league_match else ""

    # Standings
    st_home_rows = extract_standings_for_team(html, home_team)
    st_away_rows = extract_standings_for_team(html, away_team)
    st_home_splits = standings_to_splits(st_home_rows)
    st_away_splits = standings_to_splits(st_away_rows)

    # PSS - DÜZELTİLMİŞ FİLTRELEME
    prev_home_tabs, prev_away_tabs = extract_previous_from_page(html)

    # HOME TAKIMI
    prev_home_sel = []
    if prev_home_tabs:
        prev_home_raw = parse_matches_from_table_html(prev_home_tabs[0])
        prev_home_home_only = filter_team_home_only(prev_home_raw, home_team)
        if league_name:
            prev_home_filtered = filter_same_league_matches(prev_home_home_only, league_name, min_matches=3)
        else:
            prev_home_filtered = prev_home_home_only
        prev_home_sel = prev_home_filtered[:RECENT_N]

    # AWAY TAKIMI
    prev_away_sel = []
    if prev_away_tabs:
        prev_away_raw = parse_matches_from_table_html(prev_away_tabs[0])
        prev_away_away_only = filter_team_away_only(prev_away_raw, away_team)
        if league_name:
            prev_away_filtered = filter_same_league_matches(prev_away_away_only, league_name, min_matches=3)
        else:
            prev_away_filtered = prev_away_away_only
        prev_away_sel = prev_away_filtered[:RECENT_N]

    # H2H - SAME LEAGUE FİLTRESİZ
    h2h_all = extract_h2h_matches(html, home_team, away_team)
    h2h_used = h2h_all[:H2H_N]

    # Prev Stats
    home_prev = build_prev_stats(home_team, prev_home_sel)
    away_prev = build_prev_stats(away_team, prev_away_sel)

    # Lambda hesaplama (Standing ZORUNLU)
    lam_home, lam_away, lambda_info = compute_lambdas(
        st_home_splits,
        st_away_splits,
        home_prev,
        away_prev,
        h2h_used,
        home_team,
        away_team
    )

    # Poisson & Monte Carlo
    score_matrix = build_score_matrix(lam_home, lam_away, MAX_GOALS_FOR_MATRIX)
    poisson_probs = market_probs_from_matrix(score_matrix)
    mc_result = monte_carlo(lam_home, lam_away, mc_runs)

    diff, diff_label = model_agreement(poisson_probs, mc_result["p"])
    blended_probs = blend_probs(poisson_probs, mc_result["p"], BLEND_ALPHA)

    # Corner Analysis
    corner_analysis = analyze_corners(home_prev, away_prev, h2h_used)

    # Odds & Value Betting
    used_odds = odds or extract_bet365_initial_odds(url)
    value_bets = {}
    if used_odds:
        vb_table = []
        for mkt in ["1", "X", "2"]:
            if mkt in used_odds and mkt in blended_probs:
                val, kelly = value_and_kelly(blended_probs[mkt], used_odds[mkt])
                vb_table.append({
                    "market": mkt,
                    "prob": blended_probs[mkt],
                    "odds": used_odds[mkt],
                    "value": val,
                    "kelly": kelly
                })

        qualified = [
            (r["market"], r["prob"], r["odds"], r["value"], r["kelly"])
            for r in vb_table
            if r["value"] >= VALUE_MIN and r["prob"] >= PROB_MIN and r["kelly"] >= KELLY_MIN
        ]

        decision = final_decision(qualified, diff, diff_label)
        value_bets = {
            "used_odds": used_odds,
            "table": vb_table,
            "qualified": qualified,
            "decision": decision
        }

    # Result
    top7 = top_scores_from_matrix(score_matrix, top_n=7)

    result = {
        "teams": {"home": home_team, "away": away_team},
        "league": league_name,
        "lambda": {
            "home": round(lam_home, 2),
            "away": round(lam_away, 2),
            "info": lambda_info
        },
        "poisson": {
            "probs": {k: round(v, 4) for k, v in poisson_probs.items()},
            "top7_scores": [(score, round(prob, 4)) for score, prob in top7]
        },
        "monte_carlo": {
            "runs": mc_runs,
            "probs": {k: round(v, 4) for k, v in mc_result["p"].items()},
            "total_dist": {k: round(v, 2) for k, v in mc_result["TOTAL_DIST"].items()},
            "top10": [(score, round(pct, 2)) for score, pct in mc_result["TOP10"]]
        },
        "model_agreement": {"max_diff": round(diff, 4), "label": diff_label},
        "blended_probs": {k: round(v, 4) for k, v in blended_probs.items()},
        "corner_analysis": corner_analysis,
        "value_bets": value_bets,
        "data_sources": {
            "standings_used": True,  # Standing zorunlu
            "standings_mandatory": True,
            "home_prev_matches": len(prev_home_sel),
            "away_prev_matches": len(prev_away_sel),
            "h2h_matches": len(h2h_used),
            "pss_sufficient": home_prev.n_total >= 3 and away_prev.n_total >= 3,
            "h2h_sufficient": len(h2h_used) >= 3,
            "pss_same_league_used": bool(league_name),
            "h2h_same_league_used": False,
            "components_used": list(lambda_info.get("weights_used", {}).keys())
        }
    }

    result["report_comprehensive"] = format_comprehensive_report(result)

    return result

# ======================
# FLASK APP
# ======================
app = Flask(__name__)

@app.get("/")
def root():
    return jsonify(ok=True, service="nowgoal-analyzer-api", version="4.3-final")

@app.get("/health")
def health():
    return jsonify(ok=True, status="healthy")

@app.post("/analizet")
def analizet_route():
    """Turkish endpoint for Android app"""
    try:
        payload = request.get_json(silent=True) or {}
    except Exception as e:
        return jsonify(ok=False, error=f"Geçersiz JSON: {e}"), 400

    url = (payload.get("url") or "").strip()
    if not url:
        return jsonify(ok=False, error="URL boş olamaz"), 400

    if not re.match(r"^https?://", url):
        return jsonify(ok=False, error="Geçersiz URL formatı"), 400

    try:
        data = analyze_nowgoal(url, odds=None, mc_runs=10000)

        top_skor = data["poisson"]["top7_scores"][0][0]
        blend = data["blended_probs"]
        net_ou, net_ou_p, _ = net_ou_prediction(blend)
        net_btts, net_btts_p, _ = net_btts_prediction(blend)

        return jsonify(
            ok=True,
            skor=top_skor,
            altust=f"{net_ou} (%{net_ou_p*100:.1f})",
            btts=f"{net_btts} (%{net_btts_p*100:.1f})",
            karar=data["value_bets"].get("decision", "Oran gerekli"),
            odds_used=data["value_bets"].get("used_odds", False),
            odds=data.get("value_bets", {}).get("table", None),
            detay=data["report_comprehensive"]
        )

    except ValueError as ve:
        return jsonify(
            ok=False, 
            error=str(ve),
            error_type="MISSING_STANDINGS"
        ), 400

    except Exception as e:
        return jsonify(
            ok=False, 
            error=f"Analiz hatası: {str(e)}",
            traceback=traceback.format_exc()
        ), 500

@app.post("/analyze")
def analyze_route():
    """Full English API endpoint"""
    try:
        payload = request.get_json(silent=True) or {}
    except Exception as e:
        return jsonify(ok=False, error=f"Invalid JSON: {e}"), 400

    url = (payload.get("url") or "").strip()
    if not url:
        return jsonify(ok=False, error="URL required"), 400

    if not re.match(r"^https?://", url):
        return jsonify(ok=False, error="Invalid URL"), 400

    odds = payload.get("odds")
    mc_runs = payload.get("mc_runs", MC_RUNS_DEFAULT)
    try:
        mc_runs = int(mc_runs)
        if mc_runs < 100 or mc_runs > 100000:
            mc_runs = MC_RUNS_DEFAULT
    except (ValueError, TypeError):
        mc_runs = MC_RUNS_DEFAULT

    try:
        data = analyze_nowgoal(url, odds=odds, mc_runs=mc_runs)
        return jsonify(ok=True, data=data)

    except ValueError as ve:
        return jsonify(
            ok=False, 
            error=str(ve),
            error_type="MISSING_STANDINGS"
        ), 400

    except Exception as e:
        return jsonify(
            ok=False, 
            error=f"Analysis error: {str(e)}",
            traceback=traceback.format_exc()
        ), 500

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("NowGoal Analyzer v4.3 FINAL")
        print("Usage: python script.py serve")
        print("Endpoints:")
        print("  POST /analizet - Android app (Turkish)")
        print("  POST /analyze - Full API (English)")
        print("Example: url='https://live3.nowgoal26.com/match/h2h-2784678'")
