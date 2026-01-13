# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - Enhanced Version 5.0 FINAL
Flask API with Complete Data Sources & Confidence Indicators

VERİ KAYNAKLARI:
✅ Standing (Lig Tablosu): Home/Away/Total gol ortalamaları
✅ PSS (Önceki Maçlar): Same League + Home-only/Away-only filtreleme
✅ H2H (Karşılıklı Maçlar): Direct matchup geçmişi + korner verileri
✅ Corner Analysis: H2H + PSS korner ortalamaları

GÜVEN GÖSTERGELERİ:
🟢 ✅ Yeşil: Yüksek güven (%65+) - Oynanabilir
🟡 ⚠️  Sarı: Orta güven (%55-65) - Dikkatli
🔴 ❌ Kırmızı: Düşük güven (%55 altı) - Oynama

TAHMİN PAZARLARI:
- 1X2 (Maç sonucu)
- 2.5 Üst/Alt
- 3.5 Üst/Alt
- KG Var/Yok (BTTS)
- Korner: 8.5, 9.5, 10.5, 11.5 üst/alt
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

# Lambda ağırlıkları
W_ST_BASE = 0.45   # Standing weight
W_PSS_BASE = 0.30  # PSS weight
W_H2H_BASE = 0.25  # H2H weight

BLEND_ALPHA = 0.50
VALUE_MIN = 0.05
PROB_MIN = 0.55
KELLY_MIN = 0.02
MAX_GOALS_FOR_MATRIX = 5

# Güven eşikleri
CONFIDENCE_HIGH = 0.65
CONFIDENCE_MED = 0.55

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

DEBUG = True

def debug_print(msg: str, level: str = "INFO"):
    """Debug çıktısı yazdırma"""
    if DEBUG:
        icons = {"INFO": "ℹ️", "CALC": "🔢", "DATA": "📊", "WARN": "⚠️", "OK": "✅", "ERROR": "❌"}
        print(f"{icons.get(level, 'ℹ️')} [{level}] {msg}")

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
            debug_print(f"Sayfa çekildi: {url[:60]}...", "OK")
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
    home, away = mm.group(1).strip(), mm.group(2).strip()
    debug_print(f"Takımlar: {home} vs {away}", "DATA")
    return home, away

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
            league=league, date=date_val, home=home, away=away,
            ft_home=ft_h, ft_away=ft_a, ht_home=ht_h, ht_away=ht_a,
            corner_home=corner_home, corner_away=corner_away,
            corner_ht_home=corner_ht_home, corner_ht_away=corner_ht_away,
        )

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
        league=league2, date=date_val2, home=home2, away=away2,
        ft_home=ft_h, ft_away=ft_a, ht_home=ht_h, ht_away=ht_a,
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
# STANDINGS (LİG TABLOSU)
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
            matches=_to_int(g(1)), win=_to_int(g(2)), draw=_to_int(g(3)),
            loss=_to_int(g(4)), scored=_to_int(g(5)), conceded=_to_int(g(6)),
            pts=_to_int(g(7)), rank=_to_int(g(8)),
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
            debug_print(f"Standing bulundu: {team_name} ({len(parsed)} satır)", "DATA")
            return parsed
    return []

def standings_to_splits(rows: List[StandRow]) -> Dict[str, Optional[SplitGFGA]]:
    mp: Dict[str, Optional[SplitGFGA]] = {"Total": None, "Home": None, "Away": None, "Last 6": None}
    for r in rows:
        if r.matches and r.scored is not None and r.conceded is not None:
            mp[r.ft] = SplitGFGA(r.matches, r.scored, r.conceded)
    return mp

# ======================
# ODDS (Bet365)
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
                debug_print(f"Bet365 odds (fallback): 1={cand[0]:.2f}, X={cand[1]:.2f}, 2={cand[2]:.2f}", "DATA")
                return {"1": float(cand[0]), "X": float(cand[1]), "2": float(cand[2])}
        return None

    cell_vals: List[Optional[float]] = [_extract_cell_numeric_from_inner_html(td) for td in tds]

    if len(cell_vals) >= 8:
        o1, ox, o2 = cell_vals[5], cell_vals[6], cell_vals[7]
        if all(v is not None for v in [o1, ox, o2]):
            if all(1.01 <= float(v) <= 200 for v in [o1, ox, o2]):
                debug_print(f"Bet365 odds: 1={o1:.2f}, X={ox:.2f}, 2={o2:.2f}", "DATA")
                return {"1": float(o1), "X": float(ox), "2": float(o2)}

    nums = _extract_all_numbers_loose(strip_tags_keep_text(tr_html))
    if len(nums) >= 6:
        cand = nums[3:6]
        if all(1.01 <= x <= 50 for x in cand):
            debug_print(f"Bet365 odds (fallback): 1={cand[0]:.2f}, X={cand[1]:.2f}, 2={cand[2]:.2f}", "DATA")
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
                debug_print(f"H2H bulundu: {pair_count} maç", "DATA")
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

    if best_pair > 0:
        debug_print(f"H2H bulundu (fallback): {best_pair} maç", "DATA")
    return best_list

def filter_same_league_matches(matches: List[MatchRow], league_name: str) -> List[MatchRow]:
    if not league_name:
        return matches
    lk = norm_key(league_name)
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

    debug_print(f"{team} PSS: {st.n_total} maç, GF={st.gf_total:.2f}, GA={st.ga_total:.2f}, Korner For={st.corners_for:.1f}, Against={st.corners_against:.1f}", "CALC")
    return st

# ======================
# POISSON HELPERS
# ======================
def poisson_pmf(k: int, lam: float) -> float:
    """Poisson Probability Mass Function"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k > 170:
        return 0.0
    try:
        result = math.exp(-lam) * (lam ** k) / math.factorial(k)
        return result
    except (OverflowError, ValueError):
        return 0.0

def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k)"""
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))

# ======================
# CORNER ANALYSIS
# ======================
def analyze_corners_advanced(home_prev: TeamPrevStats, away_prev: TeamPrevStats, h2h_matches: List[MatchRow]) -> Dict[str, Any]:
    """
    Geliştirilmiş korner analizi
    Kaynak 1: H2H korner ortalamaları
    Kaynak 2: PSS (önceki maçlar) korner ortalamaları
    """
    debug_print("=== KORNER ANALİZİ ===", "INFO")
    
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

    debug_print(f"H2H Korner: Toplam Ort={h2h_total_avg:.1f} (Ev:{h2h_home_avg:.1f} Dep:{h2h_away_avg:.1f}) [{len(h2h_total)} maç]", "DATA")
    debug_print(f"PSS Korner: Ev For={pss_home_for:.1f}/Ag={pss_home_against:.1f}, Dep For={pss_away_for:.1f}/Ag={pss_away_against:.1f}", "DATA")

    # Lambda hesaplama
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
    debug_print(f"Korner λ: Toplam={total_corners:.2f} (Ev:{predicted_home:.1f} Dep:{predicted_away:.1f})", "CALC")

    # Poisson ile üst/alt tahminleri
    predictions = {}
    for line in [8.5, 9.5, 10.5, 11.5]:
        k = int(math.floor(line))
        p_under_or_eq = poisson_cdf(k, total_corners)
        p_over = 1.0 - p_under_or_eq
        predictions[f"O{line}"] = float(max(0.0, min(1.0, p_over)))
        predictions[f"U{line}"] = float(1.0 - predictions[f"O{line}"])
        debug_print(f"Korner {line}: Üst=%{p_over*100:.1f} Alt=%{(1-p_over)*100:.1f}", "CALC")

    data_points = len(h2h_total) + (1 if (pss_home_for > 0 or pss_away_for > 0) else 0)
    if data_points >= 8:
        confidence = "Yüksek"
    elif data_points >= 4:
        confidence = "Orta"
    else:
        confidence = "Düşük"

    debug_print(f"Korner güven: {confidence} ({data_points} veri noktası)", "INFO")

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
# LAMBDA COMPUTATION (3 KAYNAK: Standing + PSS + H2H)
# ======================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in w.values())
    if s <= 1e-9:
        return {}
    return {k: max(0.0, v) / s for k, v in w.items()}

def compute_component_standings(st_home: Dict[str, Optional[SplitGFGA]],
                                st_away: Dict[str, Optional[SplitGFGA]]) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """Standing (Lig Tablosu) verilerinden lambda hesapla"""
    hh = st_home.get("Home") or st_home.get("Total")
    aa = st_away.get("Away") or st_away.get("Total")
    if not hh or not aa or hh.matches < 3 or aa.matches < 3:
        return None
    lam_h = (hh.gf_pg + aa.ga_pg) / 2.0
    lam_a = (aa.gf_pg + hh.ga_pg) / 2.0
    meta = {
        "home_split": {"matches": hh.matches, "gf_pg": round(hh.gf_pg, 2), "ga_pg": round(hh.ga_pg, 2)},
        "away_split": {"matches": aa.matches, "gf_pg": round(aa.gf_pg, 2), "ga_pg": round(aa.ga_pg, 2)},
        "formula": "Standing: (Home_GF_pg + Away_GA_pg) / 2",
    }
    debug_print(f"Standing λ: Ev={lam_h:.2f}, Dep={lam_a:.2f} | Ev: {hh.matches}maç GF={hh.gf_pg:.2f}, Dep: {aa.matches}maç GA={aa.ga_pg:.2f}", "CALC")
    return lam_h, lam_a, meta

def compute_component_pss(home_prev: TeamPrevStats, away_prev: TeamPrevStats) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """PSS (Önceki Maçlar - Same League + Home/Away filtreli) verilerinden lambda hesapla"""
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
        "home_ga": round(h_ga, 2),
        "away_gf": round(a_gf, 2),
        "away_ga": round(a_ga, 2),
        "formula": "PSS: (home_gf + away_ga) / 2"
    }
    debug_print(f"PSS λ: Ev={lam_h:.2f}, Dep={lam_a:.2f} | Ev: {home_prev.n_total}maç GF={h_gf:.2f}, Dep: {away_prev.n_total}maç GF={a_gf:.2f}", "CALC")
    return lam_h, lam_a, meta

def compute_component_h2h(h2h_matches: List[MatchRow], home_team: str, away_team: str) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """H2H (Karşılıklı Maçlar) verilerinden lambda hesapla"""
    if not h2h_matches or len(h2h_matches) < 3:
        return None
    hk = norm_key(home_team)
    ak = norm_key(away_team)
    used = h2h_matches[:H2H_N]
    hg, ag = [], []
    for m in used:
        if norm_key(m.home) == hk and norm_key(m.away) == ak:
            hg.append(m.ft_home)
            ag.append(m.ft_away)
        elif norm_key(m.home) == ak and norm_key(m.away) == hk:
            hg.append(m.ft_away)
            ag.append(m.ft_home)
    if len(hg) < 3:
        return None
    lam_h = sum(hg) / len(hg)
    lam_a = sum(ag) / len(ag)
    meta = {"matches": len(hg), "home_goals_avg": round(lam_h, 2), "away_goals_avg": round(lam_a, 2)}
    debug_print(f"H2H λ: Ev={lam_h:.2f}, Dep={lam_a:.2f} | {len(hg)} maç", "CALC")
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
    Üç kaynaktan lambda hesapla ve ağırlıklı birleştir:
    1. Standing (Lig Tablosu)
    2. PSS (Önceki Maçlar)
    3. H2H (Karşılıklı Maçlar)
    """
    debug_print("=== LAMBDA HESAPLAMA (3 KAYNAK) ===", "INFO")
    info: Dict[str, Any] = {"components": {}, "weights_used": {}, "warnings": []}
    comps: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}

    # 1. Standing
    stc = compute_component_standings(st_home_s, st_away_s)
    if stc:
        comps["standing"] = stc
        debug_print("✓ Standing verisi kullanılıyor", "OK")

    # 2. PSS
    pss = compute_component_pss(home_prev, away_prev)
    if pss:
        comps["pss"] = pss
        debug_print("✓ PSS verisi kullanılıyor", "OK")

    # 3. H2H
    h2c = compute_component_h2h(h2h_used, home_team, away_team)
    if h2c:
        comps["h2h"] = h2c
        debug_print("✓ H2H verisi kullanılıyor", "OK")

    # Ağırlıklar
    w = {}
    if "standing" in comps: w["standing"] = W_ST_BASE
    if "pss" in comps:      w["pss"] = W_PSS_BASE
    if "h2h" in comps:      w["h2h"] = W_H2H_BASE

    w_norm = normalize_weights(w)
    info["weights_used"] = w_norm
    debug_print(f"Ağırlıklar: {', '.join([f'{k}=%{v*100:.0f}' for k, v in w_norm.items()])}", "INFO")

    if not w_norm:
        info["warnings"].append("Yetersiz veri -> default λ=1.20")
        lh, la = 1.20, 1.20
        debug_print("⚠️  Yetersiz veri, default λ kullanılıyor", "WARN")
    else:
        lh = 0.0
        la = 0.0
        for k, wk in w_norm.items():
            ch, ca, meta = comps[k]
            info["components"][k] = {"lam_home": round(ch, 2), "lam_away": round(ca, 2), "meta": meta}
            lh += wk * ch
            la += wk * ca

    lh, la, clamp_warn = clamp_lambda(lh, la)
    if clamp_warn:
        info["warnings"].extend(clamp_warn)
    
    debug_print(f"Final λ: Ev={lh:.2f}, Dep={la:.2f}, Toplam={lh+la:.2f}", "OK")
    return lh, la, info

# ======================
# POISSON & MC
# ======================
def build_score_matrix(lh: float, la: float, max_g: int = 5) -> Dict[Tuple[int, int], float]:
    debug_print("Skor matrisi oluşturuluyor (Poisson)...", "CALC")
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

    debug_print(f"Poisson: 1=%{p1*100:.1f} X=%{px*100:.1f} 2=%{p2*100:.1f} BTTS=%{btts*100:.1f} O2.5=%{out['O2.5']*100:.1f}", "CALC")
    return out

def monte_carlo(lh: float, la: float, n: int, seed: Optional[int] = 42) -> Dict[str, Any]:
    debug_print(f"Monte Carlo simulasyonu ({n:,} iterasyon)...", "CALC")
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
    debug_print(f"Monte Carlo tamamlandı. En olası skor: {top10_list[0][0]} (%{top10_list[0][1]:.1f})", "OK")
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
# GÜVEN GÖSTERGELERİ
# ======================
def get_confidence_indicator(prob: float) -> str:
    """
    🟢 ✅ Yüksek (>=65%)
    🟡 ⚠️  Orta (55-65%)
    🔴 ❌ Düşük (<55%)
    """
    if prob >= CONFIDENCE_HIGH:
        return "✅"
    elif prob >= CONFIDENCE_MED:
        return "⚠️"
    else:
        return "❌"

def get_confidence_label(prob: float) -> str:
    if prob >= CONFIDENCE_HIGH:
        return "Yüksek"
    elif prob >= CONFIDENCE_MED:
        return "Orta"
    return "Düşük"

def format_market_prediction(market: str, prob: float) -> str:
    indicator = get_confidence_indicator(prob)
    label = get_confidence_label(prob)
    return f"{indicator} {market}: %{prob*100:.1f} ({label})"

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
        return "2.5 ÜST", p_o25, get_confidence_label(p_o25)
    return "2.5 ALT", p_u25, get_confidence_label(p_u25)

def net_btts_prediction(probs: Dict[str, float]) -> Tuple[str, float, str]:
    p_btts = probs.get("BTTS", 0)
    p_no = 1.0 - p_btts
    if p_btts >= p_no:
        return "VAR", p_btts, get_confidence_label(p_btts)
    return "YOK", p_no, get_confidence_label(p_no)

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
    lines.append("=" * 75)
    lines.append(f"  {t['home']} vs {t['away']}".center(75))
    lines.append("=" * 75)

    # Olası Skorlar
    lines.append(f"\n📊 OLASI SKORLAR:")
    for i, (score, prob) in enumerate(top7[:5], 1):
        bar = "█" * int(prob * 50)
        lines.append(f"  {i}. {score:6s} %{prob*100:4.1f}  {bar}")

    # Net Tahmin
    lines.append(f"\n🎯 NET TAHMİN:")
    lines.append(f"  Ana Skor: {top7[0][0]} (%{top7[0][1]*100:.1f})")
    if len(top7) >= 3:
        lines.append(f"  Alt Skorlar: {top7[1][0]} (%{top7[1][1]*100:.1f}), {top7[2][0]} (%{top7[2][1]*100:.1f})")

    # 2.5 Üst/Alt
    net_ou, net_ou_p, net_ou_label = net_ou_prediction(blend)
    indicator_ou = get_confidence_indicator(net_ou_p)
    lines.append(f"\n{indicator_ou} 2.5 ÜST/ALT: {net_ou} (%{net_ou_p*100:.1f}) - {net_ou_label}")

    # 3.5 Üst/Alt
    p_o35 = blend.get("O3.5", 0)
    p_u35 = blend.get("U3.5", 0)
    if p_o35 >= p_u35:
        net_ou35 = "3.5 ÜST"
        net_ou35_p = p_o35
    else:
        net_ou35 = "3.5 ALT"
        net_ou35_p = p_u35
    indicator_ou35 = get_confidence_indicator(net_ou35_p)
    lines.append(f"{indicator_ou35} 3.5 ÜST/ALT: {net_ou35} (%{net_ou35_p*100:.1f}) - {get_confidence_label(net_ou35_p)}")

    # KG
    net_btts, net_btts_p, net_btts_label = net_btts_prediction(blend)
    indicator_btts = get_confidence_indicator(net_btts_p)
    lines.append(f"{indicator_btts} KARŞILIKLI GOL: {net_btts} (%{net_btts_p*100:.1f}) - {net_btts_label}")

    # 1X2
    lines.append(f"\n⚽ 1X2 OLASILILIKLARI:")
    p1 = blend.get("1", 0)
    px = blend.get("X", 0)
    p2 = blend.get("2", 0)
    lines.append(f"  {get_confidence_indicator(p1)} Ev Kazanır (1): %{p1*100:.1f} ({get_confidence_label(p1)})")
    lines.append(f"  {get_confidence_indicator(px)} Beraberlik (X): %{px*100:.1f} ({get_confidence_label(px)})")
    lines.append(f"  {get_confidence_indicator(p2)} Deplasman (2): %{p2*100:.1f} ({get_confidence_label(p2)})")

    # KORNER
    corners = data.get("corner_analysis", {})
    if corners and corners.get("total_corners", 0) > 0:
        lines.append(f"\n🚩 KORNER TAHMİNİ:")
        lines.append(f"  Toplam λ: {corners['total_corners']} (Ev: {corners['predicted_home_corners']} | Dep: {corners['predicted_away_corners']})")
        lines.append(f"  H2H Korner Ort: {corners['h2h_avg']} ({corners['h2h_data_count']} maç)")
        lines.append(f"  Güven: {corners['confidence']}")
        
        preds = corners.get("predictions", {})
        lines.append(f"\n  📌 KORNER ÜST/ALT TAHMİNLERİ:")
        for line in ["8.5", "9.5", "10.5", "11.5"]:
            p_over = preds.get(f"O{line}", 0)
            p_under = preds.get(f"U{line}", 0)
            ind_over = get_confidence_indicator(p_over)
            ind_under = get_confidence_indicator(p_under)
            lines.append(f"  {ind_over} O{line}: %{p_over*100:.1f} ({get_confidence_label(p_over)})  |  {ind_under} U{line}: %{p_under*100:.1f} ({get_confidence_label(p_under)})")

    # BAHİS ANALİZİ
    vb = data.get("value_bets", {})
    if vb.get("used_odds"):
        lines.append(f"\n💰 BAHİS ANALİZİ (Bet365 Initial 1X2):")
        has_value = False
        for row in vb.get("table", []):
            if row["value"] >= VALUE_MIN and row["prob"] >= PROB_MIN:
                lines.append(f"  ✅ {row['market']}: Oran {row['odds']:.2f} | Prob %{row['prob']*100:.1f} | Value %{row['value']*100:+.1f} | Kelly %{row['kelly']*100:.1f}")
                has_value = True
        if not has_value:
            lines.append("  ⚠️  Değerli bahis bulunamadı")
        lines.append(f"\n📋 KARAR: {vb.get('decision', 'Analiz edilemedi')}")
    else:
        lines.append("\n⚠️  Oran verisi yok - value analizi yapılamadı")

    # VERİ KAYNAKLARI
    ds = data["data_sources"]
    lambda_info = data["lambda"]["info"]

    lines.append(f"\n📚 KULLANILAN VERİ KAYNAKLARI:")
    lines.append(f"  {'✅' if ds['standings_used'] else '❌'} Standing (Lig Tablosu): {'Var' if ds['standings_used'] else 'Yok'}")
    lines.append(f"  ✅ PSS (Önceki Maçlar): Ev {ds['home_prev_matches']} maç (Home-only) | Dep {ds['away_prev_matches']} maç (Away-only)")
    if ds['pss_same_league_used']:
        lines.append(f"     └─ Same League filtresi uygulandı")
    lines.append(f"  {'✅' if ds['h2h_matches']>0 else '❌'} H2H (Karşılıklı Maçlar): {ds['h2h_matches']} maç")
    if ds['h2h_same_league_used']:
        lines.append(f"     └─ Same League filtresi uygulandı")

    if lambda_info.get("weights_used"):
        lines.append(f"\n⚖️  LAMBDA AĞIRLIKLARI:")
        for k, v in lambda_info["weights_used"].items():
            k_name = {"standing": "Standing", "pss": "PSS", "h2h": "H2H"}.get(k, k)
            lines.append(f"  {k_name}: %{v*100:.0f}")

    lines.append(f"\n🔢 LAMBDA DEĞERLERİ:")
    lines.append(f"  Ev Sahibi λ: {data['lambda']['home']:.2f}")
    lines.append(f"  Deplasman λ: {data['lambda']['away']:.2f}")
    lines.append(f"  Toplam λ: {data['lambda']['total']:.2f}")

    lines.append("=" * 75)
    return "\n".join(lines)

# ======================
# MAIN ANALYSIS
# ======================
def analyze_nowgoal(url: str, odds: Optional[Dict[str, float]] = None, mc_runs: int = MC_RUNS_DEFAULT) -> Dict[str, Any]:
    debug_print("="*75, "INFO")
    debug_print("🚀 NowGoal Match Analyzer v5.0 BAŞLIYOR", "INFO")
    debug_print("="*75, "INFO")
    
    h2h_url = build_h2h_url(url)
    html = safe_get(h2h_url, referer=extract_base_domain(url))

    home_team, away_team = parse_teams_from_title(html)
    if not home_team or not away_team:
        raise RuntimeError("Takım isimleri çıkarılamadı")

    league_match = re.search(r'<span[^>]*class=["\']?sclassLink["\']?[^>]*>([^<]+)</span>', html)
    league_name = strip_tags_keep_text(league_match.group(1)) if league_match else ""
    debug_print(f"Lig: {league_name or 'Bilinmiyor'}", "DATA")

    # 1. STANDING (Lig Tablosu)
    debug_print("\n=== VERİ TOPLAMA: STANDING ===", "INFO")
    st_home_rows = extract_standings_for_team(html, home_team)
    st_away_rows = extract_standings_for_team(html, away_team)
    st_home = standings_to_splits(st_home_rows)
    st_away = standings_to_splits(st_away_rows)

    # 2. H2H (Karşılıklı Maçlar)
    debug_print("\n=== VERİ TOPLAMA: H2H ===", "INFO")
    h2h_all = extract_h2h_matches(html, home_team, away_team)
    h2h_pair = [m for m in h2h_all if is_h2h_pair(m, home_team, away_team)]
    h2h_pair = sort_matches_desc(dedupe_matches(h2h_pair))

    h2h_same = filter_same_league_matches(h2h_pair, league_name) if league_name else h2h_pair
    if len(h2h_same) >= 3:
        h2h_used = h2h_same[:H2H_N]
        h2h_same_used = True
        debug_print(f"H2H: Same League kullanılıyor ({len(h2h_used)} maç)", "DATA")
    else:
        h2h_used = h2h_pair[:H2H_N]
        h2h_same_used = False
        debug_print(f"H2H: Tüm maçlar kullanılıyor ({len(h2h_used)} maç)", "DATA")

    # 3. PSS (Önceki Maçlar)
    debug_print("\n=== VERİ TOPLAMA: PSS ===", "INFO")
    prev_home_tabs, prev_away_tabs = extract_previous_from_page(html)

    prev_home_raw = parse_matches_from_table_html(prev_home_tabs[0]) if prev_home_tabs else []
    prev_away_raw = parse_matches_from_table_html(prev_away_tabs[0]) if prev_away_tabs else []

    if league_name:
        prev_home_raw = filter_same_league_matches(prev_home_raw, league_name)
        prev_away_raw = filter_same_league_matches(prev_away_raw, league_name)
        debug_print(f"PSS: Same League filtresi uygulandı", "DATA")

    prev_home_sel = filter_team_home_only(prev_home_raw, home_team)[:RECENT_N]
    prev_away_sel = filter_team_away_only(prev_away_raw, away_team)[:RECENT_N]
    debug_print(f"PSS: Ev {len(prev_home_sel)} maç (Home-only), Dep {len(prev_away_sel)} maç (Away-only)", "DATA")

    home_prev_stats = build_prev_stats(home_team, prev_home_sel)
    away_prev_stats = build_prev_stats(away_team, prev_away_sel)

    # LAMBDA HESAPLAMA (3 kaynak birleşimi)
    lam_home, lam_away, lambda_info = compute_lambdas(
        st_home_s=st_home,
        st_away_s=st_away,
        home_prev=home_prev_stats,
        away_prev=away_prev_stats,
        h2h_used=h2h_used,
        home_team=home_team,
        away_team=away_team
    )

    # POISSON
    debug_print("\n=== POISSON ANALİZİ ===", "INFO")
    score_mat = build_score_matrix(lam_home, lam_away, max_g=MAX_GOALS_FOR_MATRIX)
    poisson_market = market_probs_from_matrix(score_mat)
    top7 = top_scores_from_matrix(score_mat, top_n=7)

    # MONTE CARLO
    debug_print("\n=== MONTE CARLO ANALİZİ ===", "INFO")
    mc = monte_carlo(lam_home, lam_away, n=max(10_000, int(mc_runs)), seed=42)

    # MODEL UYUMU
    diff, diff_label = model_agreement(poisson_market, mc["p"])
    debug_print(f"Model uyumu: {diff_label} (fark: {diff:.4f})", "INFO")
    
    # BLENDED PROBS
    blended = blend_probs(poisson_market, mc["p"], alpha=BLEND_ALPHA)

    # KORNER ANALİZİ
    corner_analysis = analyze_corners_advanced(home_prev_stats, away_prev_stats, h2h_used)

    # ODDS & VALUE
    if not odds:
        debug_print("\n=== ODDS ÇEKİLİYOR ===", "INFO")
        odds = extract_bet365_initial_odds(url)

    value_block = {"used_odds": False}
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
        debug_print(f"Value analizi: {len(qualified)} değerli bahis", "DATA")

    # SONUÇ
    data = {
        "url": h2h_url,
        "teams": {"home": home_team, "away": away_team},
        "league": league_name,
        "lambda": {"home": lam_home, "away": lam_away, "total": lam_home + lam_away, "info": lambda_info},
        "poisson": {"market_probs": poisson_market, "top7_scores": top7},
        "mc": mc,
        "model_agreement": {"diff": diff, "label": diff_label},
        "blended_probs": blended,
        "corner_analysis": corner_analysis,
        "value_bets": value_block,
        "data_sources": {
            "standings_used": len(st_home_rows) > 0 and len(st_away_rows) > 0,
            "h2h_matches": len(h2h_used),
            "h2h_same_league_used": h2h_same_used,
            "home_prev_matches": len(prev_home_sel),
            "away_prev_matches": len(prev_away_sel),
            "pss_same_league_used": bool(league_name)
        }
    }

    data["report_comprehensive"] = format_comprehensive_report(data)
    
    debug_print("="*75, "OK")
    debug_print("✅ ANALİZ TAMAMLANDI!", "OK")
    debug_print("="*75, "OK")
    return data

# ======================
# FLASK APP
# ======================
app = Flask(__name__)

@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "nowgoal-analyzer-api",
        "version": "5.0-final",
        "data_sources": ["Standing", "PSS", "H2H", "Corner Analysis"],
        "features": ["Poisson", "Monte Carlo", "Value Betting", "Confidence Indicators"]
    })

@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "healthy"})

@app.post("/analiz_et")
def analiz_et_route():
    """Türkçe Android API endpoint"""
    try:
        payload = request.get_json(silent=True) or {}
    except Exception as e:
        return jsonify({"ok": False, "error": f"Geçersiz JSON: {e}"}), 400

    url = (payload.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "URL boş olamaz"}), 400

    if not re.match(r'^https?://', url):
        return jsonify({"ok": False, "error": "Geçersiz URL formatı"}), 400

    try:
        data = analyze_nowgoal(url, odds=None, mc_runs=10_000)

        top_skor = data["poisson"]["top7_scores"][0][0]
        blend = data["blended_probs"]

        net_ou, net_ou_p, _ = net_ou_prediction(blend)
        net_btts, net_btts_p, _ = net_btts_prediction(blend)

        # Korner tahminleri
        corners = data["corner_analysis"]["predictions"]
        corner_predictions = []
        for line in ["8.5", "9.5", "10.5", "11.5"]:
            p_over = corners.get(f"O{line}", 0)
            p_under = corners.get(f"U{line}", 0)
            corner_predictions.append({
                "line": line,
                "over": format_market_prediction(f"O{line}", p_over),
                "under": format_market_prediction(f"U{line}", p_under)
            })

        return jsonify({
            "ok": True,
            "skor": top_skor,
            "alt_ust_2_5": format_market_prediction(net_ou, net_ou_p),
            "alt_ust_3_5": format_market_prediction(
                "3.5 ÜST" if blend.get("O3.5", 0) >= blend.get("U3.5", 0) else "3.5 ALT",
                max(blend.get("O3.5", 0), blend.get("U3.5", 0))
            ),
            "kg": format_market_prediction(net_btts, net_btts_p),
            "korner_tahminleri": corner_predictions,
            "karar": data["value_bets"].get("decision", "Oran gerekli"),
            "odds_used": data["value_bets"].get("used_odds", False),
            "veri_kaynaklari": {
                "standing": data["data_sources"]["standings_used"],
                "pss_home": data["data_sources"]["home_prev_matches"],
                "pss_away": data["data_sources"]["away_prev_matches"],
                "h2h": data["data_sources"]["h2h_matches"]
            },
            "detay": data["report_comprehensive"]
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Analiz hatası: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.post("/analyze")
def analyze_route():
    """Full English API endpoint"""
    try:
        payload = request.get_json(silent=True) or {}
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid JSON: {e}"}), 400

    url = (payload.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "url required"}), 400

    if not re.match(r'^https?://', url):
        return jsonify({"ok": False, "error": "Invalid URL"}), 400

    odds = payload.get("odds")
    mc_runs = payload.get("mc_runs", MC_RUNS_DEFAULT)

    try:
        mc_runs = int(mc_runs)
        if mc_runs < 100 or mc_runs > 100_000:
            mc_runs = MC_RUNS_DEFAULT
    except (ValueError, TypeError):
        mc_runs = MC_RUNS_DEFAULT

    try:
        data = analyze_nowgoal(url, odds=odds, mc_runs=mc_runs)
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
        print("="*75)
        print("🚀 NowGoal Analyzer v5.0 FINAL".center(75))
        print("="*75)
        print("\n✅ VERİ KAYNAKLARI:")
        print("  1. Standing (Lig Tablosu): Home/Away/Total gol ortalamaları")
        print("  2. PSS (Önceki Maçlar): Same League + Home-only/Away-only")
        print("  3. H2H (Karşılıklı Maçlar): Direct matchup + korner verileri")
        print("\n✅ TAHMİN PAZARLARI:")
        print("  - 1X2 (Maç sonucu)")
        print("  - 2.5 & 3.5 Üst/Alt")
        print("  - KG Var/Yok")
        print("  - Korner: 8.5, 9.5, 10.5, 11.5 üst/alt")
        print("\n🎯 GÜVEN GÖSTERGELERİ:")
        print("  🟢 ✅ Yeşil: %65+ (Oynanabilir)")
        print("  🟡 ⚠️  Sarı: %55-65
