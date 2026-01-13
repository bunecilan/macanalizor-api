# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - Full Version 4.3 (LEAGUE FIX & UNABRIDGED)
Flask API with Corner Analysis & Enhanced Value Betting

FIX PACK (v4.3):
1) Lig İsmi Çekme (Robust Extraction):
   - HTML Title, Span Class ve JS Variable üzerinden lig ismini arar.
2) Same League Filtresi:
   - Token (Kelime) bazlı eşleşme.
   - Prefix (İlk 3 harf) eşleşmesi (Örn: TUR -> Turkey).
3) Kesin Filtreleme:
   - Lig filtresi sonuç vermezse "Fallback" yapmaz, yanlış veriyi engellemek için boş liste döner.
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
# CONFIGURATION
# ======================
MC_RUNS_DEFAULT = 10_000
RECENT_N = 10
H2H_N = 10

# Ağırlıklar (Toplamı 1.0 olmasa da normalize edilir)
W_ST_BASE = 0.45   # Puan Durumu
W_PSS_BASE = 0.30  # Son Maçlar (PSS)
W_H2H_BASE = 0.25  # Aralarındaki Maçlar

BLEND_ALPHA = 0.50 # Poisson ve Monte Carlo karışım oranı
VALUE_MIN = 0.05   # %5 Value altı gösterilmez
PROB_MIN = 0.55    # %55 Olasılık altı gösterilmez
KELLY_MIN = 0.02   # Kelly kriteri eşiği
MAX_GOALS_FOR_MATRIX = 5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# ======================
# REGEX PATTERNS
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
# UTILITY FUNCTIONS
# ======================
def norm_key(s: str) -> str:
    """Metni normalleştirir (küçük harf, sadece alfanümerik)."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def normalize_date(d: str) -> Optional[str]:
    """Tarih formatını dd-mm-yyyy standardına çevirir."""
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
    """Sıralama için tarihi tuple formatına çevirir (yyyy, mm, dd)."""
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
# HTML PARSING
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
    """Hücreleri silmeden (boş olsa bile) satır satır çeker."""
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
# NETWORK / FETCHING
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

# ======================
# DATA EXTRACTION LOGIC
# ======================
def parse_teams_from_title(html: str) -> Tuple[str, str]:
    m = re.search(r"<title>\s*(.*?)\s*</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = strip_tags_keep_text(m.group(1)) if m else ""
    mm = re.search(r"(.+?)\s+VS\s+(.+?)(?:\s+-|\s+\||$)", title, flags=re.IGNORECASE)
    if not mm:
        mm = re.search(r"(.+?)\s+vs\s+(.+?)(?:\s+-|\s+\||$)", title, flags=re.IGNORECASE)
    if not mm:
        return "", ""
    return mm.group(1).strip(), mm.group(2).strip()

def extract_league_name_robust(html: str) -> str:
    """
    Lig ismini bulmak için 3 yöntem dener.
    NowGoal bazen sclassLink, bazen title, bazen JS variable kullanır.
    """
    # 1. Yöntem: Standart sclassLink Span
    m1 = re.search(r'<span[^>]*class=["\']?sclassLink["\']?[^>]*>([^<]+)</span>', html, flags=re.I)
    if m1:
        txt = strip_tags_keep_text(m1.group(1))
        if txt: return txt

    # 2. Yöntem: Title Tag Analizi
    # Title: "Team A vs Team B ... - League Name - NowGoal"
    m_title = re.search(r"<title>\s*(.*?)\s*</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if m_title:
        title_txt = strip_tags_keep_text(m_title.group(1))
        parts = [p.strip() for p in title_txt.split('-')]
        # Genelde sondan 2. eleman lig ismidir.
        if len(parts) >= 3:
            candidate = parts[-2]
            if len(candidate) > 2 and "NowGoal" not in candidate:
                return candidate
    
    # 3. Yöntem: JavaScript Değişkeni
    m_js = re.search(r'sclassName\s*=\s*["\']([^"\']+)["\']', html)
    if m_js:
        return m_js.group(1)

    return ""

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
# PARSING: CORNERS & MATCHES
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
    
    # 1. Senaryo: Standart Tablo Yapısı
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

    # 2. Senaryo: Fallback (Skor hücresini arayıp etrafına bakma)
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

    # Skorun solunda Home, sağında Away ara
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
# STANDINGS PARSING
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
        if not all(k in text_low for k in ["matches", "win", "draw", "loss", "scored", "conceded"]):
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
# ODDS EXTRACTION
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
    return _extract_first_float(inner_html)

def extract_bet365_initial_1x2_from_oddscomp_html(odds_html: str) -> Optional[Dict[str, float]]:
    if not odds_html:
        return None
    tr_m = re.search(r"(<tr\b[^>]*>.*?Bet365.*?</tr>)", odds_html, flags=re.I | re.S)
    if not tr_m:
        return None

    tr_html = tr_m.group(1)
    tds = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr_html, flags=re.I | re.S)
    
    # Hücrelerden
    if tds and len(tds) >= 8:
        cell_vals: List[Optional[float]] = [_extract_cell_numeric_from_inner_html(td) for td in tds]
        if len(cell_vals) >= 8:
            o1, ox, o2 = cell_vals[5], cell_vals[6], cell_vals[7]
            if all(v is not None for v in [o1, ox, o2]):
                if all(1.01 <= float(v) <= 200 for v in [o1, ox, o2]):
                    return {"1": float(o1), "X": float(ox), "2": float(o2)}
    
    # Fallback (Satır içi text)
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
        return extract_bet365_initial_1x2_from_oddscomp_html(html)
    except Exception:
        return None

# ======================
# MATCH FILTERS (CORE FIX)
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

def filter_same_league_matches(matches: List[MatchRow], league_name: str) -> List[MatchRow]:
    """
    GELİŞMİŞ LİG FİLTRESİ (v4.3 FIX):
    1. Lig ismi (league_name) bulunamadıysa boş liste döner (yanlış veriyi önlemek için).
    2. Token (Kelime) eşleşmesi yapar.
    3. Prefix (İlk 3 harf) kontrolü yapar.
    """
    if not league_name:
        return []

    lk = norm_key(league_name)
    # Hedef lig isminin kelimeleri (örn: ["turkey", "super", "lig"])
    target_words = set(re.findall(r'[a-z0-9]+', league_name.lower()))
    filtered_target_words = {w for w in target_words if len(w) > 2 and w not in {"the", "and", "lig", "cup", "league"}}

    # Prefix (Örn: "Turkey" -> "tur")
    prefix = lk[:3] if len(lk) >= 3 else lk

    out = []
    for m in matches:
        ml = norm_key(m.league)
        
        # 1. Tam içerik (Substring)
        if lk and ml and (lk in ml or ml in lk):
            out.append(m)
            continue
        
        # 2. Token kesişimi
        m_words = set(re.findall(r'[a-z0-9]+', m.league.lower()))
        if filtered_target_words:
            if not filtered_target_words.isdisjoint(m_words):
                out.append(m)
                continue
        
        # 3. Prefix kontrolü (Kısaltmalar için)
        # Örn: "Turkey..." ligi için maçta "TUR SL" yazıyorsa
        if prefix and ml.startswith(prefix):
            out.append(m)
            continue

    return out

def filter_team_home_only(matches: List[MatchRow], team: str) -> List[MatchRow]:
    tk = norm_key(team)
    return [m for m in matches if norm_key(m.home) == tk]

def filter_team_away_only(matches: List[MatchRow], team: str) -> List[MatchRow]:
    tk = norm_key(team)
    return [m for m in matches if norm_key(m.away) == tk]

# ======================
# STATISTICS & LAMBDA
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
        p_over = 1.0 - poisson_cdf(k, total_corners)
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
        "predictions": predictions,
        "confidence": confidence
    }

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
    if home_prev.n_total < 3 or away_prev.n_total < 3:
        return None

    lam_h = (home_prev.gf_total + away_prev.ga_total) / 2.0
    lam_a = (away_prev.gf_total + home_prev.ga_total) / 2.0

    meta = {
        "home_matches": home_prev.n_total,
        "away_matches": away_prev.n_total,
        "home_gf": round(home_prev.gf_total, 2),
        "away_gf": round(away_prev.gf_total, 2),
        "formula": "PSS (Filtered)"
    }
    return lam_h, lam_a, meta

def compute_component_h2h(h2h_matches: List[MatchRow], home_team: str, away_team: str) -> Optional[Tuple[float, float, Dict[str, Any]]]:
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

def compute_lambdas(st_home_s, st_away_s, home_prev, away_prev, h2h_used, home_team, away_team):
    info = {"components": {}, "weights_used": {}, "warnings": []}
    comps = {}

    stc = compute_component_standings(st_home_s, st_away_s)
    if stc:
        comps["standing"] = stc

    pss = compute_component_pss(home_prev, away_prev)
    if pss:
        comps["pss"] = pss

    h2c = compute_component_h2h(h2h_used, home_team, away_team)
    if h2c:
        comps["h2h"] = h2c

    w = {}
    if "standing" in comps: w["standing"] = W_ST_BASE
    if "pss" in comps:      w["pss"] = W_PSS_BASE
    if "h2h" in comps:      w["h2h"] = W_H2H_BASE

    w_norm = normalize_weights(w)
    info["weights_used"] = w_norm

    if not w_norm:
        info["warnings"].append("Yetersiz veri -> default λ")
        lh, la = 1.20, 1.20
    else:
        lh = 0.0; la = 0.0
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
# SIMULATION
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

    return {
        "p": {
            "1": p(hg > ag),
            "X": p(hg == ag),
            "2": p(hg < ag),
            "BTTS": p((hg >= 1) & (ag >= 1)),
            "O2.5": p(total >= 3),
            "U2.5": p(total <= 2),
        },
        "TOP10": top10_list,
    }

def blend_probs(p1: Dict[str, float], p2: Dict[str, float], alpha: float) -> Dict[str, float]:
    out = {}
    for k in set(list(p1.keys()) + list(p2.keys())):
        if k in p1 and k in p2:
            out[k] = alpha * p1[k] + (1.0 - alpha) * p2[k]
        else:
            out[k] = p1.get(k, p2.get(k, 0.0))
    return out

def value_and_kelly(prob: float, odds: float) -> Tuple[float, float]:
    if odds <= 1.0 or prob <= 0.0 or prob >= 1.0:
        return 0.0, 0.0
    v = odds * prob - 1.0
    b = odds - 1.0
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    return v, max(0.0, kelly)

def final_decision(qualified: List[Tuple[str, float, float, float, float]], diff: float) -> str:
    if not qualified:
        return f"OYNAMA (Eşik sağlanmadı, Model Farkı: {diff:.2f})"
    if diff > 0.10:
        return f"TEMKİNLİ (Zayıf model uyumu: {diff:.2f})"
    best = sorted(qualified, key=lambda x: x[3], reverse=True)[0]
    mkt, prob, odds, val, qk = best
    return f"OYNANABİLİR → {mkt} (Prob: %{prob*100:.1f}, Oran: {odds:.2f}, Value: %{val*100:+.1f})"

def format_comprehensive_report(data: Dict[str, Any]) -> str:
    t = data["teams"]
    blend = data["blended_probs"]
    top7 = data["poisson"]["top7_scores"]
    vb = data.get("value_bets", {})

    lines = []
    lines.append("=" * 60)
    lines.append(f"  {t['home']} vs {t['away']}")
    lines.append(f"  LİG: {data.get('league') or 'BİLİNMİYOR'}")
    lines.append("=" * 60)

    lines.append(f"\nOLASI SKORLAR:")
    for i, (score, prob) in enumerate(top7[:5], 1):
        bar = "█" * int(prob * 50)
        lines.append(f"  {i}. {score:6s} %{prob*100:4.1f}  {bar}")

    lines.append(f"\nTAHMİN:")
    lines.append(f"  Ana Skor: {top7[0][0]}")
    lines.append(f"  Alt/Üst 2.5: {'ÜST' if blend.get('O2.5',0)>blend.get('U2.5',0) else 'ALT'} (%{max(blend.get('O2.5',0), blend.get('U2.5',0))*100:.1f})")
    lines.append(f"  KG VAR: {'EVET' if blend.get('BTTS',0)>0.5 else 'HAYIR'} (%{blend.get('BTTS',0)*100:.1f})")

    if vb.get("used_odds"):
        lines.append(f"\nDEĞER ANALİZİ:")
        has_value = False
        for row in vb.get("table", []):
            if row["value"] >= VALUE_MIN:
                lines.append(f"  ✅ {row['market']}: Oran {row['odds']:.2f} | Value %{row['value']*100:+.1f}")
                has_value = True
        if not has_value:
            lines.append("  ⚠️  Değerli oran bulunamadı.")
        lines.append(f"  KARAR: {vb.get('decision')}")

    ds = data["data_sources"]
    lines.append(f"\nVERİ KAYNAKLARI:")
    lines.append(f"  Lig Filtresi: {'Aktif' if ds['league_found'] else 'Bulunamadı (Devre Dışı)'}")
    lines.append(f"  H2H Maç: {ds['h2h_matches']} (Aynı lig: {ds['h2h_same_league_used']})")
    lines.append(f"  Ev PSS: {ds['home_prev_matches']} maç (Sadece Ev + Lig)")
    lines.append(f"  Dep PSS: {ds['away_prev_matches']} maç (Sadece Dep + Lig)")

    return "\n".join(lines)

# ======================
# MAIN LOGIC
# ======================
def analyze_nowgoal(url: str, odds: Optional[Dict[str, float]] = None, mc_runs: int = MC_RUNS_DEFAULT) -> Dict[str, Any]:
    h2h_url = build_h2h_url(url)
    html = safe_get(h2h_url, referer=extract_base_domain(url))

    home_team, away_team = parse_teams_from_title(html)
    if not home_team or not away_team:
        raise RuntimeError("Takım isimleri title'dan çekilemedi.")

    # 1. Lig İsmini Çek (Robust Yöntem)
    league_name = extract_league_name_robust(html)

    st_home_rows = extract_standings_for_team(html, home_team)
    st_away_rows = extract_standings_for_team(html, away_team)
    st_home = standings_to_splits(st_home_rows)
    st_away = standings_to_splits(st_away_rows)

    # H2H Matches
    h2h_all = extract_h2h_matches(html, home_team, away_team)
    h2h_pair = [m for m in h2h_all if is_h2h_pair(m, home_team, away_team)]
    h2h_pair = sort_matches_desc(dedupe_matches(h2h_pair))

    h2h_same = filter_same_league_matches(h2h_pair, league_name) if league_name else []
    
    # H2H: Aynı ligde 3 maç varsa onları kullan, yoksa genel kullan
    if len(h2h_same) >= 3:
        h2h_used = h2h_same[:H2H_N]
        h2h_same_used = True
    else:
        h2h_used = h2h_pair[:H2H_N]
        h2h_same_used = False

    # PSS (Previous Scores)
    prev_home_tabs, prev_away_tabs = extract_previous_from_page(html)
    prev_home_raw = parse_matches_from_table_html(prev_home_tabs[0]) if prev_home_tabs else []
    prev_away_raw = parse_matches_from_table_html(prev_away_tabs[0]) if prev_away_tabs else []

    # --- FİLTRELEME ZİNCİRİ (KESİN ÇÖZÜM) ---
    # 1. Lig Filtresi: Eğer lig bulunursa, sadece o ligdeki maçları al.
    #    Eğer filtre sonucu 0 maç dönerse, 0 maç olarak kalır (fallback yapmaz).
    prev_home_filt = filter_same_league_matches(prev_home_raw, league_name)
    prev_away_filt = filter_same_league_matches(prev_away_raw, league_name)

    # 2. Taraf Filtresi: Home Only / Away Only
    prev_home_sel = filter_team_home_only(prev_home_filt, home_team)[:RECENT_N]
    prev_away_sel = filter_team_away_only(prev_away_filt, away_team)[:RECENT_N]

    home_prev_stats = build_prev_stats(home_team, prev_home_sel)
    away_prev_stats = build_prev_stats(away_team, prev_away_sel)

    # Lambda Hesaplama
    lam_home, lam_away, lambda_info = compute_lambdas(
        st_home_s=st_home,
        st_away_s=st_away,
        home_prev=home_prev_stats,
        away_prev=away_prev_stats,
        h2h_used=h2h_used,
        home_team=home_team,
        away_team=away_team
    )

    # Simülasyonlar
    score_mat = build_score_matrix(lam_home, lam_away, max_g=MAX_GOALS_FOR_MATRIX)
    poisson_market = market_probs_from_matrix(score_mat)
    top7 = sorted(score_mat.items(), key=lambda x: x[1], reverse=True)[:7]

    mc = monte_carlo(lam_home, lam_away, n=max(10_000, int(mc_runs)), seed=42)

    # Model Farkı ve Karışım
    diff = 0.0
    for k in ["1", "X", "2", "O2.5", "BTTS"]:
        d = abs(poisson_market.get(k, 0) - mc["p"].get(k, 0))
        if d > diff: diff = d
    
    blended = blend_probs(poisson_market, mc["p"], alpha=BLEND_ALPHA)

    # Korner
    corner_analysis = analyze_corners(home_prev_stats, away_prev_stats, h2h_used)

    # Oran Analizi
    if not odds:
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
        value_block["decision"] = final_decision(qualified, diff)

    # Çıktı Paketi
    data = {
        "url": h2h_url,
        "teams": {"home": home_team, "away": away_team},
        "league": league_name,
        "lambda": {"home": lam_home, "away": lam_away, "total": lam_home + lam_away, "info": lambda_info},
        "poisson": {"market_probs": poisson_market, "top7_scores": [(f"{h}-{a}", p) for (h, a), p in top7]},
        "mc": mc,
        "model_agreement": {"diff": diff},
        "blended_probs": blended,
        "corner_analysis": corner_analysis,
        "value_bets": value_block,
        "data_sources": {
            "standings_used": len(st_home_rows) > 0 and len(st_away_rows) > 0,
            "h2h_matches": len(h2h_used),
            "h2h_same_league_used": h2h_same_used,
            "home_prev_matches": len(prev_home_sel),
            "away_prev_matches": len(prev_away_sel),
            "league_found": bool(league_name)
        }
    }

    data["report_comprehensive"] = format_comprehensive_report(data)
    return data

# ======================
# FLASK API ROUTING
# ======================
app = Flask(__name__)

@app.get("/")
def root():
    return jsonify({"ok": True, "service": "NowGoal-Analyzer-v4.3", "status": "Active"})

@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "healthy"})

@app.post("/analiz_et")
def analiz_et_route():
    """Android uygulaması için basitleştirilmiş endpoint."""
    try:
        payload = request.get_json(silent=True) or {}
    except Exception as e:
        return jsonify({"ok": False, "error": f"JSON Hatası: {e}"}), 400

    url = (payload.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "URL gerekli"}), 400

    if not re.match(r'^https?://', url):
        return jsonify({"ok": False, "error": "Geçersiz URL formatı"}), 400

    try:
        data = analyze_nowgoal(url, odds=None, mc_runs=10_000)

        top_skor = data["poisson"]["top7_scores"][0][0]
        blend = data["blended_probs"]

        p_o25 = blend.get("O2.5", 0)
        p_u25 = blend.get("U2.5", 0)
        p_btts = blend.get("BTTS", 0)

        alt_ust = f"ÜST %{p_o25*100:.1f}" if p_o25 >= p_u25 else f"ALT %{p_u25*100:.1f}"
        kg_var = f"VAR %{p_btts*100:.1f}" if p_btts >= 0.5 else f"YOK %{(1-p_btts)*100:.1f}"

        return jsonify({
            "ok": True,
            "skor": top_skor,
            "alt_ust": alt_ust,
            "btts": kg_var,
            "karar": data["value_bets"].get("decision", "Oran verisi yok"),
            "odds_used": data["value_bets"].get("used_odds", False),
            "odds": (data.get("value_bets", {}).get("table", None)),
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
    """Tam JSON çıktısı veren endpoint."""
    try:
        payload = request.get_json(silent=True) or {}
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid JSON: {e}"}), 400

    url = (payload.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "url required"}), 400

    odds = payload.get("odds")
    mc_runs = payload.get("mc_runs", MC_RUNS_DEFAULT)

    try:
        data = analyze_nowgoal(url, odds=odds, mc_runs=int(mc_runs))
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
        print("NowGoal Analyzer v4.3 (UNABRIDGED)")
        print("Usage: python script.py serve")
