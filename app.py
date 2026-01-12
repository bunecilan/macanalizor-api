# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - Enhanced Version 4.1 (ODDS FIX)
Flask API with Corner Analysis & Enhanced Value Betting
- Bet365 odds: tries HTML first, then /component/matchoddsdetailwin endpoint (user-provided)
"""

import re
import math
import json
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
RECENT_N = 10         # Previous Scores: maksimum 10 maç
H2H_N = 10            # H2H: maksimum 10 maç

# Ağırlıklar - EN ÖNEMLİDEN EN AZ ÖNEMLİYE
W_ST_BASE = 0.45      # Standing
W_PSS_BASE = 0.30     # Previous Scores Statistics
W_H2H_BASE = 0.25     # Head to Head

BLEND_ALPHA = 0.50
VALUE_MIN = 0.05      # Minimum %5 value
PROB_MIN = 0.55       # Minimum %55 probability
KELLY_MIN = 0.02      # Minimum %2 Kelly

# 0-5 yerine 0-8: daha az "tail cut" hatası
MAX_GOALS_FOR_MATRIX = 8

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# ======================
# REGEX
# ======================
DATE_ANY_RE = re.compile(r"\b(\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{2}-\d{2})\b")
SCORE_RE = re.compile(r"\b(\d{1,2})\s*-\s*(\d{1,2})(?!-\d{2,4})(?:\s*\((\d{1,2})\s*-\s*(\d{1,2})\))?\b")

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
def strip_tags(s: str) -> str:
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
        cleaned = [strip_tags(c) for c in cells]
        cleaned = [c for c in cleaned if c and c not in {"—", "-"}]
        if cleaned:
            rows.append(cleaned)
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
# FETCH (Session + Retry)
# ======================
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s

def safe_get(session: requests.Session, url: str, timeout: int = 25, retries: int = 2, headers: Optional[Dict[str, str]] = None) -> str:
    last_err = None
    for _ in range(retries + 1):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
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

def parse_teams_from_title(html: str) -> Tuple[str, str]:
    m = re.search(r"<title>\s*(.*?)\s*</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = strip_tags(m.group(1)) if m else ""
    mm = re.search(r"(.+?)\s+VS\s+(.+?)(?:\s+-|\s+\||$)", title, flags=re.IGNORECASE)
    if not mm:
        mm = re.search(r"(.+?)\s+vs\s+(.+?)(?:\s+-|\s+\||$)", title, flags=re.IGNORECASE)
    if not mm:
        return "", ""
    return mm.group(1).strip(), mm.group(2).strip()

def sort_matches_desc(matches: List[MatchRow]) -> List[MatchRow]:
    return sorted(matches, key=lambda x: parse_date_key(x.date), reverse=True)

def dedupe_matches(matches: List[MatchRow]) -> List[MatchRow]:
    seen = set()
    out = []
    for m in matches:
        key = (m.league, m.date, m.home, m.away, m.ft_home, m.ft_away)
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
# CORNER EXTRACTION
# ======================
def extract_corners_from_cell(cell: str) -> Optional[Tuple[int, int]]:
    """
    Hücrede korner verisi varsa çıkar.
    Formatlar: "5-3", "(5-3)", "6-2(5-0)"
    Not: Bu heuristic; NowGoal'da korner bazen ayrı sütunda gelir.
    """
    if not cell:
        return None

    # Parantez içi genelde HT veya başka bilgi olabilir. Korner için de kullanılabiliyor.
    m = re.search(r'\((\d{1,2})\s*-\s*(\d{1,2})\)', cell)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        # HT ile çakışmaması için kaba filtre: 0-0 / 1-0 gibi çok küçük HT değerleri olabilir.
        # Ama korner de küçük olabilir; bu yüzden burada kesin hüküm vermiyoruz.
        return a, b

    all_pairs = re.findall(r'(\d{1,2})\s*-\s*(\d{1,2})', cell)
    if len(all_pairs) >= 2:
        return int(all_pairs[1][0]), int(all_pairs[1][1])

    return None

# ======================
# MATCH PARSE
# ======================
def parse_match_from_cells(cells: List[str]) -> Optional[MatchRow]:
    if not cells or len(cells) < 4:
        return None

    # Tarih bul
    date_val = None
    date_idx = None
    for i, c in enumerate(cells):
        d = normalize_date(c)
        if d:
            date_idx = i
            date_val = d
            break
    if not date_val:
        return None

    # Skor bul
    score_idx = None
    score_m = None
    for i, c in enumerate(cells):
        if i == date_idx:
            continue
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

    # Ev/Dep takım
    home = None
    away = None
    for i in range(score_idx - 1, -1, -1):
        if cells[i] and cells[i] != date_val:
            home = cells[i].strip()
            break
    for i in range(score_idx + 1, len(cells)):
        if cells[i]:
            away = cells[i].strip()
            break
    if not home or not away:
        return None

    league = cells[0].strip() if cells[0] and cells[0] != date_val else "—"

    # Korner ara: skor hücresi + sonraki hücreler
    corner_home, corner_away = None, None
    corners = extract_corners_from_cell(cells[score_idx])
    if corners:
        corner_home, corner_away = corners
    if not corners:
        for i in range(score_idx + 1, min(score_idx + 6, len(cells))):
            corners = extract_corners_from_cell(cells[i])
            if corners:
                corner_home, corner_away = corners
                break

    return MatchRow(
        league=league, date=date_val, home=home, away=away,
        ft_home=ft_h, ft_away=ft_a, ht_home=ht_h, ht_away=ht_a,
        corner_home=corner_home, corner_away=corner_away
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
        head = cells[0].strip()
        if head not in wanted:
            continue
        label = "Last 6" if head == "Last6" else head

        def g(i): return cells[i] if i < len(cells) else ""

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
            rate=g(9) if g(9) else None
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
        text_low = strip_tags(tbl).lower()

        required_keywords = ["matches", "win", "draw", "loss", "scored", "conceded"]
        if not all(k in text_low for k in required_keywords):
            continue

        if team_key and team_key not in norm_key(strip_tags(tbl)):
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
# ODDS EXTRACTION (HTML + COMPONENT)
# ======================
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", ".")
        if not re.match(r"^\d+(\.\d+)?$", s):
            return None
        return float(s)
    except Exception:
        return None

def extract_bet365_initial_odds_from_text(text: str) -> Optional[Dict[str, float]]:
    """
    Bet365 'Initial' 1X2 oranlarını mümkün olduğunca sağlam şekilde yakala.
    text HTML snippet de olabilir, JSON string de olabilir.
    """
    if not text or len(text) < 20:
        return None

    # 1) JSON ise: kaba arama (bet365 + 3 odds)
    try:
        obj = json.loads(text)
        hit = _find_bet365_odds_in_json(obj)
        if hit:
            return hit
    except Exception:
        pass

    # 2) HTML/Metin regex: Bet365 yakınında 3 adet decimal odds ara
    # Bet365 ... Initial ... 1.90 ... 3.20 ... 4.10
    pat = re.compile(r"bet365.*?initial.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)", re.I | re.S)
    m = pat.search(text)
    if m:
        return {"1": float(m.group(1)), "X": float(m.group(2)), "2": float(m.group(3))}

    # 3) Bet365 satırı: Bet365'e yakın ilk 3 decimal
    # (Initial kelimesi yoksa da yakalamak için)
    idx = text.lower().find("bet365")
    if idx != -1:
        window = text[idx: idx + 2500]
        nums = re.findall(r"\b(\d+\.\d+)\b", window)
        # İlk 3 decimal çoğu yapıda 1X2 olur
        if len(nums) >= 3:
            a, b, c = map(float, nums[:3])
            # Basit sanity: oranlar 1.01 - 100 arası
            if 1.01 <= a <= 100 and 1.01 <= b <= 100 and 1.01 <= c <= 100:
                return {"1": a, "X": b, "2": c}

    return None

def _find_bet365_odds_in_json(obj: Any) -> Optional[Dict[str, float]]:
    """
    JSON yapısı siteye göre değişiyor. Bu fonksiyon:
    - stringlerde "bet365" geçen node bulur
    - o node çevresinde 3 adet uygun odds değerini yakalamaya çalışır
    """
    # Rekürsif dolaş
    stack = [obj]
    while stack:
        cur = stack.pop()

        if isinstance(cur, dict):
            # Bet365 belirtilmiş mi?
            joined = " ".join([str(k) + ":" + str(v) for k, v in cur.items()]).lower()
            if "bet365" in joined:
                # Olası alan isimleri
                candidates = []
                for key in ["home", "win", "h", "odds1", "o1", "w1", "a", "b", "c", "draw", "x", "oddsx", "odds2", "away", "lose", "l"]:
                    if key in cur:
                        f = _safe_float(cur.get(key))
                        if f is not None:
                            candidates.append(f)

                # Bazı JSON'larda odds array olur
                if "odds" in cur and isinstance(cur["odds"], (list, tuple)):
                    for v in cur["odds"]:
                        f = _safe_float(v)
                        if f is not None:
                            candidates.append(f)

                # En az 3 odds yakaladıysak ilk 3'ü al
                candidates = [x for x in candidates if 1.01 <= x <= 100]
                if len(candidates) >= 3:
                    return {"1": candidates[0], "X": candidates[1], "2": candidates[2]}

            for v in cur.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)

        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    stack.append(v)

    return None

def fetch_bet365_odds_via_component(session: requests.Session, base: str, match_id: str, referer: str) -> Optional[Dict[str, float]]:
    """
    Kullanıcı verdi: /component/matchoddsdetailwin
    Parametre isimleri NowGoal domain'lerine göre değişebiliyor.
    Bu yüzden birkaç olası parametre ile dener.
    """
    endpoint = f"{base}/component/matchoddsdetailwin"

    # En olası parametre kombinasyonları
    param_trials = [
        {"matchId": match_id},
        {"matchid": match_id},
        {"MatchId": match_id},
        {"id": match_id},
        {"mid": match_id},
        {"fid": match_id},
    ]

    # Bazı sistemler market/type ister; boş + birkaç tahmin
    extra_trials = [
        {},
        {"type": "1x2"},
        {"ot": "1x2"},
        {"otype": "1x2"},
        {"market": "1x2"},
        {"win": "1"},  # bazı domain'ler böyle saçma parametreler istiyor
    ]

    for p in param_trials:
        for extra in extra_trials:
            params = dict(p)
            params.update(extra)

            try:
                text = safe_get(
                    session,
                    endpoint + "?" + "&".join([f"{k}={v}" for k, v in params.items()]),
                    timeout=25,
                    retries=1,
                    headers={**HEADERS, "Referer": referer},
                )
            except Exception:
                continue

            odds = extract_bet365_initial_odds_from_text(text)
            if odds:
                return odds

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
        # fallback: maç tablosuna benzeyen ilk 2 tabloyu bul
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
    if not league_name:
        return matches
    league_key = norm_key(league_name)
    return [m for m in matches if norm_key(m.league) == league_key]

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
# CORNER ANALYSIS
# ======================
def analyze_corners(home_prev: TeamPrevStats, away_prev: TeamPrevStats, h2h_matches: List[MatchRow]) -> Dict[str, Any]:
    h2h_corners_total, h2h_corners_home, h2h_corners_away = [], [], []

    for m in h2h_matches[:H2H_N]:
        if m.corner_home is not None and m.corner_away is not None:
            h2h_corners_total.append(m.corner_home + m.corner_away)
            h2h_corners_home.append(m.corner_home)
            h2h_corners_away.append(m.corner_away)

    h2h_total_avg = sum(h2h_corners_total) / len(h2h_corners_total) if h2h_corners_total else 0.0
    h2h_home_avg = sum(h2h_corners_home) / len(h2h_corners_home) if h2h_corners_home else 0.0
    h2h_away_avg = sum(h2h_corners_away) / len(h2h_corners_away) if h2h_corners_away else 0.0

    pss_home_for = home_prev.corners_for
    pss_home_against = home_prev.corners_against
    pss_away_for = away_prev.corners_for
    pss_away_against = away_prev.corners_against

    if h2h_total_avg > 0:
        predicted_home_corners = 0.6 * h2h_home_avg + 0.4 * ((pss_home_for + pss_away_against) / 2)
        predicted_away_corners = 0.6 * h2h_away_avg + 0.4 * ((pss_away_for + pss_home_against) / 2)
    elif pss_home_for > 0 or pss_away_for > 0:
        predicted_home_corners = (pss_home_for + pss_away_against) / 2
        predicted_away_corners = (pss_away_for + pss_home_against) / 2
    else:
        predicted_home_corners = 0.0
        predicted_away_corners = 0.0

    total_corners = predicted_home_corners + predicted_away_corners

    # Basit line olasılıkları (heuristic)
    predictions = {}
    for line in [8.5, 9.5, 10.5, 11.5]:
        over_prob = 1.0 if total_corners > line else max(0.0, (total_corners - line + 1) / 2)
        predictions[f"O{line}"] = float(min(1.0, over_prob))
        predictions[f"U{line}"] = 1.0 - predictions[f"O{line}"]

    data_points = len(h2h_corners_total) + (1 if (pss_home_for > 0 or pss_away_for > 0) else 0)
    if data_points >= 8:
        confidence = "Yüksek"
    elif data_points >= 4:
        confidence = "Orta"
    else:
        confidence = "Düşük"

    # Net korner önerisi: en yüksek olasılıklı çizgiyi seç
    best_market = None
    best_p = -1.0
    for k, p in predictions.items():
        if p > best_p:
            best_p = p
            best_market = k

    return {
        "predicted_home_corners": round(predicted_home_corners, 1),
        "predicted_away_corners": round(predicted_away_corners, 1),
        "total_corners": round(total_corners, 1),
        "h2h_avg": round(h2h_total_avg, 1),
        "h2h_data_count": len(h2h_corners_total),
        "pss_data_available": (pss_home_for > 0 or pss_away_for > 0),
        "predictions": predictions,
        "net_corner_pick": {"market": best_market, "prob": round(best_p, 3)} if best_market else None,
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
    if home_prev.n_total < 3 or away_prev.n_total < 3:
        return None

    h_gf = home_prev.gf_home if home_prev.n_home >= 3 else home_prev.gf_total
    h_ga = home_prev.ga_home if home_prev.n_home >= 3 else home_prev.ga_total
    a_gf = away_prev.gf_away if away_prev.n_away >= 3 else away_prev.gf_total
    a_ga = away_prev.ga_away if away_prev.n_away >= 3 else away_prev.ga_total

    lam_h = (h_gf + a_ga) / 2.0
    lam_a = (a_gf + h_ga) / 2.0

    meta = {
        "home_matches": home_prev.n_total,
        "away_matches": away_prev.n_total,
        "home_gf": round(h_gf, 2),
        "away_gf": round(a_gf, 2),
        "formula": "PSS: (home_gf + away_ga) / 2"
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
t = stc
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
        info["warnings"].append("Yetersiz veri")
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
# POISSON & MC
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

def build_score_matrix(lh: float, la: float, max_g: int = 8) -> Dict[Tuple[int, int], float]:
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
    for k in range(0, 7):
        total_bins[str(k)] = dist_total.get(k, 0) / n * 100.0
    total_bins["7+"] = sum(v for kk, v in dist_total.items() if kk >= 7) / n * 100.0

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
    lines.append(f"  Alt Skor: {top7[1][0]}, {top7[2][0]}")

    net_ou, net_ou_p, _ = net_ou_prediction(blend)
    lines.append(f"\nAlt/Üst 2.5: {net_ou} (%{net_ou_p*100:.1f})")

    net_btts, net_btts_p, _ = net_btts_prediction(blend)
    lines.append(f"KG Var: {net_btts} (%{net_btts_p*100:.1f})")

    lines.append(f"\n1X2 Olasılıkları:")
    lines.append(f"  Ev (1): %{blend.get('1', 0)*100:.1f}")
    lines.append(f"  Ber(X): %{blend.get('X', 0)*100:.1f}")
    lines.append(f"  Dep(2): %{blend.get('2', 0)*100:.1f}")

    # KORNER + net korner alt/üst
    corners = data.get("corner_analysis", {})
    if corners and corners.get("total_corners", 0) > 0:
        lines.append(f"\nKorner Tahmini: {corners['total_corners']}")
        lines.append(f"  (Ev: {corners['predicted_home_corners']} | Dep: {corners['predicted_away_corners']})")
        nc = corners.get("net_corner_pick")
        if nc and nc.get("market"):
            # örn: U9.5 -> "9.5 ALT"
            mk = nc["market"]
            prob = nc["prob"]
            pretty = mk
            if mk.startswith("O"):
                pretty = f"{mk[1:]} ÜST"
            elif mk.startswith("U"):
                pretty = f"{mk[1:]} ALT"
            lines.append(f"  NET KORNER: {pretty} (%{prob*100:.1f})")

    vb = data.get("value_bets", {})
    if vb.get("used_odds"):
        lines.append(f"\nBAHİS ANALİZİ:")
        has_value = False
        for row in vb.get("table", []):
            if row["value"] >= VALUE_MIN and row["prob"] >= PROB_MIN:
                lines.append(f"  ✅ {row['market']}: Oran {row['odds']:.2f} | Value %{row['value']*100:+.1f}")
                has_value = True
        if not has_value:
            lines.append(f"  ⚠️  Değerli bahis bulunamadı")
        lines.append(f"\n  KARAR: {vb.get('decision', 'Analiz edilemedi')}")
    else:
        lines.append(f"\nOran verisi yok - value analizi yapılamadı")

    ds = data["data_sources"]
    lambda_info = data["lambda"]["info"]

    lines.append(f"\nKullanılan Veriler:")
    lines.append(f"  Standing: {'✓' if ds['standings_used'] else '✗'}")
    pss_home = ds['home_prev_matches']
    pss_away = ds['away_prev_matches']
    lines.append(f"  PSS: {'✓' if (pss_home>0 or pss_away>0) else '✗'} (Ev:{pss_home} | Dep:{pss_away})")
    h2h_count = ds['h2h_matches']
    lines.append(f"  H2H: {'✓' if h2h_count>0 else '✗'} ({h2h_count} maç)")

    if lambda_info.get("weights_used"):
        lines.append(f"\nAğırlıklar:")
        for k, v in lambda_info["weights_used"].items():
            k_name = {"standing": "Standing", "pss": "PSS", "h2h": "H2H"}.get(k, k)
            lines.append(f"  {k_name}: %{v*100:.0f}")

    lines.append("=" * 60)
    return "\n".join(lines)

# ======================
# MAIN ANALYSIS
# ======================
def analyze_nowgoal(url: str, odds: Optional[Dict[str, float]] = None, mc_runs: int = MC_RUNS_DEFAULT) -> Dict[str, Any]:
    session = make_session()

    h2h_url = build_h2h_url(url)
    base = extract_base_domain(h2h_url)
    match_id = extract_match_id(h2h_url)

    html = safe_get(session, h2h_url)

    home_team, away_team = parse_teams_from_title(html)
    if not home_team or not away_team:
        raise RuntimeError("Takım isimleri çıkarılamadı")

    # League
    league_match = re.search(r'<span[^>]*class=["\']?sclassLink["\']?[^>]*>([^<]+)</span>', html)
    league_name = strip_tags(league_match.group(1)) if league_match else ""

    # Standings
    st_home_rows = extract_standings_for_team(html, home_team)
    st_away_rows = extract_standings_for_team(html, away_team)
    st_home = standings_to_splits(st_home_rows)
    st_away = standings_to_splits(st_away_rows)

    # H2H
    h2h_all = extract_h2h_matches(html, home_team, away_team)
    h2h_pair = [m for m in h2h_all if is_h2h_pair(m, home_team, away_team)]
    h2h_used = sort_matches_desc(dedupe_matches(h2h_pair))[:H2H_N]

    # Previous
    prev_home_tabs, prev_away_tabs = extract_previous_from_page(html)
    prev_home_raw = parse_matches_from_table_html(prev_home_tabs[0]) if prev_home_tabs else []
    prev_away_raw = parse_matches_from_table_html(prev_away_tabs[0]) if prev_away_tabs else []
    prev_home_raw = prev_home_raw[:RECENT_N]
    prev_away_raw = prev_away_raw[:RECENT_N]

    prev_home = prev_home_raw
    prev_away = prev_away_raw

    if league_name:
        f_home = filter_same_league_matches(prev_home_raw, league_name)
        f_away = filter_same_league_matches(prev_away_raw, league_name)
        if len(f_home) >= 3: prev_home = f_home
        if len(f_away) >= 3: prev_away = f_away

    home_prev_stats = build_prev_stats(home_team, prev_home)
    away_prev_stats = build_prev_stats(away_team, prev_away)

    # Lambda
    lam_home, lam_away, lambda_info = compute_lambdas(
        st_home_s=st_home,
        st_away_s=st_away,
        home_prev=home_prev_stats,
        away_prev=away_prev_stats,
        h2h_used=h2h_used,
        home_team=home_team,
        away_team=away_team
    )

    # Poisson
    score_mat = build_score_matrix(lam_home, lam_away, max_g=MAX_GOALS_FOR_MATRIX)
    poisson_market = market_probs_from_matrix(score_mat)
    top7 = top_scores_from_matrix(score_mat, top_n=7)

    # Monte Carlo
    mc = monte_carlo(lam_home, lam_away, n=max(10_000, int(mc_runs)), seed=42)

    # Model agreement + blend
    diff, diff_label = model_agreement(poisson_market, mc["p"])
    blended = blend_probs(poisson_market, mc["p"], alpha=BLEND_ALPHA)

    # Corner
    corner_analysis = analyze_corners(home_prev_stats, away_prev_stats, h2h_used)

    # Odds
    odds_source = None
    if not odds:
        # 1) HTML içinden dene
        odds = extract_bet365_initial_odds_from_text(html)
        if odds:
            odds_source = "h2h_html"
        else:
            # 2) Component endpoint (kullanıcı verdi)
            odds = fetch_bet365_odds_via_component(session, base, match_id, referer=h2h_url)
            if odds:
                odds_source = "component_matchoddsdetailwin"

    # Value bets
    value_block = {"used_odds": False, "odds_source": odds_source}
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

    data = {
        "url": h2h_url,
        "match_id": match_id,
        "teams": {"home": home_team, "away": away_team},
        "league": league_name,
        "lambda": {
            "home": lam_home,
            "away": lam_away,
            "total": lam_home + lam_away,
            "info": lambda_info
        },
        "poisson": {"market_probs": poisson_market, "top7_scores": top7},
        "mc": mc,
        "model_agreement": {"diff": diff, "label": diff_label},
        "blended_probs": blended,
        "corner_analysis": corner_analysis,
        "value_bets": value_block,
        "data_sources": {
            "standings_used": len(st_home_rows) > 0 and len(st_away_rows) > 0,
            "h2h_matches": len(h2h_used),
            "home_prev_matches": len(prev_home),
            "away_prev_matches": len(prev_away),
            "same_league_filtered": bool(league_name)
        }
    }

    data["report_comprehensive"] = format_comprehensive_report(data)
    return data

# ======================
# FLASK APP
# ======================
app = Flask(__name__)

@app.get("/")
def root():
    return jsonify({"ok": True, "service": "nowgoal-analyzer-api", "version": "4.1-odds-fix"})

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

        corners = data.get("corner_analysis", {})
        net_corner = ""
        if corners and corners.get("net_corner_pick"):
            mk = corners["net_corner_pick"]["market"]
            pr = corners["net_corner_pick"]["prob"]
            if mk.startswith("O"):
                net_corner = f"{mk[1:]} ÜST (%{pr*100:.1f})"
            elif mk.startswith("U"):
                net_corner = f"{mk[1:]} ALT (%{pr*100:.1f})"

        return jsonify({
            "ok": True,
            "skor": top_skor,
            "alt_ust": f"{net_ou} (%{net_ou_p*100:.1f})",
            "btts": f"{net_btts} (%{net_btts_p*100:.1f})",
            "korner": net_corner or f"{corners.get('total_corners', 0)}",
            "karar": data["value_bets"].get("decision", "Oran gerekli"),
            "odds_source": data["value_bets"].get("odds_source"),
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

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        print("NowGoal Analyzer v4.1 (ODDS FIX)")
        print("Usage: python script.py serve")
        print("\nEndpoints:")
        print("  POST /analiz_et - Android app (Turkish)")
        print("  POST /analyze   - Full API (English)")
        print("\nExample request:")
        print('  {"url": "https://live3.nowgoal26.com/match/h2h-2784678"}')
