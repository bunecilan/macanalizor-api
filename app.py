# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - Enhanced Version
Flask API for Android App with Advanced Analysis
"""

import re
import math
import json
import time
import traceback
from dataclasses import dataclass, asdict
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

# Ağırlıklar artık standings'e daha fazla önem veriyor
W_ST_BASE = 0.50  # Artırıldı: 0.55 → 0.50
W_FORM_BASE = 0.30  # Azaltıldı: 0.35 → 0.30
W_H2H_BASE = 0.10
W_LAST6_BASE = 0.10  # YENİ: Last 6 ağırlığı

BLEND_ALPHA = 0.50
VALUE_MIN = 0.05
PROB_MIN = 0.55
KELLY_MIN = 0.02  # YENİ: Kelly minimum %2
MAX_GOALS_FOR_MATRIX = 5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
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
    # YENİ: Clean sheet & BTTS istatistikleri
    clean_sheets: int = 0
    scored_matches: int = 0

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
        cleaned = [c for c in cleaned if c and c != "—"]
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
# FETCH
# ======================
def safe_get(url: str, timeout: int = 25, retries: int = 2) -> str:
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
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
        key = (m.league, m.date, m.home, m.away, m.ft_home, m.ft_away, m.ht_home, m.ht_away)
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
# MATCH PARSE
# ======================
def parse_match_from_cells(cells: List[str]) -> Optional[MatchRow]:
    date_idx = None
    date_val = None
    for i, c in enumerate(cells):
        d = normalize_date(c)
        if d:
            date_idx = i
            date_val = d
            break
    if date_idx is None or not date_val:
        return None

    score_idx = None
    score_m = None
    for i, c in enumerate(cells):
        c0 = (c or "").strip()
        if normalize_date(c0):
            continue
        m = SCORE_RE.search(c0)
        if not m:
            continue
        score_idx = i
        score_m = m
        break
    if score_idx is None or score_m is None:
        return None

    ft_h = int(score_m.group(1))
    ft_a = int(score_m.group(2))
    ht_h = int(score_m.group(3)) if score_m.group(3) else None
    ht_a = int(score_m.group(4)) if score_m.group(4) else None

    if score_idx - 1 < 0 or score_idx + 1 >= len(cells):
        return None
    home = cells[score_idx - 1].strip()
    away = cells[score_idx + 1].strip()
    if not home or not away:
        return None

    league = cells[0].strip() if cells else "—"
    return MatchRow(league=league, date=date_val, home=home, away=away,
                    ft_home=ft_h, ft_away=ft_a, ht_home=ht_h, ht_away=ht_a)

def parse_matches_from_table_html(table_html: str) -> List[MatchRow]:
    out: List[MatchRow] = []
    rows = extract_table_rows_from_html(table_html)
    for cells in rows:
        m = parse_match_from_cells(cells)
        if m:
            out.append(m)
    return sort_matches_desc(dedupe_matches(out))

# ======================
# STANDINGS (GELİŞTİRİLMİŞ)
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
    candidates: List[Tuple[int, List[StandRow]]] = []

    for tbl in extract_tables_html(page_source):
        text_low = strip_tags(tbl).lower()
        if not all(k in text_low for k in ["matches", "win", "draw", "loss", "scored", "conceded"]):
            continue
        rows = extract_table_rows_from_html(tbl)
        parsed = parse_standings_table_rows(rows)
        if not parsed:
            continue
        if team_key and team_key in norm_key(strip_tags(tbl)):
            candidates.append((len(candidates), parsed))

    if candidates:
        return candidates[0][1]
    return []

def standings_to_splits(rows: List[StandRow]) -> Dict[str, Optional[SplitGFGA]]:
    mp: Dict[str, Optional[SplitGFGA]] = {"Total": None, "Home": None, "Away": None, "Last 6": None}
    for r in rows:
        if r.matches and r.scored is not None and r.conceded is not None:
            mp[r.ft] = SplitGFGA(r.matches, r.scored, r.conceded)
    return mp

# YENİ: Clean Sheet ve BTTS hesaplama
def calculate_btts_stats(rows: List[StandRow]) -> Dict[str, Any]:
    stats = {}
    for r in rows:
        if r.ft in ["Total", "Home", "Away"] and r.matches and r.scored is not None and r.conceded is not None:
            clean_sheets = 0  # Yaklaşık hesaplama
            scored_matches = r.matches if r.scored > 0 else 0
            
            # Clean sheet tahmini (eğer gol yeme ortalaması düşükse)
            if r.matches > 0:
                ga_ratio = r.conceded / r.matches
                if ga_ratio < 0.5:
                    clean_sheets = int(r.matches * (1 - ga_ratio))
            
            stats[r.ft] = {
                "clean_sheets": clean_sheets,
                "scored_matches": scored_matches,
                "clean_sheet_rate": clean_sheets / r.matches if r.matches else 0,
                "scored_rate": scored_matches / r.matches if r.matches else 0
            }
    return stats

# ======================
# PREVIOUS & H2H
# ======================
def extract_previous_from_page(page_source: str) -> Tuple[List[str], List[str]]:
    tabs = section_tables_by_marker(page_source, "Previous Scores Statistics", max_tables=6)
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
    markers = ["Head to Head Statistics", "Head to Head", "H2H Statistics", "H2H"]
    for mk in markers:
        tabs = section_tables_by_marker(page_source, mk, max_tables=4)
        for t in tabs:
            cand = parse_matches_from_table_html(t)
            pair_count = sum(1 for m in cand if is_h2h_pair(m, home_team, away_team))
            if pair_count >= 3:
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

# YENİ: Same League Filtresi
def filter_same_league_matches(matches: List[MatchRow], league_name: str) -> List[MatchRow]:
    """Sadece aynı ligdeki maçları filtrele"""
    league_key = norm_key(league_name)
    return [m for m in matches if norm_key(m.league) == league_key]

# ======================
# PREV STATS (GELİŞTİRİLMİŞ)
# ======================
def build_prev_stats(team: str, matches: List[MatchRow]) -> TeamPrevStats:
    tkey = norm_key(team)
    st = TeamPrevStats(name=team)
    if not matches:
        return st

    def team_gf_ga(m: MatchRow) -> Tuple[int, int]:
        if norm_key(m.home) == tkey:
            return m.ft_home, m.ft_away
        return m.ft_away, m.ft_home

    gfs, gas = [], []
    clean_sheets = 0
    scored_matches = 0
    
    for m in matches:
        gf, ga = team_gf_ga(m)
        gfs.append(gf)
        gas.append(ga)
        if ga == 0:
            clean_sheets += 1
        if gf > 0:
            scored_matches += 1

    st.n_total = len(matches)
    st.gf_total = sum(gfs) / st.n_total if st.n_total else 0.0
    st.ga_total = sum(gas) / st.n_total if st.n_total else 0.0
    st.clean_sheets = clean_sheets
    st.scored_matches = scored_matches

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
# ODDS EXTRACTION (YENİ)
# ======================
def extract_bet365_initial_odds(page_source: str) -> Optional[Dict[str, float]]:
    """Bet365 Initial (1X2) oranlarını çıkar"""
    try:
        # Bet365 satırını bul
        bet365_pattern = r'Bet365.*?Initial.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)'
        match = re.search(bet365_pattern, page_source, re.DOTALL)
        
        if match:
            return {
                "1": float(match.group(1)),
                "X": float(match.group(2)),
                "2": float(match.group(3))
            }
        
        # Alternatif: HTML table'dan çıkar
        tables = extract_tables_html(page_source)
        for table in tables:
            if "bet365" in table.lower() and "initial" in table.lower():
                rows = extract_table_rows_from_html(table)
                for row in rows:
                    if len(row) >= 4 and "bet365" in row[0].lower():
                        try:
                            return {
                                "1": float(row[1]),
                                "X": float(row[2]),
                                "2": float(row[3])
                            }
                        except (ValueError, IndexError):
                            continue
        
        return None
    except Exception as e:
        print(f"Odds extraction error: {e}")
        return None

# ======================
# LAMBDA (GELİŞTİRİLMİŞ)
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
        "formula": "lam_h=(home_home.gf_pg + away_away.ga_pg)/2",
    }
    return lam_h, lam_a, meta

# YENİ: Last 6 Component
def compute_component_last6(st_home: Dict[str, Optional[SplitGFGA]],
                            st_away: Dict[str, Optional[SplitGFGA]]) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    h6 = st_home.get("Last 6")
    a6 = st_away.get("Last 6")
    if not h6 or not a6 or h6.matches < 4 or a6.matches < 4:
        return None
    lam_h = (h6.gf_pg + a6.ga_pg) / 2.0
    lam_a = (a6.gf_pg + h6.ga_pg) / 2.0
    meta = {
        "home_last6": {"matches": h6.matches, "gf_pg": h6.gf_pg, "ga_pg": h6.ga_pg},
        "away_last6": {"matches": a6.matches, "gf_pg": a6.gf_pg, "ga_pg": a6.ga_pg},
        "formula": "Last 6 matches form",
    }
    return lam_h, lam_a, meta

def compute_component_form(home_prev: TeamPrevStats, away_prev: TeamPrevStats) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    if home_prev.n_total < 3 or away_prev.n_total < 3:
        return None

    h_gf_home = home_prev.gf_home if home_prev.n_home >= 2 else home_prev.gf_total
    h_ga_home = home_prev.ga_home if home_prev.n_home >= 2 else home_prev.ga_total
    a_gf_away = away_prev.gf_away if away_prev.n_away >= 2 else away_prev.gf_total
    a_ga_away = away_prev.ga_away if away_prev.n_away >= 2 else away_prev.ga_total

    lam_h = (h_gf_home + a_ga_away) / 2.0
    lam_a = (a_gf_away + h_ga_home) / 2.0
    meta = {
        "home_prev": asdict(home_prev),
        "away_prev": asdict(away_prev),
        "formula": "lam_h=(home.gf_home + away.ga_away)/2",
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
    info: Dict[str, Any] = {
        "xg_available": False,
        "components": {},
        "weights_used": {},
        "warnings": []
    }

    comps: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}

    stc = compute_component_standings(st_home_s, st_away_s)
    if stc:
        comps["standings"] = stc
    
    # YENİ: Last 6 component
    l6c = compute_component_last6(st_home_s, st_away_s)
    if l6c:
        comps["last6"] = l6c
    
    frc = compute_component_form(home_prev, away_prev)
    if frc:
        comps["form"] = frc
    
    h2c = compute_component_h2h(h2h_used, home_team, away_team)
    if h2c:
        comps["h2h"] = h2c

    w = {}
    if "standings" in comps: w["standings"] = W_ST_BASE
    if "last6" in comps:      w["last6"] = W_LAST6_BASE
    if "form" in comps:       w["form"] = W_FORM_BASE
    if "h2h" in comps:        w["h2h"] = W_H2H_BASE

    w_norm = normalize_weights(w)
    info["weights_used"] = w_norm

    if not w_norm:
        info["warnings"].append("Veri yetersiz → λ=1.20/1.20 (düşük güven)")
        lh, la = 1.20, 1.20
    else:
        lh = 0.0; la = 0.0
        for k, wk in w_norm.items():
            ch, ca, meta = comps[k]
            info["components"][k] = {"lam_home": ch, "lam_away": ca, "meta": meta}
            lh += wk * ch
            la += wk * ca

    lh, la, clamp_warn = clamp_lambda(lh, la)
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

def team_goal_probs(lam: float, max_k: int = 5) -> Dict[str, float]:
    ps = {str(k): poisson_pmf(k, lam) for k in range(0, max_k + 1)}
    p0 = ps["0"]; p1 = ps["1"]; p2 = ps["2"]
    ps["3+"] = max(0.0, 1.0 - (p0 + p1 + p2))
    return ps

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

    out["H_O0.5"] = sum(p for (h, a), p in mat.items() if h >= 1)
    out["H_O1.5"] = sum(p for (h, a), p in mat.items() if h >= 2)
    out["A_O0.5"] = sum(p for (h, a), p in mat.items() if a >= 1)
    out["A_O1.5"] = sum(p for (h, a), p in mat.items() if a >= 2)
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
        return d, "Çok iyi uyum"
    elif d <= 0.06:
        return d, "İyi uyum"
    elif d <= 0.10:
        return d, "Orta uyum"
    return d, "Zayıf uyum"

def blend_probs(p1: Dict[str, float], p2: Dict[str, float], alpha: float) -> Dict[str, float]:
    out = {}
    for k in set(list(p1.keys()) + list(p2.keys())):
        if k in p1 and k in p2:
            out[k] = alpha * p1[k] + (1.0 - alpha) * p2[k]
        else:
            out[k] = p1.get(k, p2.get(k, 0.0))
    return out

# ======================
# KELLY CRITERION (YENİ)
# ======================
def kelly_criterion(prob: float, odds: float) -> float:
    """
    Kelly formülü: f* = (bp - q) / b
    b = odds - 1 (net kazanç)
    p = olasılık
    q = 1 - p
    """
    if odds <= 1.0 or prob <= 0.0 or prob >= 1.0:
        return 0.0
    
    b = odds - 1.0
    q = 1.0 - prob
    kelly = (b * prob - q) / b
    
    # Negatif kelly'yi 0 yap
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
# BTTS ENHANCED (YENİ)
# ======================
def compute_btts_enhanced(home_prev: TeamPrevStats, away_prev: TeamPrevStats,
                          home_btts: Dict[str, Any], away_btts: Dict[str, Any],
                          poisson_btts: float) -> Tuple[float, Dict[str, Any]]:
    """
    Standings BTTS verilerini kullanarak tahmini güçlendir
    """
    # Clean sheet ve scored rates
    home_scored_rate = home_btts.get("Home", {}).get("scored_rate", 0.5)
    away_scored_rate = away_btts.get("Away", {}).get("scored_rate", 0.5)
    home_clean_rate = home_btts.get("Home", {}).get("clean_sheet_rate", 0.3)
    away_clean_rate = away_btts.get("Away", {}).get("clean_sheet_rate", 0.3)
    
    # Form-based BTTS
    form_btts = home_scored_rate * away_scored_rate
    
    # Blend: %60 Poisson, %40 Form
    final_btts = 0.6 * poisson_btts + 0.4 * form_btts
    
    meta = {
        "poisson_btts": poisson_btts,
        "form_btts": form_btts,
        "home_scored_rate": home_scored_rate,
        "away_scored_rate": away_scored_rate,
        "home_clean_rate": home_clean_rate,
        "away_clean_rate": away_clean_rate,
    }
    
    return final_btts, meta

# ======================
# REPORTING
# ======================
def determine_tempo(total_lam: float) -> str:
    if total_lam < 2.3:
        return "Düşük"
    if total_lam < 2.9:
        return "Orta"
    return "Yüksek"

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
        return f"OYNAMA (eşik sağlanmadı, uyum: {diff_label})"
    if diff > 0.10:
        return f"TEMKİNLİ (zayıf uyum: {diff_label})"
    best = sorted(qualified, key=lambda x: x[3], reverse=True)[0]
    mkt, prob, odds, val, qk = best
    return f"OYNANABİLİR → {mkt} (p={prob*100:.1f}%, oran={odds:.2f}, value={val:+.3f}, kelly={qk*100:.1f}%)"

def format_report_short(data: Dict[str, Any]) -> str:
    """Android için kısa özet (GELİŞTİRİLMİŞ)"""
    t = data["teams"]
    lh = data["lambda"]["home"]; la = data["lambda"]["away"]
    total = lh + la

    top7 = data["poisson"]["top7_scores"]
    blend = data["blended_probs"]

    net_ou, net_ou_p, net_ou_c = net_ou_prediction(blend)
    net_btts, net_btts_p, net_btts_c = net_btts_prediction(blend)

    lines = []
    lines.append(f"🏠 {t['home']} vs 🚶 {t['away']}")
    lines.append(f"\n⚽ Beklenen Goller:")
    lines.append(f"  Ev: {lh:.2f} | Dep: {la:.2f} | Top: {total:.2f}")
    
    # Top 3 skorları emoji ile göster
    lines.append(f"\n🎯 En Olası Skorlar:")
    for i in range(min(3, len(top7))):
        emoji = ["1️⃣", "2️⃣", "3️⃣"][i]
        lines.append(f"  {emoji} {top7[i][0]} ({top7[i][1]*100:.1f}%)")
    
    lines.append(f"\n📊 Tahminler:")
    
    # Alt/Üst tahmini - güven seviyesi göster
    ou_emoji = "✅" if net_ou_c == "Yüksek" else "⚠️" if net_ou_c == "Orta" else "❓"
    lines.append(f"  Alt/Üst: {net_ou} ({net_ou_p*100:.1f}%) {ou_emoji}")
    
    # BTTS tahmini - güven seviyesi göster
    btts_emoji = "✅" if net_btts_c == "Yüksek" else "⚠️" if net_btts_c == "Orta" else "❓"
    lines.append(f"  BTTS: {net_btts} ({net_btts_p*100:.1f}%) {btts_emoji}")
    
    # Tempo bilgisi
    tempo = determine_tempo(total)
    tempo_emoji = "🔥" if tempo == "Yüksek" else "⚖️" if tempo == "Orta" else "🐌"
    lines.append(f"  Tempo: {tempo} {tempo_emoji}")
    
    vb = data.get("value_bets")
    if vb and vb.get("used_odds"):
        lines.append(f"\n💰 Karar: {vb['decision']}")
        # Kelly önerileri
        if vb.get("table"):
            qualified = [t for t in vb["table"] if t["kelly"] >= KELLY_MIN]
            if qualified:
                best = max(qualified, key=lambda x: x["kelly"])
                lines.append(f"💸 En İyi Bahis: {best['market']} (Kelly: {best['kelly']*100:.1f}%)")
    
    return "\n".join(lines)

# ======================
# MAIN ANALYSIS (GELİŞTİRİLMİŞ)
# ======================
def analyze_nowgoal(url: str, odds: Optional[Dict[str, float]] = None, mc_runs: int = MC_RUNS_DEFAULT) -> Dict[str, Any]:
    h2h_url = build_h2h_url(url)
    html = safe_get(h2h_url)

    home_team, away_team = parse_teams_from_title(html)
    if not home_team or not away_team:
        raise RuntimeError("Takım isimleri çıkarılamadı")

    # League bilgisini çıkar
    league_match = re.search(r'<span[^>]*class=["\']?sclassLink["\']?[^>]*>([^<]+)</span>', html)
    league_name = strip_tags(league_match.group(1)) if league_match else ""

    # Standings
    st_home_rows = extract_standings_for_team(html, home_team)
    st_away_rows = extract_standings_for_team(html, away_team)
    st_home = standings_to_splits(st_home_rows)
    st_away = standings_to_splits(st_away_rows)
    
    # BTTS stats from standings
    home_btts_stats = calculate_btts_stats(st_home_rows)
    away_btts_stats = calculate_btts_stats(st_away_rows)

    # H2H
    h2h_all = extract_h2h_matches(html, home_team, away_team)
    h2h_pair = [m for m in h2h_all if is_h2h_pair(m, home_team, away_team)]
    h2h_used = sort_matches_desc(dedupe_matches(h2h_pair))[:H2H_N]

    # Previous Scores (SAME LEAGUE FILTERED)
    prev_home_tabs, prev_away_tabs = extract_previous_from_page(html)
    prev_home = parse_matches_from_table_html(prev_home_tabs[0])[:RECENT_N] if prev_home_tabs else []
    prev_away = parse_matches_from_table_html(prev_away_tabs[0])[:RECENT_N] if prev_away_tabs else []
    
    # Same League filtresi uygula
    if league_name:
        prev_home = filter_same_league_matches(prev_home, league_name)
        prev_away = filter_same_league_matches(prev_away, league_name)

    home_prev_stats = build_prev_stats(home_team, prev_home)
    away_prev_stats = build_prev_stats(away_team, prev_away)

    # Lambda hesaplama (enhanced)
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

    # Model agreement
    diff, diff_label = model_agreement(poisson_market, mc["p"])
    blended = blend_probs(poisson_market, mc["p"], alpha=BLEND_ALPHA)
    
    # Enhanced BTTS
    btts_enhanced, btts_meta = compute_btts_enhanced(
        home_prev_stats, away_prev_stats,
        home_btts_stats, away_btts_stats,
        blended.get("BTTS", 0)
    )
    blended["BTTS"] = btts_enhanced

    # Odds extraction (Bet365 Initial)
    if not odds:
        odds = extract_bet365_initial_odds(html)

    # Value bets & Kelly
    value_block = {"used_odds": False}
    qualified = []

    if odds and all(k in odds for k in ["1", "X", "2"]):
        value_block["used_odds"] = True
        table = []
        for mkt in ["1", "X", "2"]:
            o = float(odds[mkt])
            p = float(blended.get(mkt, 0.0))
            v, kelly = value_and_kelly(p, o)
            qk = max(0.0, 0.25 * kelly)  # Fraksiyonel Kelly (25%)
            row = {
                "market": mkt, 
                "prob": p, 
                "odds": o, 
                "value": v, 
                "kelly": kelly, 
                "qkelly": qk
            }
            table.append(row)
            if v >= VALUE_MIN and p >= PROB_MIN and kelly >= KELLY_MIN:
                qualified.append((mkt, p, o, v, qk))

        value_block["table"] = table
        value_block["thresholds"] = {
            "value_min": VALUE_MIN, 
            "prob_min": PROB_MIN,
            "kelly_min": KELLY_MIN
        }
        value_block["decision"] = final_decision(qualified, diff, diff_label)

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
        "poisson": {
            "market_probs": poisson_market,
            "top7_scores": top7
        },
        "mc": mc,
        "model_agreement": {"diff": diff, "label": diff_label},
        "blended_probs": blended,
        "btts_enhanced": {
            "prob": btts_enhanced,
            "meta": btts_meta
        },
        "value_bets": value_block,
        "data_sources": {
            "standings_used": len(st_home_rows) > 0 and len(st_away_rows) > 0,
            "last6_used": st_home.get("Last 6") is not None and st_away.get("Last 6") is not None,
            "h2h_matches": len(h2h_used),
            "home_prev_matches": len(prev_home),
            "away_prev_matches": len(prev_away),
            "same_league_filtered": bool(league_name)
        }
    }

    data["report_short"] = format_report_short(data)
    return data

# ======================
# FLASK APP
# ======================
app = Flask(__name__)

@app.get("/")
def root():
    return jsonify({"ok": True, "service": "macanalizor-api", "version": "3.0-enhanced"})

@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "healthy"})

@app.post("/analiz_et")
def analiz_et_route():
    """Android uygulaması için endpoint"""
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
        
        # Android için basitleştirilmiş response
        top_skor = data["poisson"]["top7_scores"][0][0]
        
        return jsonify({
            "ok": True,
            "skor": top_skor,
            "detay": data["report_short"]
        })
        
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Analiz hatası: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.post("/analyze")
def analyze_route():
    """Web/API için detaylı endpoint"""
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
        print("CLI mode - use 'serve' argument for Flask server")
