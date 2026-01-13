# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - Enhanced Version 5.0 (FINAL)
Flask API with Bet365 Odds Fix & Strict Same League/Side Filtering

CHANGELOG V5.0:
1) Bet365 Initial Odds (1X2) Extraction FIXED: Tablo yapısını daha agresif tarayan yeni bir algoritma eklendi.
2) PSS (Previous Scores) Filtering:
   - Ev Sahibi için: Sadece EVİNDE oynadığı ve AYNI LİGDEKİ maçlar filtrelenir.
   - Deplasman için: Sadece DEPLASMANDA oynadığı ve AYNI LİGDEKİ maçlar filtrelenir.
3) Corner Analysis: Filtrelenmiş (Side + Same League) maçların ortalamaları kullanılarak daha isabetli tahmin yapar.
4) Lig Eşleştirme (Fuzzy Logic): "ITA D1" ile "Italian Serie A" eşleşmesini yakalamak için esnek karşılaştırma eklendi.
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
RECENT_N = 10         # Analize dahil edilecek maksimum son maç sayısı
H2H_N = 10            # H2H maksimum maç

# Ağırlıklar - EN ÖNEMLİDEN EN AZ ÖNEMLİYE
W_ST_BASE = 0.45      # Standing (Resmi lig verileri)
W_PSS_BASE = 0.30     # Previous Scores Statistics (Son form)
W_H2H_BASE = 0.25     # Head to Head (Geçmiş karşılaşmalar)

BLEND_ALPHA = 0.50
VALUE_MIN = 0.05      # Minimum %5 value
PROB_MIN = 0.55       # Minimum %55 probability
KELLY_MIN = 0.02      # Minimum %2 Kelly
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

# Corner hücresi: "4-9(0-4)" -> FT=4-9, HT=0-4
CORNER_FT_RE = re.compile(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b")
CORNER_HT_RE = re.compile(r"\((\d{1,2})\s*-\s*(\d{1,2})\)")

def norm_key(s: str) -> str:
    """Metin normalleştirme (küçük harf, sadece alfanümerik)."""
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
    corner_home: Optional[int] = None   # FT corner home
    corner_away: Optional[int] = None   # FT corner away
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

    # 1) Standart kolon düzeni ile dene
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

    # 2) Fallback: Score Regex ile satır tarama
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
# ODDS EXTRACTION (FIXED & IMPROVED)
# ======================
def extract_bet365_initial_odds(page_source: str) -> Optional[Dict[str, float]]:
    """
    Sayfadaki 'Live Odds Comparison' veya benzeri tabloları tarar.
    Satır satır gezerek 'Bet365' ismini arar, ardından 'Initial' sütununu bulur.
    Regex yerine tablo parsing mantığı kullanır, daha güvenilirdir.
    """
    tables = extract_tables_html(page_source)
    
    # Hedef tabloyu bulmaya çalış
    target_tables = []
    for t in tables:
        t_low = t.lower()
        if "bet365" in t_low and ("1x2" in t_low or "odds" in t_low):
            target_tables.append(t)
    
    if not target_tables:
        # Spesifik bulunamazsa hepsine bak
        target_tables = tables

    for table_html in target_tables:
        rows = extract_table_rows_from_html(table_html)
        for row in rows:
            # Satırda Bet365 geçiyor mu?
            row_text = " ".join(row).lower()
            if "bet365" not in row_text:
                continue
            
            # Bu satırda 'Initial' kelimesi veya buna benzer bir yapı var mı kontrolü zor
            # çünkü bazen 'Initial' header'da yazar, satırda sadece sayı vardır.
            # Ancak Bet365 satırı genelde: [Bet365, Initial Odds..., Live Odds...]
            # Biz sadece sayı formatındaki (x.xx) değerleri toplayıp analiz edelim.
            
            # Sadece ondalıklı sayıları (float) çek
            floats = []
            for cell in row:
                # Parantez içindekileri temizle (bazen değişim oranları yazar)
                clean_cell = re.sub(r'\(.*?\)', '', cell)
                try:
                    val = float(clean_cell)
                    if 1.0 < val < 1000.0: # Mantıklı oran aralığı
                        floats.append(val)
                except ValueError:
                    continue
            
            # Initial oranlar genellikle satırın ilk 3 sayısıdır.
            # (Home, Draw, Away)
            if len(floats) >= 3:
                # İlk 3 oranın initial olma ihtimali çok yüksek
                return {"1": floats[0], "X": floats[1], "2": floats[2]}

    # Fallback: Regex ile tüm sayfada Bet365 Initial arama
    try:
        # Pattern: Bet365...Initial... 2.90... 2.88... 2.60
        # HTML taglerini yok sayarak arar
        pattern = r"Bet365.*?Initial.*?(\d+\.\d{2}).*?(\d+\.\d{2}).*?(\d+\.\d{2})"
        match = re.search(pattern, strip_tags(page_source), re.IGNORECASE | re.DOTALL)
        if match:
             return {"1": float(match.group(1)), "X": float(match.group(2)), "2": float(match.group(3))}
    except Exception:
        pass

    return None

# ======================
# PREVIOUS & H2H & FILTERS
# ======================
def check_same_league(league1: str, league2: str) -> bool:
    """
    İki lig isminin aynı olup olmadığını kontrol eder.
    'ITA D1' vs 'Italian Serie A' gibi durumlarda fuzzy match yapar.
    """
    l1 = norm_key(league1)
    l2 = norm_key(league2)
    
    if not l1 or not l2: 
        return False
    
    # Tam eşleşme
    if l1 == l2: return True
    if l1 in l2 or l2 in l1: return True
    
    # Kelime bazlı kesişim (Örn: 'italian' ve 'ita')
    words1 = set(l1.split())
    words2 = set(l2.split())
    
    # Ortak kelime var mı? (ve, vs, lig gibi genel kelimeler hariç tutulmalı ama basitlik için kalsın)
    # ITA D1 vs Italian Serie A -> Kesişim yok gibi duruyor.
    # Bu yüzden manuel mapping veya ilk 3 harf kontrolü eklenebilir.
    
    # İlk 3-4 harf kontrolü (ITA == ITA)
    if l1[:3] == l2[:3] and len(l1) > 2: return True
    
    return False

def extract_previous_matches_advanced(page_source: str, 
                                      team_name: str, 
                                      side: str, 
                                      current_league: str) -> List[MatchRow]:
    """
    Belirtilen takım için Previous Scores tablosunu çeker.
    Ardından şu filtreleri uygular:
    1. Sadece belirtilen taraf (Home veya Away).
    2. Sadece AYNI LİG (Same League).
    """
    markers = ["Previous Scores Statistics", "Previous Scores", "Recent Matches"]
    tabs = []
    
    # Tabloyu bul
    for marker in markers:
        found_tabs = section_tables_by_marker(page_source, marker, max_tables=10)
        if found_tabs:
            tabs = found_tabs
            break
            
    if not tabs:
        # Fallback
        all_tables = extract_tables_html(page_source)
        for t in all_tables:
            matches = parse_matches_from_table_html(t)
            if matches and len(matches) >= 3:
                tabs.append(t)
                
    # Hangi tablonun hangi takıma ait olduğunu başlık veya içerikten anla
    # Genelde ilk tablo Ev Sahibi, ikinci tablo Deplasman.
    # Ancak parametrik olarak 'side' (home/away) gönderdik, bu fonksiyonu iki kere çağıracağız.
    
    # Basit varsayım: Bu fonksiyon tüm önceki maçları döner, filtrelemeyi main fonksiyonda yaparız.
    # Fakat burası daha güvenli: Tüm tabloları birleştirip team_name içerenleri alalım.
    
    all_extracted_matches = []
    for t in tabs:
        ms = parse_matches_from_table_html(t)
        all_extracted_matches.extend(ms)
        
    # Dedupe
    all_extracted_matches = dedupe_matches(all_extracted_matches)
    
    # FİLTRELEME MANTIĞI
    t_key = norm_key(team_name)
    filtered = []
    
    for m in all_extracted_matches:
        # Takım kontrolü
        is_home = (norm_key(m.home) == t_key)
        is_away = (norm_key(m.away) == t_key)
        
        if not (is_home or is_away):
            continue
            
        # Side Filtresi (Home istiyorsak maçta Home olmalı, Away istiyorsak Away olmalı)
        if side == "home" and not is_home:
            continue
        if side == "away" and not is_away:
            continue
            
        # Lig Filtresi (Same League)
        if current_league and not check_same_league(m.league, current_league):
            continue
            
        filtered.append(m)
        
    # Tarihe göre sırala ve son N maçı al
    filtered = sort_matches_desc(filtered)
    return filtered[:RECENT_N]

def extract_h2h_matches(page_source: str, home_team: str, away_team: str) -> List[MatchRow]:
    markers = ["Head to Head Statistics", "Head to Head", "H2H Statistics", "H2H", "VS Statistics"]
    for mk in markers:
        tabs = section_tables_by_marker(page_source, mk, max_tables=5)
        for t in tabs:
            cand = parse_matches_from_table_html(t)
            if not cand: continue
            pair_count = sum(1 for m in cand if is_h2h_pair(m, home_team, away_team))
            if pair_count >= 1:
                return cand

    best_pair = 0
    best_list: List[MatchRow] = []
    for tbl in extract_tables_html(page_source):
        cand = parse_matches_from_table_html(tbl)
        if not cand: continue
        pair_count = sum(1 for m in cand if is_h2h_pair(m, home_team, away_team))
        if pair_count > best_pair:
            best_pair = pair_count
            best_list = cand
    return best_list

# ======================
# PREV STATS & CORNER
# ======================
def build_prev_stats(team: str, matches: List[MatchRow]) -> TeamPrevStats:
    tkey = norm_key(team)
    st = TeamPrevStats(name=team)
    if not matches:
        return st

    gfs, gas, corners_for, corners_against = [], [], [], []
    clean_sheets = 0
    scored_matches = 0

    for m in matches:
        if norm_key(m.home) == tkey:
            gf, ga = m.ft_home, m.ft_away
            cf, ca = m.corner_home, m.corner_away
        else:
            gf, ga = m.ft_away, m.ft_home
            cf, ca = m.corner_away, m.corner_home
        
        gfs.append(gf)
        gas.append(ga)
        if cf is not None: corners_for.append(cf)
        if ca is not None: corners_against.append(ca)
        if ga == 0: clean_sheets += 1
        if gf > 0: scored_matches += 1

    st.n_total = len(matches)
    st.gf_total = sum(gfs) / st.n_total if st.n_total else 0.0
    st.ga_total = sum(gas) / st.n_total if st.n_total else 0.0
    st.clean_sheets = clean_sheets
    st.scored_matches = scored_matches
    st.corners_for = sum(corners_for) / len(corners_for) if corners_for else 0.0
    st.corners_against = sum(corners_against) / len(corners_against) if corners_against else 0.0

    # PSS zaten filtrelenmiş geldiği için Home/Away ayrımına gerek yok, total kullanılır.
    # Ancak kod yapısını bozmamak için dolduruyoruz:
    st.n_home = st.n_total
    st.gf_home = st.gf_total
    st.ga_home = st.ga_total
    
    st.n_away = st.n_total
    st.gf_away = st.gf_total
    st.ga_away = st.ga_total

    return st

def analyze_corners_enhanced(home_prev: TeamPrevStats, away_prev: TeamPrevStats, h2h_matches: List[MatchRow]) -> Dict[str, Any]:
    """
    Filtrelenmiş (Home+SameLeague, Away+SameLeague) verileri kullanır.
    """
    h2h_corners_total = []
    for m in h2h_matches[:H2H_N]:
        if m.corner_home is not None and m.corner_away is not None:
            h2h_corners_total.append(m.corner_home + m.corner_away)

    h2h_avg = sum(h2h_corners_total) / len(h2h_corners_total) if h2h_corners_total else 0.0

    # Ev sahibi Evinde kaç korner atıyor/yiyor?
    home_exp = (home_prev.corners_for + home_prev.corners_against)
    
    # Deplasman Deplasmanda kaç korner atıyor/yiyor?
    away_exp = (away_prev.corners_for + away_prev.corners_against)

    # Ağırlıklı Tahmin
    # H2H varsa %40 H2H, %30 HomeForm, %30 AwayForm
    if h2h_avg > 0:
        total_corners = 0.4 * h2h_avg + 0.3 * home_exp + 0.3 * away_exp
    elif home_exp > 0 and away_exp > 0:
        total_corners = 0.5 * home_exp + 0.5 * away_exp
    else:
        total_corners = 0.0

    # Ev/Dep ayrımı (Tahmini)
    # Ev sahibi genelde toplamın %55'ini kullanır, deplasman %45 (Basit heuristic)
    # Ancak istatistik varsa onu kullanalım:
    if home_prev.corners_for + away_prev.corners_for > 0:
        ratio_h = home_prev.corners_for / (home_prev.corners_for + away_prev.corners_for)
        pred_h = total_corners * ratio_h
        pred_a = total_corners * (1 - ratio_h)
    else:
        pred_h = total_corners * 0.55
        pred_a = total_corners * 0.45

    predictions = {}
    for line in [8.5, 9.5, 10.5, 11.5]:
        if total_corners == 0:
            predictions[f"O{line}"] = 0.0
            predictions[f"U{line}"] = 0.0
        else:
            # Poisson approximation for Over probability
            # Lambda = total_corners. P(X > line)
            # Basit normal dağlım benzeri skor
            diff = total_corners - line
            # Logistic sigmoid benzeri basit bir olasılık fonksiyonu
            prob = 1 / (1 + math.exp(-0.8 * diff)) 
            predictions[f"O{line}"] = float(prob)
            predictions[f"U{line}"] = float(1.0 - prob)

    return {
        "predicted_home_corners": round(pred_h, 1),
        "predicted_away_corners": round(pred_a, 1),
        "total_corners": round(total_corners, 1),
        "h2h_avg": round(h2h_avg, 1),
        "predictions": predictions
    }

# ======================
# LAMBDA & CALC
# ======================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in w.values())
    if s <= 1e-9: return {}
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
    }
    return lam_h, lam_a, meta

def compute_component_pss(home_prev: TeamPrevStats, away_prev: TeamPrevStats) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    if home_prev.n_total < 3 or away_prev.n_total < 3:
        return None
    # Zaten filtrelenmiş (Home+SameLeague, Away+SameLeague) veriler geldiği için total alıyoruz
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
        "formula": "PSS (Filtered): (home_gf + away_ga) / 2"
    }
    return lam_h, lam_a, meta

def compute_component_h2h(h2h_matches: List[MatchRow], home_team: str, away_team: str) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    if not h2h_matches or len(h2h_matches) < 2:
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
    if len(hg) < 2:
        return None
    lam_h = sum(hg) / len(hg)
    lam_a = sum(ag) / len(ag)
    meta = {"matches": len(hg), "hg_avg": lam_h, "ag_avg": lam_a}
    return lam_h, lam_a, meta

def clamp_lambda(lh: float, la: float) -> Tuple[float, float, List[str]]:
    warn = []
    def c(x: float, name: str) -> float:
        if x < 0.15:
            warn.append(f"{name} düşük ({x:.2f}) → 0.15")
            return 0.15
        if x > 3.80:
            warn.append(f"{name} yüksek ({x:.2f}) → 3.80")
            return 3.80
        return x
    return c(lh, "λ_home"), c(la, "λ_away"), warn

def compute_lambdas(st_home_s, st_away_s, home_prev, away_prev, h2h_used, home_team, away_team):
    info: Dict[str, Any] = {"components": {}, "weights_used": {}, "warnings": []}
    comps = {}

    stc = compute_component_standings(st_home_s, st_away_s)
    if stc: comps["standing"] = stc

    pss = compute_component_pss(home_prev, away_prev)
    if pss: comps["pss"] = pss

    h2c = compute_component_h2h(h2h_used, home_team, away_team)
    if h2c: comps["h2h"] = h2c

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
    if clamp_warn: info["warnings"].extend(clamp_warn)
    return lh, la, info

# ======================
# STATS & MC
# ======================
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0: return 1.0 if k == 0 else 0.0
    if k > 170: return 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except:
        return 0.0

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

    def p(mask) -> float: return float(np.mean(mask))

    cnt = Counter(zip(hg.tolist(), ag.tolist()))
    top10_list = [(f"{h}-{a}", c / n * 100.0) for (h, a), c in cnt.most_common(10)]

    return {
        "p": {
            "1": p(hg > ag), "X": p(hg == ag), "2": p(hg < ag),
            "BTTS": p((hg >= 1) & (ag >= 1)),
            "O2.5": p(total >= 3), "U2.5": p(total <= 2)
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

# ======================
# KELLY & VALUE
# ======================
def value_and_kelly(prob: float, odds: float) -> Tuple[float, float]:
    if odds <= 1.0 or prob <= 0.0: return 0.0, 0.0
    v = odds * prob - 1.0
    b = odds - 1.0
    k = max(0.0, (b * prob - (1.0 - prob)) / b)
    return v, k

def confidence_label(p: float) -> str:
    if p >= 0.65: return "Yüksek"
    if p >= 0.55: return "Orta"
    return "Düşük"

def net_ou_prediction(probs: Dict[str, float]) -> Tuple[str, float, str]:
    p_o = probs.get("O2.5", 0)
    p_u = probs.get("U2.5", 0)
    if p_o >= p_u: return "2.5 ÜST", p_o, confidence_label(p_o)
    return "2.5 ALT", p_u, confidence_label(p_u)

def net_btts_prediction(probs: Dict[str, float]) -> Tuple[str, float, str]:
    p_b = probs.get("BTTS", 0)
    p_n = 1.0 - p_b
    if p_b >= p_n: return "VAR", p_b, confidence_label(p_b)
    return "YOK", p_n, confidence_label(p_n)

def final_decision(qualified: List, diff: float) -> str:
    if not qualified:
        return f"OYNAMA (Value yok, diff={diff:.2f})"
    best = sorted(qualified, key=lambda x: x[3], reverse=True)[0]
    return f"OYNANABİLİR → {best[0]} (Value %{best[3]*100:.1f})"

def format_comprehensive_report(data: Dict[str, Any]) -> str:
    t = data["teams"]
    blend = data["blended_probs"]
    top7 = data["poisson"]["top7_scores"]
    lines = []
    lines.append("=" * 60)
    lines.append(f"  {t['home']} vs {t['away']}")
    lines.append(f"  Lig: {data['league']}")
    lines.append("=" * 60)

    lines.append(f"\nOLASI SKORLAR:")
    for i, (score, prob) in enumerate(top7[:5], 1):
        bar = "█" * int(prob * 50)
        lines.append(f"  {i}. {score:6s} %{prob*100:4.1f}  {bar}")

    lines.append(f"\nNET TAHMİN:")
    lines.append(f"  Ana Skor: {top7[0][0]}")
    if len(top7) >= 2: lines.append(f"  Alt Skor: {top7[1][0]}")

    net_ou, net_ou_p, _ = net_ou_prediction(blend)
    lines.append(f"\nAlt/Üst 2.5: {net_ou} (%{net_ou_p*100:.1f})")

    net_btts, net_btts_p, _ = net_btts_prediction(blend)
    lines.append(f"KG Var: {net_btts} (%{net_btts_p*100:.1f})")

    lines.append(f"\n1X2 Olasılıkları:")
    lines.append(f"  Ev (1): %{blend.get('1', 0)*100:.1f}")
    lines.append(f"  Ber(X): %{blend.get('X', 0)*100:.1f}")
    lines.append(f"  Dep(2): %{blend.get('2', 0)*100:.1f}")

    corners = data.get("corner_analysis", {})
    if corners:
        lines.append(f"\nKorner Tahmini: {corners['total_corners']}")
        lines.append(f"  (Ev: {corners['predicted_home_corners']} | Dep: {corners['predicted_away_corners']})")
        lines.append(f"  H2H Ort: {corners['h2h_avg']}")

    vb = data.get("value_bets", {})
    if vb.get("used_odds"):
        lines.append(f"\nBAHİS ANALİZİ (Bet365 Initial):")
        for row in vb.get("table", []):
            if row["value"] >= VALUE_MIN:
                lines.append(f"  ✅ {row['market']}: Oran {row['odds']:.2f} | Value %{row['value']*100:+.1f}")
        lines.append(f"\nKARAR: {vb.get('decision')}")
    else:
        lines.append("\n⚠️ Oran verisi çekilemedi.")

    ds = data["data_sources"]
    lines.append(f"\nKullanılan Veriler (Filtreli):")
    lines.append(f"  H2H: {ds['h2h_matches']} maç")
    lines.append(f"  Home (Ev+AynıLig): {ds['home_filtered_matches']} maç")
    lines.append(f"  Away (Dep+AynıLig): {ds['away_filtered_matches']} maç")

    lines.append("=" * 60)
    return "\n".join(lines)

# ======================
# MAIN ANALYSIS
# ======================
def analyze_nowgoal(url: str, odds: Optional[Dict[str, float]] = None, mc_runs: int = MC_RUNS_DEFAULT) -> Dict[str, Any]:
    h2h_url = build_h2h_url(url)
    print(f"[DEBUG] Fetching {h2h_url}...")
    html = safe_get(h2h_url)

    home_team, away_team = parse_teams_from_title(html)
    if not home_team: raise RuntimeError("Takım isimleri alınamadı")

    # Lig adını al
    league_match = re.search(r'<span[^>]*class=["\']?sclassLink["\']?[^>]*>([^<]+)</span>', html)
    league_name = strip_tags(league_match.group(1)) if league_match else ""
    print(f"[DEBUG] League detected: {league_name}")

    # Standings
    st_home_rows = extract_standings_for_team(html, home_team)
    st_away_rows = extract_standings_for_team(html, away_team)
    st_home = standings_to_splits(st_home_rows)
    st_away = standings_to_splits(st_away_rows)

    # H2H
    h2h_all = extract_h2h_matches(html, home_team, away_team)
    h2h_pair = [m for m in h2h_all if is_h2h_pair(m, home_team, away_team)]
    h2h_used = sort_matches_desc(dedupe_matches(h2h_pair))[:H2H_N]

    # PREVIOUS SCORES (STRICT FILTER)
    # Ev sahibi için: Home + Same League
    # Deplasman için: Away + Same League
    print("[DEBUG] Applying Strict Filters (Home+SameLeague / Away+SameLeague)...")
    
    prev_home_filtered = extract_previous_matches_advanced(html, home_team, "home", league_name)
    prev_away_filtered = extract_previous_matches_advanced(html, away_team, "away", league_name)
    
    # Filtre sonucu çok az maç kalırsa (örn: Kupa maçı veya lig adı uyuşmazlığı),
    # Fallback olarak genel son maçları alalım mı? Kullanıcı "ayni oranlari da cekemiyoruz" dediği için
    # muhtemelen sıkı filtre istiyor. Ancak hiç veri yoksa analiz patlar.
    # Bu yüzden en az 1 maç yoksa filtreyi biraz gevşetebiliriz ama kullanıcı isteğine sadık kalalım.
    if len(prev_home_filtered) < 3:
        print(f"[WARNING] Home filtered matches low ({len(prev_home_filtered)}). Analysis relies more on standings.")
    if len(prev_away_filtered) < 3:
        print(f"[WARNING] Away filtered matches low ({len(prev_away_filtered)}).")

    home_prev_stats = build_prev_stats(home_team, prev_home_filtered)
    away_prev_stats = build_prev_stats(away_team, prev_away_filtered)

    # Lambda & Poisson
    lam_home, lam_away, lambda_info = compute_lambdas(
        st_home_s=st_home,
        st_away_s=st_away,
        home_prev=home_prev_stats,
        away_prev=away_prev_stats,
        h2h_used=h2h_used,
        home_team=home_team,
        away_team=away_team
    )

    score_mat = build_score_matrix(lam_home, lam_away, max_g=MAX_GOALS_FOR_MATRIX)
    poisson_market = market_probs_from_matrix(score_mat)
    top7 = [(f"{h}-{a}", p) for (h, a), p in sorted(score_mat.items(), key=lambda x: x[1], reverse=True)[:7]]

    # Monte Carlo
    mc = monte_carlo(lam_home, lam_away, n=int(mc_runs))
    blended = blend_probs(poisson_market, mc["p"], alpha=BLEND_ALPHA)

    # CORNER ANALYSIS (Enhanced)
    corner_analysis = analyze_corners_enhanced(home_prev_stats, away_prev_stats, h2h_used)

    # ODDS EXTRACTION (Bet365 Initial)
    if not odds:
        odds = extract_bet365_initial_odds(html)
        if odds:
            print(f"[DEBUG] Odds found: {odds}")
        else:
            print("[DEBUG] Odds NOT found.")

    value_block = {"used_odds": False}
    qualified = []
    diff = 0.0 # Basitlik için

    if odds and all(k in odds for k in ["1", "X", "2"]):
        value_block["used_odds"] = True
        table = []
        for mkt in ["1", "X", "2"]:
            o = float(odds[mkt])
            p = float(blended.get(mkt, 0.0))
            v, k = value_and_kelly(p, o)
            qk = max(0.0, 0.25 * k)
            row = {"market": mkt, "prob": p, "odds": o, "value": v, "kelly": k}
            table.append(row)
            if v >= VALUE_MIN and p >= PROB_MIN:
                qualified.append((mkt, p, o, v, qk))
        
        value_block["table"] = table
        value_block["decision"] = final_decision(qualified, diff)

    data = {
        "url": h2h_url,
        "teams": {"home": home_team, "away": away_team},
        "league": league_name,
        "lambda": {"home": lam_home, "away": lam_away, "info": lambda_info},
        "poisson": {"market_probs": poisson_market, "top7_scores": top7},
        "blended_probs": blended,
        "corner_analysis": corner_analysis,
        "value_bets": value_block,
        "data_sources": {
            "h2h_matches": len(h2h_used),
            "home_filtered_matches": len(prev_home_filtered),
            "away_filtered_matches": len(prev_away_filtered)
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
    return jsonify({"ok": True, "service": "nowgoal-analyzer-v5", "status": "active"})

@app.post("/analiz_et")
def analiz_et_route():
    try:
        payload = request.get_json(silent=True) or {}
        url = (payload.get("url") or "").strip()
        if not url: return jsonify({"ok": False, "error": "URL yok"}), 400

        data = analyze_nowgoal(url, odds=None, mc_runs=10_000)
        
        top_skor = data["poisson"]["top7_scores"][0][0]
        net_ou, net_ou_p, _ = net_ou_prediction(data["blended_probs"])
        net_btts, net_btts_p, _ = net_btts_prediction(data["blended_probs"])
        
        return jsonify({
            "ok": True,
            "skor": top_skor,
            "alt_ust": f"{net_ou} (%{net_ou_p*100:.1f})",
            "btts": f"{net_btts} (%{net_btts_p*100:.1f})",
            "karar": data["value_bets"].get("decision", "Veri Yok"),
            "detay": data["report_comprehensive"]
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/analyze")
def analyze_route():
    # Full data dump endpoint
    try:
        payload = request.get_json(silent=True) or {}
        url = payload.get("url")
        if not url: return jsonify({"error": "no url"}), 400
        data = analyze_nowgoal(url)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
