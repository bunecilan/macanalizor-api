# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - Enhanced Version 4.1 (ODDS+PSS FIX)
Flask API with Corner Analysis & Enhanced Value Betting

FIX PACK (v4.1):
1) Bet365 Initial 1X2 odds artık H2H sayfasından değil /oddscomp/{matchid} üzerinden çekilir.
   - Hücre text'i boşsa title/data-* gibi attribute içinden de sayı yakalanır.
2) PSS "tıklama" mantığı taklit edildi:
   - Home team: Same League + HOME-only
   - Away team: Same League + AWAY-only
   Böylece bazı maçlarda 10 yerine 7 maç doğal olarak kalır.
3) H2H için Same League filtre (en az 3 maç varsa) uygulanır; yoksa all H2H fallback.
4) Corner O/U olasılıkları lineer yaklaşık yerine Poisson ile hesaplanır.
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

W_ST_BASE = 0.45
W_PSS_BASE = 0.30
W_H2H_BASE = 0.25

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

# --- Yeni yardımcı: hem dict hem obje için güvenli alan erişimi ---
def get_field_value(row: Any, field_name: str) -> Optional[str]:
    """
    row: MatchRow objesi veya dict
    field_name: orijinal başlık, örn. "Same League"
    Denenen anahtarlar (sırasıyla):
      - dict['Same League']
      - obj.same_league (alt çizgi versiyonu)
      - obj.Same League (nadiren mümkün)
      - obj.league (fallback)
    Döndürülen değer string veya None.
    """
    if row is None:
        return None

    # 1) dict olarak erişim (CSV/DictReader durumları)
    try:
        if isinstance(row, dict):
            if field_name in row:
                return row.get(field_name)
            # küçük/normalize edilmiş anahtarlar
            alt = field_name.replace(" ", "_")
            if alt in row:
                return row.get(alt)
            alt2 = alt.lower()
            if alt2 in row:
                return row.get(alt2)
    except Exception:
        pass

    # 2) obje (dataclass/obj) için alt çizgi versiyonu
    try:
        attr = field_name.replace(" ", "_")
        if hasattr(row, attr):
            return getattr(row, attr)
        # küçük harfli alt çizgi
        attr2 = attr.lower()
        if hasattr(row, attr2):
            return getattr(row, attr2)
        # doğrudan orijinal isim (çoğu zaman geçersiz ama deneyelim)
        if hasattr(row, field_name):
            return getattr(row, field_name)
    except Exception:
        pass

    # 3) __dict__ içinde orijinal anahtar
    try:
        d = getattr(row, "__dict__", None)
        if isinstance(d, dict):
            if field_name in d:
                return d[field_name]
            alt = field_name.replace(" ", "_")
            if alt in d:
                return d[alt]
            alt2 = alt.lower()
            if alt2 in d:
                return d[alt2]
    except Exception:
        pass

    # 4) fallback: common 'league' field
    try:
        if isinstance(row, dict) and "league" in row:
            return row.get("league")
        if hasattr(row, "league"):
            return getattr(row, "league")
    except Exception:
        pass

    return None

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
    Hücreleri SİLMEYİN. Boş hücreler kolon hizası için gerekli.
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
    """
    '4-9(0-4)' -> FT=(4,9), HT=(0,4)
    """
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

    # fallback: skor hücresini tara
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
# ODDS (Bet365 Initial 1X2) - NEW
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
    """
    Hem float hem int alır (OU line gibi 2, 2.5 vs).
    """
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
    """
    NowGoal bazen hücre text'ini boş bırakıp odds'u attribute'ta saklar:
    title="2.90" data-odd="2.90" vb.
    """
    if not inner_html:
        return None
    # 1) text
    txt = strip_tags_keep_text(inner_html)
    v = _extract_first_float(txt)
    if v is not None:
        return v

    # 2) attribute taraması
    # title="2.90" veya data-xxx="2.90"
    m = re.search(r'(?:title|data-[a-z0-9_-]+)\s*=\s*"(\d+\.\d+)"', inner_html, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None

    # 3) fallback: inner_html içinde geçen ilk float
    v2 = _extract_first_float(inner_html)
    return v2

def extract_bet365_initial_1x2_from_oddscomp_html(odds_html: str) -> Optional[Dict[str, float]]:
    """
    oddscomp tablosunda Bet365 satırının "Initial" kısmından 1X2 (Home/Draw/Away) çeker.
    Beklenen kolon düzeni (genel):
      Company | Initial/Live/In-Play | AH(3) | 1X2(3) | OU(3) | Trends
    """
    if not odds_html:
        return None

    # Bet365 satırını yakala
    tr_m = re.search(r"(<tr\b[^>]*>.*?Bet365.*?</tr>)", odds_html, flags=re.I | re.S)
    if not tr_m:
        return None

    tr_html = tr_m.group(1)
    tds = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr_html, flags=re.I | re.S)
    if not tds or len(tds) < 8:
        # bazı sayfalarda satır <td> yerine farklı; fallback numeric scan
        nums = _extract_all_numbers_loose(strip_tags_keep_text(tr_html))
        # tipik: [AHh, AHH, AHa, 1, X, 2, OUo, line, OUu]
        if len(nums) >= 6:
            # 1X2 çoğu zaman AH'den sonra gelen üçlü
            cand = nums[3:6]
            if all(1.01 <= x <= 50 for x in cand):
                return {"1": float(cand[0]), "X": float(cand[1]), "2": float(cand[2])}
        return None

    # Hücrelerden numeric çıkar
    cell_vals: List[Optional[float]] = [_extract_cell_numeric_from_inner_html(td) for td in tds]

    # Kolon düzeni varsayımı:
    # idx0: company
    # idx1: Initial/Live/In-Play label
    # idx2-4: AH
    # idx5-7: 1X2
    if len(cell_vals) >= 8:
        o1, ox, o2 = cell_vals[5], cell_vals[6], cell_vals[7]
        if all(v is not None for v in [o1, ox, o2]):
            if all(1.01 <= float(v) <= 200 for v in [o1, ox, o2]):
                return {"1": float(o1), "X": float(ox), "2": float(o2)}

    # Fallback: satır içi numeric dizisinden 1X2 seç
    nums = _extract_all_numbers_loose(strip_tags_keep_text(tr_html))
    if len(nums) >= 6:
        cand = nums[3:6]
        if all(1.01 <= x <= 50 for x in cand):
            return {"1": float(cand[0]), "X": float(cand[1]), "2": float(cand[2])}

    return None

def extract_bet365_initial_odds(url: str) -> Optional[Dict[str, float]]:
    """
    PRIMARY: /oddscomp/{matchid}
    """
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
    """
    Previous Scores Statistics tablosunu bul.
    Genelde 2 tablo: Ev sahibi blok + Deplasman blok
    """
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

# --- Düzeltilmiş: Same League filtresi (team home gibi tek satırlık, "Same League" başlığını destekler) ---
def filter_same_league_matches(matches: List[MatchRow], league_name: str) -> List[MatchRow]:
    """
    league_name boşsa matches döner.
    'Same League' başlığı olan kaynaklar için güvenli erişim yapar.
    Eğer filtre sonucu boşsa orijinal matches döner (fallback).
    """
    if not league_name:
        return matches
    lk = norm_key(league_name)
    # tek satırlık comprehension, get_field_value ile 'Same League' veya 'league' alanını dener
    out = [
        m for m in matches
        if lk and (
            lk in norm_key(get_field_value(m, "Same League") or "") or
            norm_key(get_field_value(m, "Same League") or "") in lk or
            lk in norm_key(get_field_value(m, "league") or "") or
            norm_key(get_field_value(m, "league") or "") in lk
        )
    ]
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
    # P(X <= k)
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))

# ======================
# CORNER ANALYSIS (Poisson O/U)
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
        # Over 9.5 => X >= 10
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
    if home_prev.n_total < 3 or away_prev.n_total < 3:
        return None

    # PSS zaten Home-only ve Away-only’e çekildiği için doğrudan total kullanıyoruz
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
        return x
    lh2 = c(lh, "home_lambda")
    la2 = c(la, "away_lambda")
    return lh2, la2, warn

# ======================
# BLEND & MC
# ======================
def blend_lambdas(components: List[Tuple[float, float, Dict[str, Any]]], weights: Dict[str, float]) -> Tuple[float, float, Dict[str, Any]]:
    """
    components: list of (lam_h, lam_a, meta) tuples
    weights: normalized weights for keys 'stand', 'pss', 'h2h'
    """
    # collect available components
    lam_h_vals = []
    lam_a_vals = []
    metas = {}
    for name, w in weights.items():
        # find component by name in meta
        for comp in components:
            meta = comp[2]
            if meta and meta.get("formula", "").lower().find(name) != -1 or meta.get("formula", "").lower().find(name.replace("pss", "pss")) != -1:
                lam_h_vals.append((comp[0], w))
                lam_a_vals.append((comp[1], w))
                metas[name] = meta
                break
    # fallback: if weights empty, average all components equally
    if not lam_h_vals:
        for comp in components:
            lam_h_vals.append((comp[0], 1.0))
            lam_a_vals.append((comp[1], 1.0))
    # weighted average
    def wavg(pairs):
        s = sum(w for _, w in pairs)
        if s <= 1e-9:
            return sum(v for v, _ in pairs) / max(1, len(pairs))
        return sum(v * w for v, w in pairs) / s
    lam_h = wavg(lam_h_vals)
    lam_a = wavg(lam_a_vals)
    lam_h, lam_a, warns = clamp_lambda(lam_h, lam_a)
    return lam_h, lam_a, {"meta": metas, "warnings": warns}

def simulate_match_probs(lh: float, la: float, max_goals: int = MAX_GOALS_FOR_MATRIX) -> Dict[str, Any]:
    """
    Monte Carlo veya exact Poisson convolution to compute probabilities for 1X2 and totals.
    For speed and determinism we compute exact Poisson matrix up to max_goals.
    """
    probs = {}
    # compute matrix
    matrix = [[poisson_pmf(i, lh) * poisson_pmf(j, la) for j in range(0, max_goals + 1)] for i in range(0, max_goals + 1)]
    # tail mass for >max_goals
    tail_h = 1.0 - sum(sum(row) for row in matrix)
    # approximate tail by adding to last row/col diagonal
    matrix[-1][-1] += tail_h

    p_home = sum(matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix)) if i > j)
    p_draw = sum(matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix)) if i == j)
    p_away = sum(matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix)) if i < j)

    # totals probabilities (over/under lines)
    total_probs = {}
    # compute distribution of total goals
    total_dist = {}
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            s = i + j
            total_dist[s] = total_dist.get(s, 0.0) + matrix[i][j]
    # collapse >max_goals into last bucket
    total_cum = 0.0
    for k, v in total_dist.items():
        total_cum += v
    # compute over/under for common lines
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        # under = sum total_dist[s] for s <= floor(line)
        k = int(math.floor(line))
        p_under = sum(v for s, v in total_dist.items() if s <= k)
        total_probs[f"U{line}"] = p_under
        total_probs[f"O{line}"] = 1.0 - p_under

    probs["1"] = p_home
    probs["X"] = p_draw
    probs["2"] = p_away
    probs["totals"] = total_probs
    probs["matrix_total_mass"] = sum(total_dist.values())
    return probs

def implied_prob_from_odds(o: float) -> float:
    if not o or o <= 0:
        return 0.0
    return 1.0 / o

def kelly_fraction(p: float, o: float) -> float:
    """
    p: estimated probability
    o: decimal odds
    returns fraction of bankroll (Kelly criterion) with simple formula
    """
    b = o - 1.0
    if b <= 0:
        return 0.0
    num = p * (b + 1) - 1
    den = b
    if den <= 0:
        return 0.0
    f = num / den
    return max(0.0, f)

def find_value_bets(probs: Dict[str, float], odds: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    probs: {'1':p1,'X':pX,'2':p2}
    odds: {'1':o1,'X':oX,'2':o2}
    """
    out = []
    for k in ["1", "X", "2"]:
        p = probs.get(k, 0.0)
        o = odds.get(k)
        if o is None:
            continue
        imp = implied_prob_from_odds(o)
        value = p - imp
        if value >= VALUE_MIN and p >= PROB_MIN:
            kelly = kelly_fraction(p, o)
            out.append({
                "market": k,
                "prob": round(p, 4),
                "odds": round(o, 3),
                "implied": round(imp, 4),
                "value": round(value, 4),
                "kelly": round(kelly, 4)
            })
    return sorted(out, key=lambda x: x["value"], reverse=True)

# ======================
# HIGH LEVEL ANALYSIS
# ======================
def analyze_match_from_urls(h2h_url: str, odds_url: str, home_team: str, away_team: str) -> Dict[str, Any]:
    """
    High level pipeline:
      - fetch pages
      - extract previous matches (PSS)
      - extract H2H
      - apply Same League filters and home/away filters for PSS
      - compute prev stats
      - compute components (standings, pss, h2h)
      - blend lambdas
      - simulate probabilities
      - extract odds and find value bets
      - corner analysis
    """
    result = {"errors": [], "warnings": []}
    try:
        # fetch pages
        h2h_html = safe_get(h2h_url)
        odds_html = safe_get(build_oddscomp_url(odds_url))
    except Exception as e:
        result["errors"].append(str(e))
        return result

    # parse teams if not provided
    try:
        t1, t2 = parse_teams_from_title(h2h_html)
        if not home_team:
            home_team = t1
        if not away_team:
            away_team = t2
    except Exception:
        pass

    # extract previous (PSS) tables
    prev_home_tables, prev_away_tables = extract_previous_from_page(h2h_html)
    prev_home_matches = []
    prev_away_matches = []
    if prev_home_tables:
        prev_home_matches = parse_matches_from_table_html(prev_home_tables[0])
    if prev_away_tables:
        prev_away_matches = parse_matches_from_table_html(prev_away_tables[0])

    # apply Same League + HOME-only for home PSS
    same_league_filtered_home = filter_same_league_matches(prev_home_matches, get_field_value(prev_home_matches[0], "Same League") or "")
    home_pss_home_only = filter_team_home_only(same_league_filtered_home, home_team)
    # if not enough, fallback to unfiltered recent
    if len(home_pss_home_only) < 3:
        home_pss_home_only = prev_home_matches[:RECENT_N]

    # apply Same League + AWAY-only for away PSS
    same_league_filtered_away = filter_same_league_matches(prev_away_matches, get_field_value(prev_away_matches[0], "Same League") or "")
    away_pss_away_only = filter_team_away_only(same_league_filtered_away, away_team)
    if len(away_pss_away_only) < 3:
        away_pss_away_only = prev_away_matches[:RECENT_N]

    # build prev stats
    home_prev = build_prev_stats(home_team, home_pss_home_only)
    away_prev = build_prev_stats(away_team, away_pss_away_only)

    # extract H2H
    h2h_matches = extract_h2h_matches(h2h_html, home_team, away_team)
    # try Same League filter on H2H if available
    h2h_filtered = filter_same_league_matches(h2h_matches, get_field_value(h2h_matches[0], "Same League") or "") if h2h_matches else []
    if len(h2h_filtered) >= 3:
        h2h_used = h2h_filtered
    else:
        h2h_used = h2h_matches

    # compute components
    components = []
    # standings component (try to extract standings from page)
    try:
        st_home_rows = extract_standings_for_team(h2h_html, home_team)
        st_away_rows = extract_standings_for_team(h2h_html, away_team)
        st_home = standings_to_splits(st_home_rows)
        st_away = standings_to_splits(st_away_rows)
        comp_st = compute_component_standings(st_home, st_away)
        if comp_st:
            components.append(comp_st)
    except Exception:
        pass

    # pss component
    comp_pss = compute_component_pss(home_prev, away_prev)
    if comp_pss:
        components.append(comp_pss)

    # h2h component
    comp_h2h = compute_component_h2h(h2h_used, home_team, away_team)
    if comp_h2h:
        components.append(comp_h2h)

    # prepare weights (simple heuristic: prefer components that exist)
    weights = {}
    if any(c for c in components if c[2].get("formula", "").lower().find("standing") != -1):
        weights["stand"] = W_ST_BASE
    if comp_pss:
        weights["pss"] = W_PSS_BASE
    if comp_h2h:
        weights["h2h"] = W_H2H_BASE
    # normalize
    weights = normalize_weights(weights)

    # blend lambdas
    lam_h, lam_a, blend_meta = blend_lambdas(components, weights)

    # simulate probabilities
    probs = simulate_match_probs(lam_h, lam_a)

    # extract odds
    odds = extract_bet365_initial_odds(odds_url) or {}

    # find value bets
    value_bets = find_value_bets({"1": probs["1"], "X": probs["X"], "2": probs["2"]}, odds)

    # corner analysis
    corner_info = analyze_corners(home_prev, away_prev, h2h_used)

    result.update({
        "home_team": home_team,
        "away_team": away_team,
        "lam_home": round(lam_h, 3),
        "lam_away": round(lam_a, 3),
        "probs": {"1": round(probs["1"], 4), "X": round(probs["X"], 4), "2": round(probs["2"], 4)},
        "odds": odds,
        "value_bets": value_bets,
        "corner": corner_info,
        "blend_meta": blend_meta
    })

    return result

# ======================
# FLASK APP
# ======================
app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def api_analyze():
    """
    POST JSON:
    {
      "h2h_url": "...",
      "odds_url": "...",   # can be same as h2h_url
      "home": "Team A",
      "away": "Team B"
    }
    """
    try:
        data = request.get_json(force=True)
        h2h_url = data.get("h2h_url") or data.get("url") or ""
        odds_url = data.get("odds_url") or h2h_url
        home = data.get("home", "") or ""
        away = data.get("away", "") or ""
        if not h2h_url:
            return jsonify({"error": "h2h_url veya url alanı gerekli"}), 400
        res = analyze_match_from_urls(h2h_url, odds_url, home, away)
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # development server
    app.run(host="0.0.0.0", port=5000, debug=False)
