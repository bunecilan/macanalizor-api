# app.py
# Flask API: NowGoal link -> JSON analiz (Standings + Previous + H2H + Poisson + MC + Value + Kelly)

import re
import math
import json
import html as html_lib
from dataclasses import dataclass, asdict, is_dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

import numpy as np
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# =========================
# AYARLAR
# =========================
MC_RUNS = 10_000
MC_MINI_SAMPLE = 100

RECENT_N = 10
H2H_N = 10

VALUE_MIN = 0.05
PROB_MIN = 0.55

W_ST_BASE = 0.55
W_PREV_BASE = 0.35
W_H2H_BASE = 0.10

OU_SANITY_MAX_DIFF = 0.60
OU_SANITY_BLEND = 0.40

BLEND_ALPHA = 0.50
MAX_GOALS_CAP = 12

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
}

# =========================
# REGEX
# =========================
DATE_ANY_RE = re.compile(r"\b(\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{2}-\d{2})\b")
SCORE_RE = re.compile(r"\b(\d{1,2})\s*-\s*(\d{1,2})(?!-\d{2,4})(?:\s*\((\d{1,2})\s*-\s*(\d{1,2})\))?\b")
LEAGUE_TOK_RE = re.compile(r"^[A-Z0-9]{2,6}$")
LEAGUE_HDR_RE = re.compile(r"\[([A-Z]{2,5}(?:\s+[A-Z0-9]{2,5})?)-\d+\]")

# =========================
# DATA CLASSES
# =========================
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
class StandingsFullRow:
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

# =========================
# JSON SAFE
# =========================
def json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]
    return str(obj)

# =========================
# HELPERS
# =========================
def clean_team_name(s: str) -> str:
    s = " ".join((s or "").split()).strip()
    s = re.sub(r"\s+Live Score.*$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+Football Analysis.*$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+Preview.*$", "", s, flags=re.IGNORECASE).strip()
    return s.strip(" ,;-")

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

def extract_match_id(url: str) -> str:
    m = re.search(r"(?:h2h-|/match/h2h-)(\d+)", url)
    if m:
        return m.group(1)
    nums = re.findall(r"\d{6,}", url)
    if not nums:
        raise ValueError("Match ID çıkaramadım")
    return nums[-1]

def extract_base_domain(url: str) -> str:
    m = re.match(r"^(https?://[^/]+)", (url or "").strip())
    return m.group(1) if m else "https://live3.nowgoal26.com"

def is_h2h_pair(m: MatchRow, home_team: str, away_team: str) -> bool:
    hk, ak = norm_key(home_team), norm_key(away_team)
    mh, ma = norm_key(m.home), norm_key(m.away)
    return (mh == hk and ma == ak) or (mh == ak and ma == hk)

def confidence_label(p: float) -> str:
    if p >= 0.65:
        return "Yüksek"
    if p >= 0.55:
        return "Orta"
    return "Düşük"

# =========================
# HTTP FETCH
# =========================
def fetch_html(urls: List[str], timeout: int = 25) -> Tuple[str, Optional[str]]:
    s = requests.Session()
    for u in urls:
        try:
            r = s.get(u, headers=HEADERS, timeout=timeout)
            if r.status_code != 200:
                continue
            html = r.text or ""
            if len(html) < 5_000:
                continue
            return html, u
        except Exception:
            continue
    return "", None

def build_h2h_urls(base: str, match_id: str) -> List[str]:
    # base + fallback live1..live6 + generic
    urls = [f"{base}/match/h2h-{match_id}"]

    m = re.match(r"^(https?://)([^/]+)", base)
    if m:
        scheme, host = m.group(1), m.group(2)
        m2 = re.match(r"^(live)(\d+)(\..+)$", host)
        if m2:
            rest = m2.group(3)
            for k in [1, 2, 3, 4, 5, 6]:
                urls.append(f"{scheme}live{k}{rest}/match/h2h-{match_id}")

    urls.extend([
        f"https://www.nowgoal26.com/match/h2h-{match_id}",
        f"https://live.nowgoal26.com/match/h2h-{match_id}",
    ])

    # uniq
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def build_odds_urls(base: str, match_id: str) -> List[str]:
    urls = [f"{base}/oddscomp/{match_id}"]

    m = re.match(r"^(https?://)([^/]+)", base)
    if m:
        scheme, host = m.group(1), m.group(2)
        m2 = re.match(r"^(live)(\d+)(\..+)$", host)
        if m2:
            rest = m2.group(3)
            for k in [1, 2, 3, 4, 5, 6]:
                urls.append(f"{scheme}live{k}{rest}/oddscomp/{match_id}")

    urls.extend([
        f"https://www.nowgoal26.com/oddscomp/{match_id}",
        f"https://live.nowgoal26.com/oddscomp/{match_id}",
    ])

    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# =========================
# HTML UTILS (seninkiyle aynı mantık)
# =========================
def strip_tags(s: str) -> str:
    s = re.sub(r"<script\b.*?</script>", " ", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"<style\b.*?</style>", " ", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html_lib.unescape(s)
    return " ".join(s.split()).strip()

def extract_tables_html(page_source: str) -> List[Tuple[int, str]]:
    out = []
    for m in re.finditer(r"<table\b[^>]*>.*?</table>", page_source or "", flags=re.IGNORECASE | re.DOTALL):
        out.append((m.start(), m.group(0)))
    return out

def extract_table_rows_from_html(table_html: str) -> List[List[str]]:
    rows = []
    for tr in re.findall(r"<tr\b[^>]*>.*?</tr>", table_html or "", flags=re.IGNORECASE | re.DOTALL):
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
    tables = [tbl for _, tbl in extract_tables_html(sub)]
    return tables[:max_tables]

# =========================
# MATCH PARSE
# =========================
def detect_league_from_cells(cells: List[str], date_idx: int) -> str:
    league = "—"
    if not cells:
        return league
    cand0 = (cells[0] or "").replace("_", " ").upper().strip()
    toks = cand0.split()
    if len(toks) >= 2 and all(LEAGUE_TOK_RE.match(x) for x in toks[:2]):
        return " ".join(toks[:2])
    if date_idx - 1 >= 0:
        cand2 = (cells[date_idx - 1] or "").replace("_", " ").upper().strip()
        toks2 = cand2.split()
        if len(toks2) >= 2 and all(LEAGUE_TOK_RE.match(x) for x in toks2[:2]):
            return " ".join(toks2[:2])
    return league

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
        after = c0[m.end():]
        if re.search(r"^\s*-\s*\d{2,4}\b", after):
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

    if ft_h > 9 or ft_a > 9:
        return None

    if score_idx - 1 < 0 or score_idx + 1 >= len(cells):
        return None

    home = clean_team_name(cells[score_idx - 1])
    away = clean_team_name(cells[score_idx + 1])
    if not home or not away:
        return None

    league = detect_league_from_cells(cells, date_idx)
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

# =========================
# STANDINGS PARSE
# =========================
def _to_int(x: str) -> Optional[int]:
    try:
        x = (x or "").strip()
        if x in {"", "-", "—"}:
            return None
        return int(x)
    except Exception:
        return None

def parse_standings_table_rows(rows: List[List[str]]) -> List[StandingsFullRow]:
    wanted = {"Total", "Home", "Away", "Last 6", "Last6"}
    out: List[StandingsFullRow] = []

    for cells in rows:
        if not cells:
            continue
        head = cells[0].strip()
        if head not in wanted:
            continue
        label = "Last 6" if head == "Last6" else head

        def g(i): return cells[i] if i < len(cells) else ""

        r = StandingsFullRow(
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
        if all(any(x.ft == z for x in out) for z in ["Total", "Home", "Away"]):
            break

    order = {"Total": 0, "Home": 1, "Away": 2, "Last 6": 3}
    out.sort(key=lambda x: order.get(x.ft, 99))
    return out

def extract_standings_from_html(page_source: str, home_team: str, away_team: str) -> Tuple[List[StandingsFullRow], List[StandingsFullRow]]:
    home_key = norm_key(home_team)
    away_key = norm_key(away_team)

    candidates = []
    for start, tbl in extract_tables_html(page_source):
        t = strip_tags(tbl).lower()
        if not all(k in t for k in ["matches", "win", "draw", "loss", "scored", "conceded"]):
            continue
        rows = extract_table_rows_from_html(tbl)
        parsed = parse_standings_table_rows(rows)
        if not parsed:
            continue

        ctx = strip_tags(page_source[max(0, start - 1800): start + 800])
        ctxk = norm_key(ctx)
        score_home = 2 if home_key and home_key in ctxk else 0
        score_away = 2 if away_key and away_key in ctxk else 0
        candidates.append((start, score_home, score_away, parsed))

    if not candidates:
        return [], []

    home_best = None
    away_best = None

    for item in sorted(candidates, key=lambda x: (-(x[1]), x[0])):
        if item[1] > 0 and home_best is None:
            home_best = item
    for item in sorted(candidates, key=lambda x: (-(x[2]), x[0])):
        if item[2] > 0 and away_best is None:
            away_best = item

    sorted_by_pos = sorted(candidates, key=lambda x: x[0])
    if home_best is None and len(sorted_by_pos) >= 1:
        home_best = sorted_by_pos[0]
    if away_best is None and len(sorted_by_pos) >= 2:
        away_best = sorted_by_pos[1]

    home_rows = home_best[3] if home_best else []
    away_rows = away_best[3] if away_best else []

    if home_best and away_best and home_best[0] == away_best[0]:
        for cand in sorted_by_pos:
            if cand[0] != home_best[0]:
                away_rows = cand[3]
                break

    return home_rows, away_rows

def standings_full_to_splits(rows: List[StandingsFullRow]) -> Dict[str, Optional[SplitGFGA]]:
    mp: Dict[str, Optional[SplitGFGA]] = {"Total": None, "Home": None, "Away": None}
    for r in rows:
        if r.matches and r.scored is not None and r.conceded is not None:
            mp[r.ft] = SplitGFGA(r.matches, r.scored, r.conceded)
    return mp

def extract_same_league_code_from_page(page_source: str) -> Optional[str]:
    txt = strip_tags(page_source)
    m = LEAGUE_HDR_RE.search(txt)
    if m:
        cand = " ".join(m.group(1).split()[:2]).upper().strip()
        if cand:
            return cand

    toks = re.findall(r"\b([A-Z]{2,5})\s+([A-Z0-9]{2,5})\b", txt)
    for a, b in toks:
        cand = f"{a} {b}"
        if cand.startswith("GMT "):
            continue
        if LEAGUE_TOK_RE.match(a) and LEAGUE_TOK_RE.match(b):
            if a in {"ITA", "ENG", "ESP", "GER", "FRA", "NED", "POR", "TUR", "UEFA"}:
                return cand
    return None

# =========================
# PREVIOUS / H2H EXTRACT
# =========================
def filter_same_league(matches: List[MatchRow], league_code: Optional[str]) -> List[MatchRow]:
    if not matches:
        return []
    if not league_code:
        return matches[:]
    lk = norm_key(league_code)
    return [m for m in matches if norm_key(m.league) == lk]

def extract_previous_and_h2h_from_h2h_page(page_source: str, home_team: str, away_team: str) -> Tuple[List[MatchRow], List[MatchRow], List[MatchRow]]:
    prev_tables = section_tables_by_marker(page_source, "Previous Scores Statistics", max_tables=4)
    prev_home, prev_away = [], []
    if len(prev_tables) >= 1:
        prev_home = parse_matches_from_table_html(prev_tables[0])
    if len(prev_tables) >= 2:
        prev_away = parse_matches_from_table_html(prev_tables[1])

    h2h_markers = ["Head to Head Statistics", "Head to Head", "H2H Statistics", "H2H", "Head-to-Head", "Head2Head"]
    h2h_matches = []
    for mk in h2h_markers:
        tabs = section_tables_by_marker(page_source, mk, max_tables=3)
        if not tabs:
            continue
        for t in tabs:
            cand = parse_matches_from_table_html(t)
            pair_count = sum(1 for m in cand if is_h2h_pair(m, home_team, away_team))
            if pair_count >= 3:
                h2h_matches = cand
                break
        if h2h_matches:
            break

    if not h2h_matches:
        best = (0, [])
        for _, tbl in extract_tables_html(page_source):
            cand = parse_matches_from_table_html(tbl)
            if not cand:
                continue
            pair_count = sum(1 for m in cand if is_h2h_pair(m, home_team, away_team))
            if pair_count > best[0]:
                best = (pair_count, cand)
        if best[0] >= 3:
            h2h_matches = best[1]

    return prev_home, prev_away, h2h_matches

# =========================
# PREV STATS
# =========================
def build_prev_stats(team: str, matches: List[MatchRow]) -> TeamPrevStats:
    tkey = norm_key(team)
    st = TeamPrevStats(name=team)
    if not matches:
        return st

    home_ms = [m for m in matches if norm_key(m.home) == tkey]
    away_ms = [m for m in matches if norm_key(m.away) == tkey]

    def team_gf_ga(m: MatchRow) -> Tuple[int, int]:
        if norm_key(m.home) == tkey:
            return m.ft_home, m.ft_away
        return m.ft_away, m.ft_home

    gfs, gas = [], []
    for m in matches:
        gf, ga = team_gf_ga(m)
        gfs.append(gf)
        gas.append(ga)

    st.n_total = len(matches)
    st.gf_total = sum(gfs) / st.n_total if st.n_total else 0.0
    st.ga_total = sum(gas) / st.n_total if st.n_total else 0.0

    st.n_home = len(home_ms)
    if st.n_home:
        st.gf_home = sum(m.ft_home for m in home_ms) / st.n_home
        st.ga_home = sum(m.ft_away for m in home_ms) / st.n_home

    st.n_away = len(away_ms)
    if st.n_away:
        st.gf_away = sum(m.ft_away for m in away_ms) / st.n_away
        st.ga_away = sum(m.ft_home for m in away_ms) / st.n_away

    return st

# =========================
# ODDS PARSE (ROBUST)
# =========================
BOOK_PREF = ["Bet365", "Pinnacle", "SBOBET", "Sbobet", "188Bet", "12Bet", "Crown", "M88", "Ladbrokes", "Interwetten"]

def _safe_float(x: str) -> Optional[float]:
    try:
        x = (x or "").strip()
        if x in {"", "-", "—", "---"}:
            return None
        return float(x)
    except Exception:
        return None

def _valid_1x2_triplet(o1: float, ox: float, o2: float) -> bool:
    if not (1.01 <= o1 <= 50 and 1.01 <= ox <= 50 and 1.01 <= o2 <= 50):
        return False
    overround = (1.0 / o1) + (1.0 / ox) + (1.0 / o2)
    return 0.95 <= overround <= 1.30

def _book_rank(book: str) -> int:
    if not book:
        return 999
    for j, bn in enumerate(BOOK_PREF):
        if book.lower() == bn.lower():
            return j
    return 999

def parse_oddscomp_robust(odds_html: str) -> Tuple[Optional[Tuple[float, float, float, str]], Optional[Tuple[float, str]]]:
    if not odds_html:
        return None, None

    best_1x2 = None  # (rank, score, o1, ox, o2, book)
    best_ou = None   # (rank, line_score, line, book)

    for _, tbl in extract_tables_html(odds_html):
        rows = extract_table_rows_from_html(tbl)
        if not rows or len(rows) < 5:
            continue

        for r in rows:
            if not r or len(r) < 4:
                continue
            book = (r[0] or "").strip()
            if not book:
                continue
            if _safe_float(book) is not None:
                continue

            rank = _book_rank(book)

            floats: List[Tuple[int, float]] = []
            for i, cell in enumerate(r):
                val = _safe_float(cell)
                if val is None:
                    continue
                floats.append((i, val))

            only_odds = [(i, v) for (i, v) in floats if 1.01 <= v <= 50]
            if len(only_odds) >= 3:
                limit = min(len(only_odds), 12)
                for j in range(0, limit - 2):
                    o1 = only_odds[j][1]
                    ox = only_odds[j + 1][1]
                    o2 = only_odds[j + 2][1]
                    if _valid_1x2_triplet(o1, ox, o2):
                        overround = (1/o1) + (1/ox) + (1/o2)
                        score = abs(overround - 1.06)
                        cand = (rank, score, o1, ox, o2, book)
                        if best_1x2 is None or cand < best_1x2:
                            best_1x2 = cand

            for _, v in floats:
                if 1.25 <= v <= 4.75 and abs(v * 4 - round(v * 4)) < 1e-9:
                    line_score = abs(v - 2.75)
                    cand2 = (rank, line_score, v, book)
                    if best_ou is None or cand2 < best_ou:
                        best_ou = cand2

    out_1x2 = (best_1x2[2], best_1x2[3], best_1x2[4], best_1x2[5]) if best_1x2 else None
    out_ou = (best_ou[2], best_ou[3]) if best_ou else None
    return out_1x2, out_ou

# =========================
# LAMBDA MODEL
# =========================
def compute_component_standings(st_home: Dict[str, Optional[SplitGFGA]],
                                st_away: Dict[str, Optional[SplitGFGA]]) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    hh = st_home.get("Home") or st_home.get("Total")
    aa = st_away.get("Away") or st_away.get("Total")
    if not hh or not aa or hh.matches < 3 or aa.matches < 3:
        return None

    lam_h = (hh.gf_pg + aa.ga_pg) / 2.0
    lam_a = (aa.gf_pg + hh.ga_pg) / 2.0
    meta = {"home_split": {"matches": hh.matches, "gf_pg": hh.gf_pg, "ga_pg": hh.ga_pg},
            "away_split": {"matches": aa.matches, "gf_pg": aa.gf_pg, "ga_pg": aa.ga_pg}}
    return lam_h, lam_a, meta

def compute_component_previous(home_prev: TeamPrevStats, away_prev: TeamPrevStats) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    if home_prev.n_total < 3 or away_prev.n_total < 3:
        return None

    h_gf_home = home_prev.gf_home if home_prev.n_home >= 2 else home_prev.gf_total
    h_ga_home = home_prev.ga_home if home_prev.n_home >= 2 else home_prev.ga_total
    a_gf_away = away_prev.gf_away if away_prev.n_away >= 2 else away_prev.gf_total
    a_ga_away = away_prev.ga_away if away_prev.n_away >= 2 else away_prev.ga_total

    lam_h = (h_gf_home + a_ga_away) / 2.0
    lam_a = (a_gf_away + h_ga_home) / 2.0
    meta = {"home_prev": asdict(home_prev), "away_prev": asdict(away_prev)}
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

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in w.values())
    if s <= 1e-9:
        return {}
    return {k: max(0.0, v) / s for k, v in w.items()}

def apply_ou_sanity(lh: float, la: float, ou_line: Optional[float]) -> Tuple[float, float, Optional[str]]:
    if ou_line is None:
        return lh, la, None
    if not (1.25 <= ou_line <= 4.75):
        return lh, la, None

    total = lh + la
    if total <= 1e-9:
        return lh, la, None

    if abs(total - ou_line) > OU_SANITY_MAX_DIFF:
        new_total = (1.0 - OU_SANITY_BLEND) * total + OU_SANITY_BLEND * ou_line
        ratio = lh / total if total > 0 else 0.5
        lh2 = new_total * ratio
        la2 = new_total * (1.0 - ratio)
        note = f"OU sanity: {total:.2f} -> {new_total:.2f} (line={ou_line:.2f})"
        return lh2, la2, note

    return lh, la, None

def compute_lambdas(
    st_home: Dict[str, Optional[SplitGFGA]],
    st_away: Dict[str, Optional[SplitGFGA]],
    home_prev: TeamPrevStats,
    away_prev: TeamPrevStats,
    h2h_matches: List[MatchRow],
    home_team: str,
    away_team: str,
    ou_line: Optional[float]
) -> Tuple[float, float, Dict[str, Any]]:
    info: Dict[str, Any] = {"components": {}, "warnings": [], "weights_used": {}}
    components: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}

    st_comp = compute_component_standings(st_home, st_away)
    if st_comp:
        components["standings"] = st_comp

    prev_comp = compute_component_previous(home_prev, away_prev)
    if prev_comp:
        components["previous"] = prev_comp

    h2h_comp = compute_component_h2h(h2h_matches, home_team, away_team)
    if h2h_comp:
        components["h2h"] = h2h_comp

    w = {}
    if "standings" in components: w["standings"] = W_ST_BASE
    if "previous" in components:  w["previous"]  = W_PREV_BASE
    if "h2h" in components:       w["h2h"]       = W_H2H_BASE

    w_norm = normalize_weights(w)
    info["weights_used"] = w_norm

    if not w_norm:
        info["warnings"].append("Standings/Previous/H2H yok -> fallback λ=1.20/1.20 (düşük güven)")
        lh, la = 1.20, 1.20
    else:
        lh, la = 0.0, 0.0
        for k, wk in w_norm.items():
            ch, ca, meta = components[k]
            info["components"][k] = {"lam_home": ch, "lam_away": ca, "meta": meta}
            lh += wk * ch
            la += wk * ca

    lh2, la2, note = apply_ou_sanity(lh, la, ou_line)
    if note:
        info["warnings"].append(note)
    return lh2, la2, info

# =========================
# POISSON & MONTE CARLO
# =========================
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def build_score_matrix(lam_home: float, lam_away: float, max_g: int) -> Dict[Tuple[int, int], float]:
    mat: Dict[Tuple[int, int], float] = {}
    for h in range(max_g + 1):
        ph = poisson_pmf(h, lam_home)
        for a in range(max_g + 1):
            mat[(h, a)] = ph * poisson_pmf(a, lam_away)
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

def monte_carlo(lam_home: float, lam_away: float, n: int) -> Dict[str, Any]:
    rng = np.random.default_rng(42)
    hg = rng.poisson(lam_home, size=n)
    ag = rng.poisson(lam_away, size=n)
    total = hg + ag

    cnt = Counter(zip(hg.tolist(), ag.tolist()))
    top10 = cnt.most_common(10)
    top10_list = [(f"{h}-{a}", c / n) for (h, a), c in top10]

    def p(mask) -> float:
        return float(np.mean(mask))

    out = {
        "p": {
            "1": p(hg > ag),
            "X": p(hg == ag),
            "2": p(hg < ag),
            "BTTS": p((hg >= 1) & (ag >= 1)),
            "O2.5": p(total >= 3),
            "U2.5": p(total <= 2),
        },
        "TOP10": top10_list,
        "mini_sample": [f"{h}-{a}" for h, a in zip(
            rng.poisson(lam_home, size=MC_MINI_SAMPLE).tolist(),
            rng.poisson(lam_away, size=MC_MINI_SAMPLE).tolist()
        )]
    }
    return out

def model_agreement(p1: Dict[str, float], p2: Dict[str, float]) -> Tuple[float, str]:
    keys = ["1", "X", "2", "BTTS", "O2.5", "U2.5"]
    diffs = [abs(p1.get(k, 0) - p2.get(k, 0)) for k in keys]
    d = max(diffs) if diffs else 0.0
    if d <= 0.03: return d, "Çok iyi uyum"
    if d <= 0.06: return d, "İyi uyum"
    if d <= 0.10: return d, "Orta uyum"
    return d, "Zayıf uyum (belirsizlik yüksek)"

def blend_probs(p1: Dict[str, float], p2: Dict[str, float], alpha: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in set(list(p1.keys()) + list(p2.keys())):
        if k in p1 and k in p2:
            out[k] = alpha * p1[k] + (1.0 - alpha) * p2[k]
        else:
            out[k] = p1.get(k, p2.get(k, 0.0))
    return out

# =========================
# VALUE BET & KELLY
# =========================
def value_and_kelly(prob: float, odds: float) -> Tuple[float, float]:
    v = odds * prob - 1.0
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0:
        return v, 0.0
    k = (b * prob - q) / b
    return v, k

# =========================
# NET PREDICTIONS
# =========================
def generate_top_scores(mat: Dict[Tuple[int, int], float], topn: int = 3) -> List[str]:
    sorted_scores = sorted(mat.items(), key=lambda x: x[1], reverse=True)
    return [f"{h}-{a}" for (h, a), _ in sorted_scores[:topn]]

def net_ou(probs: Dict[str, float]) -> Tuple[str, float, str]:
    p_o25 = probs.get("O2.5", 0.0)
    p_u25 = probs.get("U2.5", 0.0)
    if p_o25 >= p_u25:
        return "2.5 ÜST", p_o25, confidence_label(p_o25)
    return "2.5 ALT", p_u25, confidence_label(p_u25)

def net_btts(probs: Dict[str, float]) -> Tuple[str, float, str]:
    p = probs.get("BTTS", 0.0)
    if p >= 0.5:
        return "VAR", p, confidence_label(p)
    return "YOK", 1.0 - p, confidence_label(1.0 - p)

# =========================
# TEAM EXTRACTION (title)
# =========================
def extract_teams_from_title(page_source: str) -> Tuple[str, str]:
    m = re.search(r"<title>(.*?)</title>", page_source or "", flags=re.IGNORECASE | re.DOTALL)
    title = strip_tags(m.group(1)) if m else strip_tags(page_source[:5000])
    title = " ".join(title.split()).strip()

    mm = re.search(r"(.+?)\s+vs\s+(.+?)(?:\s+-|\s+\||$)", title, flags=re.IGNORECASE)
    if mm:
        return clean_team_name(mm.group(1)), clean_team_name(mm.group(2))

    # fallback: text içinde ilk " vs "
    txt = strip_tags(page_source)
    for line in txt.splitlines():
        if " vs " in line.lower():
            mm2 = re.search(r"(.+?)\s+vs\s+(.+?)$", " ".join(line.split()), flags=re.IGNORECASE)
            if mm2:
                return clean_team_name(mm2.group(1)), clean_team_name(mm2.group(2))

    return "", ""

# =========================
# MAIN ANALYZE FUNCTION
# =========================
def analyze_nowgoal(match_url: str) -> Dict[str, Any]:
    match_id = extract_match_id(match_url)
    base = extract_base_domain(match_url)

    # 1) H2H sayfasını çek
    h2h_urls = build_h2h_urls(base, match_id)
    h2h_html, h2h_src = fetch_html(h2h_urls)
    if not h2h_html:
        raise RuntimeError("H2H sayfası çekilemedi (site engeli / domain / bağlantı).")

    home_team, away_team = extract_teams_from_title(h2h_html)
    if not home_team or not away_team:
        raise RuntimeError("Takım isimleri çıkarılamadı (title/HTML).")

    # 2) Standings + league code
    st_home_full, st_away_full = extract_standings_from_html(h2h_html, home_team, away_team)
    st_home = standings_full_to_splits(st_home_full)
    st_away = standings_full_to_splits(st_away_full)
    same_league_code = extract_same_league_code_from_page(h2h_html)

    # 3) Previous + H2H matches
    prev_home_raw, prev_away_raw, h2h_raw = extract_previous_and_h2h_from_h2h_page(h2h_html, home_team, away_team)

    prev_home = filter_same_league(prev_home_raw, same_league_code)[:RECENT_N]
    prev_away = filter_same_league(prev_away_raw, same_league_code)[:RECENT_N]

    h2h_pair = [m for m in h2h_raw if is_h2h_pair(m, home_team, away_team)]
    h2h_pair = sort_matches_desc(dedupe_matches(h2h_pair))
    h2h_same_league = filter_same_league(h2h_pair, same_league_code)
    h2h_used = h2h_same_league[:H2H_N] if len(h2h_same_league) >= 3 else h2h_pair[:H2H_N]

    # 4) Oddscomp çek + parse
    odds_urls = build_odds_urls(base, match_id)
    odds_html, odds_src = fetch_html(odds_urls)
    odds_1x2 = None
    ou_line = None
    odds_book = None
    ou_book = None

    if odds_html:
        pack_1x2, pack_ou = parse_oddscomp_robust(odds_html)
        if pack_1x2:
            o1, ox, o2, book = pack_1x2
            odds_1x2 = (o1, ox, o2)
            odds_book = book
        if pack_ou:
            ln, book2 = pack_ou
            ou_line = ln
            ou_book = book2

    # 5) Prev stats
    home_prev_stats = build_prev_stats(home_team, prev_home)
    away_prev_stats = build_prev_stats(away_team, prev_away)

    # 6) λ
    lam_home, lam_away, lambda_info = compute_lambdas(
        st_home=st_home,
        st_away=st_away,
        home_prev=home_prev_stats,
        away_prev=away_prev_stats,
        h2h_matches=h2h_used,
        home_team=home_team,
        away_team=away_team,
        ou_line=ou_line
    )
    total_lam = lam_home + lam_away

    # 7) Poisson + MC + Blend
    matrix_max = int(min(MAX_GOALS_CAP, max(10, math.ceil(max(lam_home, lam_away) + 6)))))
    poisson_mat = build_score_matrix(lam_home, lam_away, max_g=matrix_max)
    poisson_market = market_probs_from_matrix(poisson_mat)

    mc = monte_carlo(lam_home, lam_away, n=MC_RUNS)
    mc_market = mc["p"]

    diff, diff_label = model_agreement(poisson_market, mc_market)
    blended = blend_probs(poisson_market, mc_market, alpha=BLEND_ALPHA)

    # 8) Net predictions
    top_scores = generate_top_scores(poisson_mat, topn=3)
    ou_text, ou_prob, ou_conf = net_ou(blended)
    btts_text, btts_prob, btts_conf = net_btts(blended)

    # 9) Value/Kelly (sadece 1X2)
    value_rows = []
    qualified = []
    decision = "OYNAMA (oran verisi yok)" if not odds_1x2 else "OYNAMA (eşik yok)"

    if odds_1x2:
        o1, ox, o2 = odds_1x2
        for mkt, odds in [("1", o1), ("X", ox), ("2", o2)]:
            p = blended.get(mkt, 0.0)
            v, k = value_and_kelly(p, odds)
            qk = max(0.0, 0.25 * k)
            row = {"market": mkt, "prob": p, "odds": odds, "value": v, "kelly": k, "qkelly": qk}
            value_rows.append(row)
            if v >= VALUE_MIN and p >= PROB_MIN:
                qualified.append(row)

        if qualified:
            best = sorted(qualified, key=lambda r: r["value"], reverse=True)[0]
            if diff > 0.10:
                decision = f"RİSKLİ (model uyumu zayıf: {diff_label}) - {best['market']}"
            else:
                decision = f"OYNANABİLİR - {best['market']} (p={best['prob']:.3f}, odds={best['odds']:.2f}, value={best['value']:+.3f})"
        else:
            decision = "OYNAMA (eşik şartları sağlanmadı)"

    result = {
        "match_id": match_id,
        "source": {"h2h_src": h2h_src, "odds_src": odds_src},
        "teams": {"home": home_team, "away": away_team},
        "same_league_code": same_league_code,
        "previous_counts": {"home": len(prev_home), "away": len(prev_away)},
        "h2h_count": len(h2h_used),
        "odds": {"book_1x2": odds_book, "book_ou": ou_book, "1x2": odds_1x2, "ou_line": ou_line},
        "lambda": {"home": lam_home, "away": lam_away, "total": total_lam, "info": lambda_info},
        "model_agreement": {"diff": diff, "label": diff_label},
        "predictions": {
            "top_scores": top_scores,
            "ou": {"pick": ou_text, "prob": ou_prob, "confidence": ou_conf},
            "btts": {"pick": btts_text, "prob": btts_prob, "confidence": btts_conf},
            "decision": decision,
        },
        "probs_blended": {
            "1": blended.get("1", 0.0),
            "X": blended.get("X", 0.0),
            "2": blended.get("2", 0.0),
            "BTTS": blended.get("BTTS", 0.0),
            "O2.5": blended.get("O2.5", 0.0),
            "U2.5": blended.get("U2.5", 0.0),
        },
        "value_1x2": value_rows,
        "top10_mc": mc.get("TOP10", []),
    }
    return json_safe(result)

# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/analiz_et")
def analiz_et():
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or data.get("link") or "").strip()
        if not url:
            return jsonify({"ok": False, "error": "url/link alanı boş"}), 400

        res = analyze_nowgoal(url)
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
