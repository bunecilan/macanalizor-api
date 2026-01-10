# app.py
# -*- coding: utf-8 -*-

import re
import math
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS


# =========================
# CONFIG
# =========================
MC_RUNS = 10_000
MC_MINI_SAMPLE = 100

RECENT_N = 10
H2H_N = 10

VALUE_MIN = 0.05
PROB_MIN = 0.55

# Weights: standings + previous + h2h (normalize by availability)
W_ST_BASE = 0.55
W_PREV_BASE = 0.35
W_H2H_BASE = 0.10

MAX_GOALS_CAP = 12

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

DATE_ANY_RE = re.compile(r"\b(\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{2}-\d{2})\b")
SCORE_RE = re.compile(r"\b(\d{1,2})\s*-\s*(\d{1,2})(?!-\d{2,4})(?:\s*\((\d{1,2})\s*-\s*(\d{1,2})\))?\b")
LEAGUE_TOK_RE = re.compile(r"^[A-Z0-9]{2,6}$")
LEAGUE_HDR_RE = re.compile(r"\[([A-Z]{2,5}(?:\s+[A-Z0-9]{2,5})?)-\d+\]")  # [ITA D1-17]


# =========================
# APP
# =========================
app = Flask(__name__)
CORS(app)


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
    ft: str  # Total/Home/Away/Last 6
    matches: Optional[int]
    win: Optional[int]
    draw: Optional[int]
    loss: Optional[int]
    scored: Optional[int]
    conceded: Optional[int]
    pts: Optional[int]
    rank: Optional[int]


# =========================
# HELPERS
# =========================
def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def clean_team_name(s: str) -> str:
    s = " ".join((s or "").split()).strip()
    s = re.sub(r"\s+Live Score.*$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+Football Analysis.*$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+Preview.*$", "", s, flags=re.IGNORECASE).strip()
    return s.strip(" ,;-")

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
        key = (m.league, m.date, m.home, m.away, m.ft_home, m.ft_away)
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

def ascii_bar(p: float, width: int = 26) -> str:
    p = max(0.0, min(1.0, p))
    filled = int(round(p * width))
    return "█" * filled + "░" * (width - filled)

def confidence_label(p: float) -> str:
    if p >= 0.65:
        return "Yüksek"
    if p >= 0.55:
        return "Orta"
    return "Düşük"


# =========================
# FETCH HTML
# =========================
def fetch_html(url: str, timeout: int = 25) -> str:
    headers = {
        "User-Agent": UA,
        "Accept-Language": "en-US,en;q=0.9,tr-TR,tr;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "close",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text or ""


# =========================
# PARSE TEAMS
# =========================
def parse_teams_from_title(html: str) -> Tuple[str, str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.text if soup.title else "").strip()
    # Example: "Girona VS Osasuna - Football Analysis, Preview..."
    m = re.search(r"(.+?)\s+vs\s+(.+?)(?:\s+-|\s+\||$)", title, flags=re.IGNORECASE)
    if not m:
        raise ValueError("Takım isimlerini başlıktan çıkaramadım.")
    home = clean_team_name(m.group(1))
    away = clean_team_name(m.group(2))
    if not home or not away:
        raise ValueError("Takım isimlerini temizleyemedim.")
    return home, away, title


# =========================
# HTML TABLE HELPERS
# =========================
def table_to_rows(table) -> List[List[str]]:
    rows: List[List[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        txt = []
        for c in cells:
            t = " ".join(c.get_text(" ", strip=True).split())
            if t and t != "—":
                txt.append(t)
        if txt:
            rows.append(txt)
    return rows

def all_tables(html: str):
    soup = BeautifulSoup(html, "lxml")
    return soup.find_all("table")

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
    # date
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

    # score
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
    if ft_h > 9 or ft_a > 9:
        return None

    if score_idx - 1 < 0 or score_idx + 1 >= len(cells):
        return None

    home = clean_team_name(cells[score_idx - 1])
    away = clean_team_name(cells[score_idx + 1])
    if not home or not away:
        return None

    league = detect_league_from_cells(cells, date_idx)
    return MatchRow(league=league, date=date_val, home=home, away=away, ft_home=ft_h, ft_away=ft_a)

def parse_matches_from_rows(rows: List[List[str]]) -> List[MatchRow]:
    out: List[MatchRow] = []
    for cells in rows:
        m = parse_match_from_cells(cells)
        if m:
            out.append(m)
    return sort_matches_desc(dedupe_matches(out))


# =========================
# PARSE STANDINGS
# =========================
def _to_int(x: str) -> Optional[int]:
    try:
        x = (x or "").strip()
        if x in {"", "-", "—"}:
            return None
        return int(x)
    except Exception:
        return None

def parse_standings_rows(rows: List[List[str]]) -> List[StandRow]:
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

def standings_to_splits(rows: List[StandRow]) -> Dict[str, Optional[SplitGFGA]]:
    mp: Dict[str, Optional[SplitGFGA]] = {"Total": None, "Home": None, "Away": None}
    for r in rows:
        if r.matches and (r.scored is not None) and (r.conceded is not None):
            mp[r.ft] = SplitGFGA(matches=r.matches, gf=r.scored, ga=r.conceded)
    return mp

def extract_standings(html: str, home_team: str, away_team: str) -> Tuple[List[StandRow], List[StandRow]]:
    """
    HTML içindeki tüm tabloları tarar:
    - İçinde Matches/Win/Draw/Loss/Scored/Conceded gibi kelimeler olan tabloları aday alır
    - Home/Away takım adına yakın olanı seçmeye çalışır (yakınlık = üst DOM metni)
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")

    home_key = norm_key(home_team)
    away_key = norm_key(away_team)

    candidates = []
    for tbl in tables:
        txt = " ".join(tbl.get_text(" ", strip=True).split()).lower()
        need = ["matches", "win", "draw", "loss", "scored", "conceded"]
        if not all(k in txt for k in need):
            continue

        rows = table_to_rows(tbl)
        parsed = parse_standings_rows(rows)
        if not parsed:
            continue

        # context: önceki birkaç sibling/parent metni
        ctx_parts = []
        p = tbl.parent
        for _ in range(3):
            if not p:
                break
            ctx_parts.append(" ".join(p.get_text(" ", strip=True).split()))
            p = p.parent
        ctx = " ".join(ctx_parts)
        ctxk = norm_key(ctx)

        score_home = 2 if home_key and home_key in ctxk else 0
        score_away = 2 if away_key and away_key in ctxk else 0
        candidates.append((score_home, score_away, parsed))

    if not candidates:
        return [], []

    # best pick
    home_best = max(candidates, key=lambda x: x[0])
    away_best = max(candidates, key=lambda x: x[1])

    home_rows = home_best[2] if home_best[0] > 0 else candidates[0][2]
    away_rows = away_best[2] if away_best[1] > 0 else (candidates[1][2] if len(candidates) > 1 else candidates[0][2])

    return home_rows, away_rows


# =========================
# PARSE PREVIOUS + H2H
# =========================
def extract_tables_near_text(html: str, marker: str, max_tables: int = 4) -> List[List[List[str]]]:
    soup = BeautifulSoup(html, "lxml")
    marker_low = marker.lower()
    hits = []
    # metin içeren elementleri tarayıp marker geçen yeri bul
    for el in soup.find_all(text=True):
        t = (el.strip() if el else "")
        if not t:
            continue
        if marker_low in t.lower():
            hits.append(el)
            break

    if not hits:
        return []

    # marker bulunduğu yerden sonra gelen tablolar
    el = hits[0].parent
    tables = []
    # aynı parent zincirinde ileri doğru tablo ara
    cur = el
    steps = 0
    while cur and steps < 120 and len(tables) < max_tables:
        nxt = cur.find_next("table")
        if not nxt:
            break
        tables.append(table_to_rows(nxt))
        cur = nxt
        steps += 1

    return tables

def extract_previous_and_h2h(html: str, home_team: str, away_team: str) -> Tuple[List[MatchRow], List[MatchRow], List[MatchRow]]:
    prev_tables = extract_tables_near_text(html, "Previous Scores Statistics", max_tables=4)
    prev_home = parse_matches_from_rows(prev_tables[0]) if len(prev_tables) >= 1 else []
    prev_away = parse_matches_from_rows(prev_tables[1]) if len(prev_tables) >= 2 else []

    # H2H: en çok pair-match içeren tabloyu seç
    best_pair = (0, [])
    for tbl in all_tables(html):
        rows = table_to_rows(tbl)
        cand = parse_matches_from_rows(rows)
        if not cand:
            continue
        pair_count = sum(1 for m in cand if is_h2h_pair(m, home_team, away_team))
        if pair_count > best_pair[0]:
            best_pair = (pair_count, cand)

    h2h = best_pair[1] if best_pair[0] >= 3 else []
    return prev_home, prev_away, h2h


# =========================
# LEAGUE CODE (optional)
# =========================
def extract_same_league_code(html: str) -> Optional[str]:
    text = BeautifulSoup(html, "lxml").get_text("\n", strip=True)
    m = LEAGUE_HDR_RE.search(text)
    if m:
        cand = " ".join(m.group(1).split()[:2]).upper().strip()
        return cand or None
    return None

def filter_same_league(matches: List[MatchRow], league_code: Optional[str]) -> List[MatchRow]:
    if not matches or not league_code:
        return matches[:]
    lk = norm_key(league_code)
    return [m for m in matches if norm_key(m.league) == lk]


# =========================
# ODDS COMP (optional)
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

def _book_rank(book: str) -> int:
    if not book:
        return 999
    for j, bn in enumerate(BOOK_PREF):
        if book.lower() == bn.lower():
            return j
    return 999

def _valid_1x2_triplet(o1: float, ox: float, o2: float) -> bool:
    if not (1.01 <= o1 <= 50 and 1.01 <= ox <= 50 and 1.01 <= o2 <= 50):
        return False
    overround = (1.0 / o1) + (1.0 / ox) + (1.0 / o2)
    return 0.95 <= overround <= 1.30

def fetch_oddscomp(base: str, match_id: str) -> str:
    urls = [
        f"{base}/oddscomp/{match_id}",
        f"https://www.nowgoal26.com/oddscomp/{match_id}",
        f"https://live.nowgoal26.com/oddscomp/{match_id}",
    ]
    for u in urls:
        try:
            html = fetch_html(u, timeout=20)
            # kaba doğrulama: çok sayıda 1.xx görünüyor mu?
            if len(re.findall(r"\b[12]\.\d{2}\b", html)) >= 10:
                return html
        except Exception:
            continue
    return ""

def parse_oddscomp(odds_html: str) -> Tuple[Optional[Tuple[float, float, float, str]], Optional[Tuple[float, str]]]:
    if not odds_html:
        return None, None

    soup = BeautifulSoup(odds_html, "lxml")
    best_1x2 = None  # (rank, score, o1, ox, o2, book)
    best_ou = None   # (rank, line_score, line, book)

    for tbl in soup.find_all("table"):
        rows = table_to_rows(tbl)
        if len(rows) < 5:
            continue
        for r in rows:
            if len(r) < 4:
                continue
            book = (r[0] or "").strip()
            if not book or _safe_float(book) is not None:
                continue
            rank = _book_rank(book)

            floats = []
            for cell in r:
                v = _safe_float(cell)
                if v is not None:
                    floats.append(v)

            # 1X2 sliding triplets
            odds_only = [v for v in floats if 1.01 <= v <= 50]
            for j in range(0, min(len(odds_only), 12) - 2):
                o1, ox, o2 = odds_only[j], odds_only[j+1], odds_only[j+2]
                if _valid_1x2_triplet(o1, ox, o2):
                    overround = (1/o1) + (1/ox) + (1/o2)
                    score = abs(overround - 1.06)
                    cand = (rank, score, o1, ox, o2, book)
                    if best_1x2 is None or cand < best_1x2:
                        best_1x2 = cand

            # OU line candidates 1.25-4.75 quarters
            for v in floats:
                if 1.25 <= v <= 4.75 and abs(v * 4 - round(v * 4)) < 1e-9:
                    line_score = abs(v - 2.75)
                    cand2 = (rank, line_score, v, book)
                    if best_ou is None or cand2 < best_ou:
                        best_ou = cand2

    out_1x2 = (best_1x2[2], best_1x2[3], best_1x2[4], best_1x2[5]) if best_1x2 else None
    out_ou = (best_ou[2], best_ou[3]) if best_ou else None
    return out_1x2, out_ou


# =========================
# MODEL: LAMBDA
# =========================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in w.values())
    if s <= 1e-9:
        return {}
    return {k: max(0.0, v) / s for k, v in w.items()}

def compute_component_standings(st_home: Dict[str, Optional[SplitGFGA]], st_away: Dict[str, Optional[SplitGFGA]]):
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

def avg_goals_for_team(matches: List[MatchRow], team: str) -> Tuple[float, float]:
    """returns (gf_avg, ga_avg)"""
    tkey = norm_key(team)
    if not matches:
        return 0.0, 0.0
    gf, ga = [], []
    for m in matches:
        if norm_key(m.home) == tkey:
            gf.append(m.ft_home); ga.append(m.ft_away)
        elif norm_key(m.away) == tkey:
            gf.append(m.ft_away); ga.append(m.ft_home)
    if not gf:
        return 0.0, 0.0
    return float(sum(gf))/len(gf), float(sum(ga))/len(ga)

def compute_component_previous(prev_home: List[MatchRow], prev_away: List[MatchRow], home_team: str, away_team: str):
    if len(prev_home) < 3 or len(prev_away) < 3:
        return None
    h_gf, h_ga = avg_goals_for_team(prev_home, home_team)
    a_gf, a_ga = avg_goals_for_team(prev_away, away_team)
    lam_h = (h_gf + a_ga) / 2.0
    lam_a = (a_gf + h_ga) / 2.0
    meta = {"home_prev_n": len(prev_home), "away_prev_n": len(prev_away), "home_gf": h_gf, "home_ga": h_ga, "away_gf": a_gf, "away_ga": a_ga}
    return lam_h, lam_a, meta

def compute_component_h2h(h2h: List[MatchRow], home_team: str, away_team: str):
    if len(h2h) < 3:
        return None
    hk, ak = norm_key(home_team), norm_key(away_team)
    used = h2h[:H2H_N]
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
    meta = {"matches": len(hg)}
    return lam_h, lam_a, meta

def compute_lambdas(st_home, st_away, prev_home, prev_away, h2h, home_team, away_team) -> Tuple[float, float, Dict[str, Any]]:
    info: Dict[str, Any] = {"components": {}, "weights_used": {}, "warnings": []}
    components = {}

    st = compute_component_standings(st_home, st_away)
    if st: components["standings"] = st
    pr = compute_component_previous(prev_home, prev_away, home_team, away_team)
    if pr: components["previous"] = pr
    hh = compute_component_h2h(h2h, home_team, away_team)
    if hh: components["h2h"] = hh

    w = {}
    if "standings" in components: w["standings"] = W_ST_BASE
    if "previous" in components:  w["previous"] = W_PREV_BASE
    if "h2h" in components:       w["h2h"] = W_H2H_BASE

    w_norm = normalize_weights(w)
    info["weights_used"] = w_norm

    if not w_norm:
        info["warnings"].append("Standings/Previous/H2H yeterli değil -> λ fallback=1.20/1.20 (düşük güven)")
        return 1.20, 1.20, info

    lh = la = 0.0
    for k, wk in w_norm.items():
        ch, ca, meta = components[k]
        info["components"][k] = {"lam_home": ch, "lam_away": ca, "meta": meta}
        lh += wk * ch
        la += wk * ca

    # sanity: aşırı uçuksa yumuşat
    if lh > 3.5 or la > 3.5:
        info["warnings"].append("λ aşırı yüksek göründü; veriler uçuk olabilir (düşük güven).")
        lh = min(lh, 3.5)
        la = min(la, 3.5)

    return lh, la, info


# =========================
# POISSON + MC
# =========================
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def build_score_matrix(lh: float, la: float, max_g: int) -> Dict[Tuple[int, int], float]:
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

    for ln in [0.5, 1.5]:
        need = int(math.floor(ln) + 1)
        out[f"H_O{ln}"] = sum(p for (h, a), p in mat.items() if h >= need)
        out[f"A_O{ln}"] = sum(p for (h, a), p in mat.items() if a >= need)

    return out

def monte_carlo(lh: float, la: float, n: int) -> Dict[str, Any]:
    rng = np.random.default_rng(42)
    hg = rng.poisson(lh, size=n)
    ag = rng.poisson(la, size=n)
    total = hg + ag

    cnt = Counter(zip(hg.tolist(), ag.tolist()))
    top10 = cnt.most_common(10)
    top10_list = [(f"{h}-{a}", c / n * 100.0) for (h, a), c in top10]

    dist_total = Counter(total.tolist())
    total_bins = {str(k): dist_total.get(k, 0) / n * 100.0 for k in range(0, 5)}
    total_bins["5+"] = sum(v for kk, v in dist_total.items() if kk >= 5) / n * 100.0

    def p(mask) -> float:
        return float(np.mean(mask))

    out = {
        "p": {
            "1": p(hg > ag),
            "X": p(hg == ag),
            "2": p(hg < ag),
            "BTTS": p((hg >= 1) & (ag >= 1)),
            "O0.5": p(total >= 1),
            "O1.5": p(total >= 2),
            "O2.5": p(total >= 3),
            "O3.5": p(total >= 4),
        },
        "TOTAL_DIST": total_bins,
        "TOP10": top10_list,
    }
    out["p"]["U0.5"] = 1.0 - out["p"]["O0.5"]
    out["p"]["U1.5"] = 1.0 - out["p"]["O1.5"]
    out["p"]["U2.5"] = 1.0 - out["p"]["O2.5"]
    out["p"]["U3.5"] = 1.0 - out["p"]["O3.5"]

    mini_hg = rng.poisson(lh, size=MC_MINI_SAMPLE)
    mini_ag = rng.poisson(la, size=MC_MINI_SAMPLE)
    out["mini_sample"] = [f"{h}-{a}" for h, a in zip(mini_hg.tolist(), mini_ag.tolist())]
    return out

def model_agreement(p1: Dict[str, float], p2: Dict[str, float]) -> Tuple[float, str]:
    keys = ["1", "X", "2", "BTTS", "O2.5", "U2.5"]
    diffs = [abs(p1.get(k, 0) - p2.get(k, 0)) for k in keys]
    d = max(diffs) if diffs else 0.0
    if d <= 0.03:
        return d, "Çok iyi uyum"
    elif d <= 0.06:
        return d, "İyi uyum"
    elif d <= 0.10:
        return d, "Orta uyum"
    else:
        return d, "Zayıf uyum (belirsizlik yüksek)"

def blend_probs(p1: Dict[str, float], p2: Dict[str, float], alpha: float = 0.50) -> Dict[str, float]:
    out = {}
    for k in set(list(p1.keys()) + list(p2.keys())):
        out[k] = alpha * p1.get(k, 0.0) + (1.0 - alpha) * p2.get(k, 0.0)
    return out


# =========================
# VALUE + KELLY
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
# RESULT
# =========================
def determine_tempo(lh: float, la: float) -> str:
    total = lh + la
    if total < 2.3:
        return "Düşük"
    elif total < 2.9:
        return "Orta"
    return "Yüksek"

def top_scores(mat: Dict[Tuple[int, int], float], n: int = 7) -> List[Tuple[str, float]]:
    items = sorted(mat.items(), key=lambda x: x[1], reverse=True)[:n]
    return [(f"{h}-{a}", p) for (h, a), p in items]

def net_ou(b: Dict[str, float]) -> Tuple[str, float, str]:
    p_o = b.get("O2.5", 0.0)
    p_u = b.get("U2.5", 0.0)
    if p_o >= p_u:
        return "2.5 ÜST", p_o, confidence_label(p_o)
    return "2.5 ALT", p_u, confidence_label(p_u)

def net_btts(b: Dict[str, float]) -> Tuple[str, float, str]:
    p_var = b.get("BTTS", 0.0)
    p_yok = 1.0 - p_var
    if p_var >= p_yok:
        return "VAR", p_var, confidence_label(p_var)
    return "YOK", p_yok, confidence_label(p_yok)

def pick_bets(blended: Dict[str, float], odds_1x2: Optional[Tuple[float, float, float]]) -> Dict[str, Any]:
    if not odds_1x2:
        return {
            "note": "Oran yok → value/kelly hesaplanmadı. Oranları gönderirsen hesaplarım.",
            "qualified": [],
            "decision": "OYNAMA (oran verisi eksik)"
        }

    o1, ox, o2 = odds_1x2
    qualified = []
    rows = []
    for mkt, o in [("1", o1), ("X", ox), ("2", o2)]:
        p = blended.get(mkt, 0.0)
        v, k = value_and_kelly(p, o)
        qk = max(0.0, 0.25 * k)
        rows.append({"market": mkt, "p": p, "odds": o, "value": v, "kelly": k, "qkelly": qk})
        if v >= VALUE_MIN and p >= PROB_MIN:
            qualified.append({"market": mkt, "p": p, "odds": o, "value": v, "qkelly": qk})

    # decision
    if not qualified:
        decision = "OYNAMA (eşikler sağlanmadı)"
    else:
        best = sorted(qualified, key=lambda x: x["value"], reverse=True)[0]
        decision = f"OYNANABİLİR → {best['market']} (p={best['p']*100:.1f}%, oran={best['odds']:.2f}, value={best['value']:+.3f}, ¼Kelly={best['qkelly']:.3f})"

    return {"rows": rows, "qualified": qualified, "decision": decision}


def build_report(payload: Dict[str, Any]) -> str:
    """Android ekranda direkt göstermek istersen diye, okunur Türkçe rapor."""
    teams = payload["teams"]
    lh = payload["lambda"]["home"]
    la = payload["lambda"]["away"]
    total = lh + la
    tempo = payload["predictions"]["tempo"]

    b = payload["blended_probs"]
    ou_name, ou_p, ou_conf = net_ou(b)
    btts_name, btts_p, btts_conf = net_btts(b)

    lines = []
    lines.append(f"MAÇ: {teams['home']} vs {teams['away']}")
    lines.append(f"λ (beklenen gol): Ev={lh:.3f} Dep={la:.3f} Toplam={total:.3f}")
    lines.append(f"Tempo: {tempo}")
    lines.append("")
    lines.append("BLEND OLASILIKLARI (Poisson+MC):")
    for k in ["1","X","2","BTTS","O2.5","U2.5","O3.5","U3.5"]:
        p = b.get(k, 0.0)
        lines.append(f"  {k:5s} [{ascii_bar(p)}] {p*100:5.1f}% ({confidence_label(p)})")
    lines.append("")
    lines.append("EN OLASI 7 SKOR (Poisson):")
    for sc, p in payload["poisson_top_scores"]:
        lines.append(f"  {sc:5s}  {p*100:5.2f}%")
    lines.append("")
    lines.append(f"NET ALT/ÜST: {ou_name} → {ou_p*100:.1f}% ({ou_conf})")
    lines.append(f"NET BTTS   : {btts_name} → {btts_p*100:.1f}% ({btts_conf})")
    lines.append(f"NET SKOR   : {payload['predictions']['scores'][0]} (alt: {payload['predictions']['scores'][1]}, {payload['predictions']['scores'][2]})")
    lines.append("")
    lines.append(f"SON KARAR: {payload['value_bet']['decision']}")
    return "\n".join(lines)


# =========================
# MAIN ANALYZE FUNCTION
# =========================
def analyze_nowgoal(url: str, user_odds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    match_id = extract_match_id(url)
    base = extract_base_domain(url)
    h2h_url = f"{base}/match/h2h-{match_id}"

    html = fetch_html(h2h_url)
    home_team, away_team, title = parse_teams_from_title(html)

    # standings
    st_home_rows, st_away_rows = extract_standings(html, home_team, away_team)
    st_home = standings_to_splits(st_home_rows)
    st_away = standings_to_splits(st_away_rows)

    # league code (optional)
    league_code = extract_same_league_code(html)

    # previous + h2h
    prev_home_raw, prev_away_raw, h2h_raw = extract_previous_and_h2h(html, home_team, away_team)

    prev_home = filter_same_league(prev_home_raw, league_code)[:RECENT_N]
    prev_away = filter_same_league(prev_away_raw, league_code)[:RECENT_N]
    h2h_pair = [m for m in h2h_raw if is_h2h_pair(m, home_team, away_team)]
    h2h_used = sort_matches_desc(dedupe_matches(h2h_pair))[:H2H_N]

    # lambdas
    lh, la, lambda_info = compute_lambdas(st_home, st_away, prev_home, prev_away, h2h_used, home_team, away_team)

    # poisson
    matrix_max = int(min(MAX_GOALS_CAP, max(10, math.ceil(max(lh, la) + 6))))
    mat = build_score_matrix(lh, la, matrix_max)
    poisson_market = market_probs_from_matrix(mat)
    poisson_top = top_scores(mat, 7)

    # monte carlo
    mc = monte_carlo(lh, la, MC_RUNS)
    mc_market = mc["p"]

    # blend
    diff, diff_label = model_agreement(poisson_market, mc_market)
    blended = blend_probs(poisson_market, mc_market, alpha=0.50)

    # odds: priority user provided; else try oddscomp
    odds_1x2 = None
    odds_source = None

    if user_odds and all(k in user_odds for k in ["1", "X", "2"]):
        odds_1x2 = (float(user_odds["1"]), float(user_odds["X"]), float(user_odds["2"]))
        odds_source = "user"
    else:
        odds_html = fetch_oddscomp(base, match_id)
        pack_1x2, _ou = parse_oddscomp(odds_html)
        if pack_1x2:
            o1, ox, o2, book = pack_1x2
            odds_1x2 = (o1, ox, o2)
            odds_source = f"oddscomp:{book}"

    # predictions
    tempo = determine_tempo(lh, la)
    scores3 = [sc for sc, _ in top_scores(mat, 3)]
    ou_name, ou_p, ou_conf = net_ou(blended)
    btts_name, btts_p, btts_conf = net_btts(blended)

    vb = pick_bets(blended, odds_1x2)

    payload = {
        "match_id": match_id,
        "url": h2h_url,
        "title": title,
        "teams": {"home": home_team, "away": away_team},
        "same_league_code": league_code,
        "data_quality": {
            "standings_found": bool(st_home_rows) and bool(st_away_rows),
            "prev_home_n": len(prev_home),
            "prev_away_n": len(prev_away),
            "h2h_n": len(h2h_used),
        },
        "standings_home": [asdict(x) for x in st_home_rows],
        "standings_away": [asdict(x) for x in st_away_rows],
        "previous": {
            "home": [asdict(x) for x in prev_home],
            "away": [asdict(x) for x in prev_away],
        },
        "h2h_used": [asdict(x) for x in h2h_used],
        "lambda": {"home": lh, "away": la, "total": lh + la, "info": lambda_info},
        "poisson_probs": poisson_market,
        "poisson_top_scores": poisson_top,
        "mc": mc,
        "model_agreement": {"diff": diff, "label": diff_label},
        "blended_probs": blended,
        "odds": {"1x2": odds_1x2, "source": odds_source},
        "value_bet": vb,
        "predictions": {
            "tempo": tempo,
            "scores": scores3 if len(scores3) == 3 else (scores3 + ["?","?"])[:3],
            "net_ou": {"pick": ou_name, "p": ou_p, "confidence": ou_conf},
            "net_btts": {"pick": btts_name, "p": btts_p, "confidence": btts_conf},
        },
    }

    payload["report"] = build_report(payload)
    return payload


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/analyze")
def analyze():
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        if not url:
            return jsonify({"ok": False, "error": "url zorunlu"}), 400

        odds = data.get("odds")
        if odds is not None and not isinstance(odds, dict):
            return jsonify({"ok": False, "error": "odds dict olmalı (örn: {\"1\":2.1,\"X\":3.3,\"2\":3.6})"}), 400

        out = analyze_nowgoal(url, user_odds=odds)
        return jsonify({"ok": True, "result": out})

    except requests.HTTPError as e:
        return jsonify({"ok": False, "error": f"HTTP hata: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": f"Hata: {str(e)}"}), 500
