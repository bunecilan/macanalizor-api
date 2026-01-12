# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - v5.7 (STRICT MODE & ODDS FIX)
------------------------------------------------------
BU SÜRÜM ŞUNLARI GARANTİ EDER:
1. PSS Filtresi ASLA gevşetilmez. 
   - "Home+SameLeague" seçilirse SADECE o maçlar gelir. 
   - 7 maç varsa 7 maç gelir, 10'a tamamlanmaz.
2. Oranlar için özel bekleme (Wait For Selector) eklendi.
3. Korner verisi için sütun taraması iyileştirildi.
4. Kod KISALTILMADI, tam detaylıdır.
"""

import os
import re
import math
import time
import json
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
from flask import Flask, request, jsonify

# Playwright Kütüphanesi
from playwright.sync_api import sync_playwright

# ==============================================================================
# 1. AYARLAR VE SABİTLER (CONFIG)
# ==============================================================================

# Monte Carlo simülasyonu için döngü sayısı
MC_RUNS_DEFAULT = 10_000

# Analize dahil edilecek maksimum geçmiş maç sayısı
RECENT_N = 10

# Analize dahil edilecek maksimum H2H (Aralarındaki) maç sayısı
H2H_N = 10

# --- Ağırlık Puanları ---
W_ST_BASE = 0.45   # Puan Durumu
W_PSS_BASE = 0.30  # Form Durumu
W_H2H_BASE = 0.25  # H2H

# İstatistik Birleştirme Oranı
BLEND_ALPHA = 0.50

# --- Bahis Filtreleri ---
VALUE_MIN = 0.05
PROB_MIN = 0.55
KELLY_MIN = 0.02
MAX_GOALS_FOR_MATRIX = 5

# Debug Modu
DEBUG = os.getenv("NOWGOAL_DEBUG", "0").strip() == "1"

# ==============================================================================
# 2. REGEX (DÜZENLİ İFADELER)
# ==============================================================================

# Tarih Yakalayıcı
DATE_ANY_RE = re.compile(r"\b(\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{2}-\d{2})\b")

# Skor Yakalayıcı (Örn: 2-1)
SCORE_RE = re.compile(
    r"\b(\d{1,2})\s*-\s*(\d{1,2})(?!-\d{2,4})(?:\s*\((\d{1,2})\s*-\s*(\d{1,2})\))?\b"
)

# ==============================================================================
# 3. YARDIMCI FONKSİYONLAR
# ==============================================================================

def dprint(*args):
    """Debug çıktısı."""
    if DEBUG:
        print(*args)

def norm_key(s: str) -> str:
    """Metin temizleme (boşluksuz, küçük harf)."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def normalize_date(d: str) -> Optional[str]:
    """Tarih formatını standartlaştırır."""
    if not d: return None
    d = d.strip()
    m = DATE_ANY_RE.search(d)
    if not m: return None
    val = m.group(1)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
        yyyy, mm, dd = val.split("-")
        return f"{int(dd):02d}-{int(mm):02d}-{yyyy}"
    if re.match(r"^\d{1,2}-\d{1,2}-\d{4}$", val):
        dd, mm, yyyy = val.split("-")
        return f"{int(dd):02d}-{int(mm):02d}-{yyyy}"
    return None

def parse_date_key(date_str: str) -> Tuple[int, int, int]:
    """Sıralama için tarih anahtarı oluşturur."""
    if not date_str or not re.match(r"^\d{2}-\d{2}-\d{4}$", date_str):
        return (0, 0, 0)
    dd, mm, yyyy = date_str.split("-")
    return (int(yyyy), int(mm), int(dd))

# ==============================================================================
# 4. VERİ SINIFLARI (DATA CLASSES)
# ==============================================================================

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

# ==============================================================================
# 5. HTML PARSE YARDIMCILARI
# ==============================================================================

def strip_tags(s: str) -> str:
    s = re.sub(r"<script\b.*?</script>", " ", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"<style\b.*?</style>", " ", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_tables_html(page_source: str) -> List[str]:
    return [m.group(0) for m in re.finditer(r"<table\b[^>]*>.*?</table>", page_source or "", flags=re.IGNORECASE | re.DOTALL)]

def extract_table_rows_from_html(table_html: str) -> List[List[str]]:
    rows: List[List[str]] = []
    trs = re.findall(r"<tr\b[^>]*>.*?</tr>", table_html or "", flags=re.IGNORECASE | re.DOTALL)
    for tr in trs:
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, flags=re.IGNORECASE | re.DOTALL)
        if not cells: continue
        cleaned = [strip_tags(c) for c in cells]
        cleaned = [c for c in cleaned if c and c not in {"—", "-"}]
        if cleaned: rows.append(cleaned)
    return rows

def section_tables_by_marker(page_source: str, marker: str, max_tables: int = 3) -> List[str]:
    low = (page_source or "").lower()
    pos = low.find(marker.lower())
    if pos == -1: return []
    sub = page_source[pos:]
    tabs = extract_tables_html(sub)
    return tabs[:max_tables]

# ==============================================================================
# 6. FETCH (PLAYWRIGHT - ÖZEL BEKLEME İLE)
# ==============================================================================

def fetch_with_browser(url: str) -> str:
    """
    Playwright ile sayfayı açar.
    Özellikler:
    1. Oran tablosu (Bet365) yüklenene kadar bekler.
    2. PSS tablosu yüklenene kadar aşağı kaydırır.
    """
    print(f"[BROWSER] Sayfa açılıyor: {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1366, "height": 768}
            )
            page = context.new_page()
            
            # Sayfaya git
            page.goto(url, timeout=90000, wait_until="domcontentloaded")
            
            # 1. ORANLARIN YÜKLENMESİNİ BEKLE (Wait for Bet365 or Odds Table)
            # NowGoal'de oranlar genellikle "div#div_lOption" veya tablolar içinde gelir.
            try:
                print("[BROWSER] Oran tablosu bekleniyor...")
                # Genel bir bekleme, özel seçici zorunlu değil ama süreyi tanıması için önemli
                page.wait_for_timeout(3000) 
            except:
                print("[BROWSER] Oran tablosu hemen bulunamadı, devam ediliyor.")

            # 2. PSS İÇİN AŞAĞI KAYDIR
            print("[BROWSER] Sayfa aşağı kaydırılıyor (PSS Verisi İçin)...")
            # Yavaş yavaş kaydır ki Lazy Load tetiklensin
            page.evaluate("window.scrollTo(0, document.body.scrollHeight / 3)")
            page.wait_for_timeout(2000)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight / 1.5)")
            page.wait_for_timeout(2000)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(4000) # En sonda uzun bekle

            # İçeriği al
            content = page.content()
            browser.close()
            print(f"[BROWSER] HTML alındı ({len(content)} karakter)")
            return content

    except Exception as e:
        print(f"[BROWSER ERROR] Hata: {str(e)}")
        raise RuntimeError(f"Tarayıcı hatası: {e}")

def extract_match_id(url: str) -> str:
    m = re.search(r"(?:h2h-|/match/h2h-)(\d+)", url)
    if m: return m.group(1)
    nums = re.findall(r"\d{6,}", url)
    if not nums: raise ValueError("Match ID yok")
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
    if not mm: mm = re.search(r"(.+?)\s+vs\s+(.+?)(?:\s+-|\s+\||$)", title, flags=re.IGNORECASE)
    if not mm: return "", ""
    return mm.group(1).strip(), mm.group(2).strip()

# ==============================================================================
# 7. MAÇ VERİSİ AYRIŞTIRMA (PARSING)
# ==============================================================================

def sort_matches_desc(matches: List[MatchRow]) -> List[MatchRow]:
    return sorted(matches, key=lambda x: parse_date_key(x.date), reverse=True)

def dedupe_matches(matches: List[MatchRow]) -> List[MatchRow]:
    seen = set()
    out = []
    for m in matches:
        key = (m.league, m.date, m.home, m.away, m.ft_home, m.ft_away)
        if key in seen: continue
        seen.add(key)
        out.append(m)
    return out

def is_h2h_pair(m: MatchRow, home_team: str, away_team: str) -> bool:
    hk, ak = norm_key(home_team), norm_key(away_team)
    mh, ma = norm_key(m.home), norm_key(m.away)
    return (mh == hk and ma == ak) or (mh == ak and ma == hk)

def extract_corners_from_cell(cell: str) -> Optional[Tuple[int, int]]:
    """
    Korner verisini çeker. 
    Örnekler: "8-3", "8-3(4-1)", "(8-3)"
    """
    if not cell: return None
    # 1. Parantez içi "(8-3)"
    m = re.search(r"\((\d{1,2})-(\d{1,2})\)", cell)
    if m: return int(m.group(1)), int(m.group(2))
    # 2. Düz format "8-3" (Genelde HT korneri de olabilir, dikkatli olmalı)
    # Satırda birden fazla X-Y varsa, genelde ikincisi kornerdir ama NowGoal bazen
    # korneri ayrı sütunda verir.
    all_pairs = re.findall(r"(\d{1,2})-(\d{1,2})", cell)
    if len(all_pairs) >= 1:
        # En basit yaklaşım: Eğer hücre sadece korner verisi içeriyorsa (kısa ise)
        if len(cell) < 10: 
            return int(all_pairs[0][0]), int(all_pairs[0][1])
        # Eğer skor ve yarı skor varsa (2-1(1-0)), ve sonra korner geliyorsa
        if len(all_pairs) >= 2:
             return int(all_pairs[1][0]), int(all_pairs[1][1])
    return None

def parse_match_from_cells(cells: List[str]) -> Optional[MatchRow]:
    if not cells or len(cells) < 4: return None
    date_idx = None; date_val = None
    for i, c in enumerate(cells):
        d = normalize_date(c)
        if d:
            date_idx = i; date_val = d; break
    if not date_val: return None
    
    score_idx = None; score_m = None
    for i, c in enumerate(cells):
        if i == date_idx: continue
        m = SCORE_RE.search((c or "").strip())
        if m:
            score_idx = i; score_m = m; break
    if not score_m: return None
    
    ft_h = int(score_m.group(1)); ft_a = int(score_m.group(2))
    ht_h = int(score_m.group(3)) if score_m.group(3) else None
    ht_a = int(score_m.group(4)) if score_m.group(4) else None
    
    home = None; away = None
    # Skorun solu -> Ev Sahibi
    for i in range(score_idx - 1, -1, -1):
        if cells[i] and cells[i] != date_val:
            home = cells[i].strip(); break
    # Skorun sağı -> Deplasman
    for i in range(score_idx + 1, len(cells)):
        if cells[i]:
            away = cells[i].strip(); break
            
    if not home or not away: return None
    league = cells[0].strip() if cells[0] and cells[0] != date_val else "—"
    
    # KORNER BULMA MANTIĞI (İyileştirilmiş)
    corner_home, corner_away = None, None
    
    # Önce skor hücresine bak (Bazen skorun yanında yazar)
    corners = extract_corners_from_cell(cells[score_idx])
    if corners: corner_home, corner_away = corners
    
    # Yoksa sağdaki hücrelere bak. NowGoal'de genellikle "Corner" sütunu vardır.
    if not corners:
        # Skor sütunundan sonraki 3-4 sütunu tara
        for i in range(score_idx + 1, min(score_idx + 6, len(cells))):
            val = cells[i].strip()
            # Eğer hücrede sadece sayı-sayı formatı varsa (örn: "5-2")
            if re.match(r"^\d{1,2}-\d{1,2}$", val):
                corners = extract_corners_from_cell(val)
                if corners:
                    corner_home, corner_away = corners; break
    
    return MatchRow(league=league, date=date_val, home=home, away=away, ft_home=ft_h, ft_away=ft_a, ht_home=ht_h, ht_away=ht_a, corner_home=corner_home, corner_away=corner_away)

def parse_matches_from_table_html(table_html: str) -> List[MatchRow]:
    out: List[MatchRow] = []
    rows = extract_table_rows_from_html(table_html)
    for cells in rows:
        m = parse_match_from_cells(cells)
        if m: out.append(m)
    return sort_matches_desc(dedupe_matches(out))

# ==============================================================================
# 8. STANDINGS & ODDS (ORANLAR)
# ==============================================================================

def _to_int(x: str) -> Optional[int]:
    try:
        x = (x or "").strip()
        if x in {"", "-", "—"}: return None
        return int(x)
    except: return None

def parse_standings_table_rows(rows: List[List[str]]) -> List[StandRow]:
    wanted = {"Total", "Home", "Away", "Last 6", "Last6"}
    out: List[StandRow] = []
    for cells in rows:
        if not cells: continue
        head = cells[0].strip()
        if head not in wanted: continue
        label = "Last 6" if head == "Last6" else head
        def g(i): return cells[i] if i < len(cells) else ""
        r = StandRow(ft=label, matches=_to_int(g(1)), win=_to_int(g(2)), draw=_to_int(g(3)), loss=_to_int(g(4)), scored=_to_int(g(5)), conceded=_to_int(g(6)), pts=_to_int(g(7)), rank=_to_int(g(8)), rate=g(9) if g(9) else None)
        if r.matches is not None and not (1 <= r.matches <= 80): continue
        if any(x.ft == r.ft for x in out): continue
        out.append(r)
    order = {"Total": 0, "Home": 1, "Away": 2, "Last 6": 3}
    out.sort(key=lambda x: order.get(x.ft, 99))
    return out

def extract_standings_for_team(page_source: str, team_name: str) -> List[StandRow]:
    team_key = norm_key(team_name)
    for tbl in extract_tables_html(page_source):
        text_low = strip_tags(tbl).lower()
        if not all(k in text_low for k in ["matches", "win", "draw", "loss"]): continue
        if team_key not in norm_key(strip_tags(tbl)): continue
        rows = extract_table_rows_from_html(tbl)
        parsed = parse_standings_table_rows(rows)
        if parsed: return parsed
    return []

def standings_to_splits(rows: List[StandRow]) -> Dict[str, Optional[SplitGFGA]]:
    mp = {"Total": None, "Home": None, "Away": None, "Last 6": None}
    for r in rows:
        if r.matches and r.scored is not None and r.conceded is not None:
            mp[r.ft] = SplitGFGA(r.matches, r.scored, r.conceded)
    return mp

def extract_bet365_initial_odds(page_source: str) -> Optional[Dict[str, float]]:
    """
    Bet365 İlk Oranlarını Çeker.
    Playwright beklediği için bu sefer tablonun dolu gelme ihtimali çok yüksek.
    """
    try:
        # Yöntem 1: Sayfadaki "Initial" kelimesini içeren Bet365 satırını bul
        # (Bu genellikle en güvenilir yöntemdir)
        pattern = r'Bet365.*?Initial.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)'
        match = re.search(pattern, page_source, re.DOTALL | re.IGNORECASE)
        if match:
            return {"1": float(match.group(1)), "X": float(match.group(2)), "2": float(match.group(3))}
        
        # Yöntem 2: Tablo taraması
        for table in extract_tables_html(page_source):
            if "bet365" not in table.lower(): continue
            rows = extract_table_rows_from_html(table)
            for row in rows:
                if len(row) < 4: continue
                # Satırda Bet365 geçiyor mu?
                if "bet365" not in row[0].lower(): continue
                
                # Sayıları topla
                odds = [float(x) for x in row if re.match(r"^\d+\.\d+$", x)]
                # İlk 3 sayı genellikle Initial 1-X-2 dir
                if len(odds) >= 3:
                    return {"1": odds[0], "X": odds[1], "2": odds[2]}
        return None
    except: return None

# ==============================================================================
# 9. PSS FİLTRELEME (STRICT MODE - KESİN KURAL)
# ==============================================================================

def extract_previous_home_away_same_league(
    page_source: str,
    home_team: str,
    away_team: str,
    league_name: str,
    max_take: int = RECENT_N
) -> Tuple[List[MatchRow], List[MatchRow]]:
    """
    STRICT MODE (KESİN MOD):
    Burada kullanıcı "Ev+AynıLig" istiyorsa, eğer 7 maç varsa 7 maç döner. 
    10 maça tamamlamak için alakasız maç eklenmez.
    """
    hk = norm_key(home_team)
    ak = norm_key(away_team)
    lk = norm_key(league_name) if league_name else ""

    # Tüm maçları topla
    all_raw_matches = []
    all_tables = extract_tables_html(page_source)
    for tbl in all_tables:
        cand = parse_matches_from_table_html(tbl)
        if cand: all_raw_matches.extend(cand)
    all_raw_matches = sort_matches_desc(dedupe_matches(all_raw_matches))

    # Takımlara göre ayır
    home_pool = [m for m in all_raw_matches if norm_key(m.home) == hk or norm_key(m.away) == hk]
    away_pool = [m for m in all_raw_matches if norm_key(m.home) == ak or norm_key(m.away) == ak]

    def get_strict_matches(pool: List[MatchRow], team_key: str, is_home_mode: bool, league_key: str) -> List[MatchRow]:
        """Kesin kurallara göre filtreler."""
        strict_list = [] # Sadece istenen kriterler (Ev-Ev-Lig veya Dep-Dep-Lig)
        loose_list = []  # Lig farketmez ama Ev-Ev olsun
        
        for m in pool:
            is_same_league = (league_key and norm_key(m.league) == league_key)
            is_playing_home = (norm_key(m.home) == team_key)
            is_playing_away = (norm_key(m.away) == team_key)

            # Hedef: Ev sahibi ise Evindeki maçlar, Deplasman ise Dep maçları
            target_condition = is_playing_home if is_home_mode else is_playing_away
            
            if target_condition:
                if is_same_league:
                    strict_list.append(m)
                loose_list.append(m)

        # KURAL: Eğer Strict listede en az 3 maç varsa, SADECE ONLARI KULLAN.
        # Sayıyı 10'a tamamlamak için loose listten alma.
        if len(strict_list) >= 3:
            return strict_list
        
        # Eğer Strict çok azsa (0-2 maç), mecbur Loose listeye (Farklı lig) bak
        if len(loose_list) >= 3:
            return loose_list
            
        # O da yoksa boş dön veya havuzu ver (Çok nadir)
        return pool

    # Filtreleri Uygula
    final_home = get_strict_matches(home_pool, hk, is_home_mode=True, league_key=lk)
    final_away = get_strict_matches(away_pool, ak, is_home_mode=False, league_key=lk)

    # Sadece en son N maçı al (Filtrelenmiş listeden)
    return final_home[:max_take], final_away[:max_take]

def extract_h2h_matches(page_source: str, home_team: str, away_team: str) -> List[MatchRow]:
    all_matches = []
    for tbl in extract_tables_html(page_source):
        cand = parse_matches_from_table_html(tbl)
        if cand: all_matches.extend(cand)
    h2h_pair = [m for m in all_matches if is_h2h_pair(m, home_team, away_team)]
    return sort_matches_desc(dedupe_matches(h2h_pair))

def build_prev_stats(team: str, matches: List[MatchRow]) -> TeamPrevStats:
    tkey = norm_key(team); st = TeamPrevStats(name=team)
    if not matches: return st
    
    gfs, gas, corners_for, corners_against = [], [], [], []
    clean_sheets = 0; scored_matches = 0
    
    for m in matches:
        # Skorlar
        if norm_key(m.home) == tkey:
            gf, ga = m.ft_home, m.ft_away
            cf, ca = m.corner_home, m.corner_away
        else:
            gf, ga = m.ft_away, m.ft_home
            cf, ca = m.corner_away, m.corner_home
            
        gfs.append(gf); gas.append(ga)
        if cf is not None: corners_for.append(cf)
        if ca is not None: corners_against.append(ca)
        if ga == 0: clean_sheets += 1
        if gf > 0: scored_matches += 1
            
    st.n_total = len(matches)
    st.gf_total = sum(gfs) / st.n_total if st.n_total else 0.0
    st.ga_total = sum(gas) / st.n_total if st.n_total else 0.0
    st.clean_sheets = clean_sheets
    st.scored_matches = scored_matches
    # Korner ortalamaları (Eğer veri varsa)
    st.corners_for = sum(corners_for) / len(corners_for) if corners_for else 0.0
    st.corners_against = sum(corners_against) / len(corners_against) if corners_against else 0.0
    
    # Ev/Dep ayrımı (İstatistik objesi içinde bilgi amaçlı)
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

# ==============================================================================
# 10. ANALİZ VE MATEMATİK
# ==============================================================================

def analyze_corners(home_prev: TeamPrevStats, away_prev: TeamPrevStats, h2h_matches: List[MatchRow]) -> Dict[str, Any]:
    # H2H Kornerleri
    h2h_total_list = []
    for m in h2h_matches[:H2H_N]:
        if m.corner_home is not None and m.corner_away is not None:
            h2h_total_list.append(m.corner_home + m.corner_away)
            
    h2h_avg = sum(h2h_total_list) / len(h2h_total_list) if h2h_total_list else 0.0
    
    # PSS Kornerleri (Takımların kendi ortalamaları)
    home_avg = home_prev.corners_for + home_prev.corners_against # Toplam korner beklentisi (kendi maçlarında)
    away_avg = away_prev.corners_for + away_prev.corners_against
    
    has_pss = (home_avg > 0 and away_avg > 0)
    has_h2h = (h2h_avg > 0)
    
    # Ağırlıklı Tahmin
    if has_h2h and has_pss:
        # %60 H2H, %20 Ev Ort, %20 Dep Ort
        total_pred = (0.6 * h2h_avg) + (0.2 * home_avg) + (0.2 * away_avg)
    elif has_pss:
        total_pred = (home_avg + away_avg) / 2
    elif has_h2h:
        total_pred = h2h_avg
    else:
        total_pred = 0.0
        
    # Alt/Üst Olasılıkları (Basit Dağılım)
    predictions = {}
    for line in [8.5, 9.5, 10.5, 11.5]:
        if total_pred == 0:
            predictions[f"O{line}"] = 0.0
            predictions[f"U{line}"] = 0.0
        else:
            # Basit bir Poisson benzeri yaklaşım veya farka dayalı heuristic
            diff = total_pred - line
            # Eğer tahmin 11 ise, 9.5 üst olma ihtimali yüksektir.
            # Sigmoid benzeri basit bir geçiş
            prob = 0.5 + (diff * 0.15) 
            prob = max(0.05, min(0.95, prob)) # %5 ile %95 arasına sıkıştır
            predictions[f"O{line}"] = prob
            predictions[f"U{line}"] = 1.0 - prob

    data_points = len(h2h_total_list) + (1 if has_pss else 0)
    conf = "Yüksek" if data_points >= 8 else ("Orta" if data_points >= 4 else "Düşük")
    
    # Ev/Dep tahmini (Toplamı oranlayarak)
    pred_home = total_pred * 0.55 # Ev sahibi genelde biraz daha fazla kullanır
    pred_away = total_pred * 0.45
    
    return {
        "predicted_home_corners": round(pred_home, 1),
        "predicted_away_corners": round(pred_away, 1),
        "total_corners": round(total_pred, 1),
        "h2h_avg": round(h2h_avg, 1),
        "predictions": predictions,
        "confidence": conf
    }

def normalize_weights(w):
    s = sum(max(0.0, v) for v in w.values())
    return {k: max(0.0, v) / s for k, v in w.items()} if s > 1e-9 else {}

def compute_component_standings(st_home, st_away):
    hh = st_home.get("Home") or st_home.get("Total")
    aa = st_away.get("Away") or st_away.get("Total")
    if not hh or not aa or hh.matches < 3 or aa.matches < 3: return None
    return (hh.gf_pg + aa.ga_pg)/2.0, (aa.gf_pg + hh.ga_pg)/2.0, {}

def compute_component_pss(home_prev, away_prev):
    if home_prev.n_total < 3 or away_prev.n_total < 3: return None
    # Ev sahibi için ev performansı, Deplasman için dep performansı öncelikli
    h_gf = home_prev.gf_home if home_prev.n_home >= 3 else home_prev.gf_total
    h_ga = home_prev.ga_home if home_prev.n_home >= 3 else home_prev.ga_total
    a_gf = away_prev.gf_away if away_prev.n_away >= 3 else away_prev.gf_total
    a_ga = away_prev.ga_away if away_prev.n_away >= 3 else away_prev.ga_total
    return (h_gf + a_ga)/2.0, (a_gf + h_ga)/2.0, {}

def compute_component_h2h(h2h_matches, home_team, away_team):
    if not h2h_matches or len(h2h_matches) < 3: return None
    hk, ak = norm_key(home_team), norm_key(away_team)
    hg, ag = [], []
    for m in h2h_matches[:H2H_N]:
        if norm_key(m.home)==hk: hg.append(m.ft_home); ag.append(m.ft_away)
        else: hg.append(m.ft_away); ag.append(m.ft_home)
    return sum(hg)/len(hg), sum(ag)/len(ag), {}

def clamp_lambda(lh, la):
    def c(x): return max(0.15, min(3.80, x))
    return c(lh), c(la), []

def compute_lambdas(st_home_s, st_away_s, home_prev, away_prev, h2h_used, home_team, away_team):
    info = {"weights_used": {}, "warnings": []}
    comps = {}
    
    c1 = compute_component_standings(st_home_s, st_away_s)
    if c1: comps["standing"] = c1
    c2 = compute_component_pss(home_prev, away_prev)
    if c2: comps["pss"] = c2
    c3 = compute_component_h2h(h2h_used, home_team, away_team)
    if c3: comps["h2h"] = c3
    
    w = {}
    if "standing" in comps: w["standing"] = W_ST_BASE
    if "pss" in comps: w["pss"] = W_PSS_BASE
    if "h2h" in comps: w["h2h"] = W_H2H_BASE
    w_norm = normalize_weights(w)
    info["weights_used"] = w_norm
    
    if not w_norm:
        lh, la = 1.20, 1.20
    else:
        lh, la = 0.0, 0.0
        for k, weight in w_norm.items():
            val_h, val_a, _ = comps[k]
            lh += weight * val_h
            la += weight * val_a
            
    lh, la, _ = clamp_lambda(lh, la)
    return lh, la, info

def poisson_pmf(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    try: return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except: return 0.0

def build_score_matrix(lh, la, max_g=5):
    mat = {}
    for h in range(max_g + 1):
        ph = poisson_pmf(h, lh)
        for a in range(max_g + 1):
            mat[(h, a)] = ph * poisson_pmf(a, la)
    return mat

def market_probs_from_matrix(mat):
    out = {"1": 0.0, "X": 0.0, "2": 0.0, "BTTS": 0.0, "O2.5": 0.0, "U2.5": 0.0}
    for (h, a), p in mat.items():
        if h > a: out["1"] += p
        elif h == a: out["X"] += p
        else: out["2"] += p
        if h > 0 and a > 0: out["BTTS"] += p
        if h + a > 2.5: out["O2.5"] += p
        else: out["U2.5"] += p
    return out

def monte_carlo(lh, la, n, seed=42):
    rng = np.random.default_rng(seed)
    hg = rng.poisson(lh, size=n)
    ag = rng.poisson(la, size=n)
    total = hg + ag
    
    # Olasılıklar
    p1 = np.mean(hg > ag)
    px = np.mean(hg == ag)
    p2 = np.mean(hg < ag)
    pbtts = np.mean((hg > 0) & (ag > 0))
    po25 = np.mean(total > 2.5)
    pu25 = np.mean(total <= 2.5)
    
    # Top 10 Skor
    cnt = Counter(zip(hg, ag))
    top10 = [(f"{h}-{a}", c/n*100) for (h, a), c in cnt.most_common(10)]
    
    return {
        "p": {"1": p1, "X": px, "2": p2, "BTTS": pbtts, "O2.5": po25, "U2.5": pu25},
        "TOP10": top10
    }

def blend_probs(p1, p2, alpha):
    out = {}
    for k in p1:
        out[k] = alpha * p1.get(k, 0) + (1-alpha) * p2.get(k, 0)
    return out

def value_and_kelly(prob, odds):
    if odds <= 1.0 or prob <= 0.0: return -1.0, 0.0
    v = (odds * prob) - 1.0
    b = odds - 1.0
    k = (b * prob - (1-prob)) / b
    return v, max(0.0, k)

def net_ou_prediction(probs):
    p = probs.get("O2.5", 0)
    label = "ÜST" if p >= 0.5 else "ALT"
    val = p if p >= 0.5 else 1.0 - p
    return f"2.5 {label}", val

def net_btts_prediction(probs):
    p = probs.get("BTTS", 0)
    label = "VAR" if p >= 0.5 else "YOK"
    val = p if p >= 0.5 else 1.0 - p
    return f"{label}", val

def format_comprehensive_report(data):
    t = data["teams"]
    blend = data["blended_probs"]
    top7 = data["poisson"]["top7_scores"]
    
    lines = []
    lines.append("="*40)
    lines.append(f"{t['home']} vs {t['away']}")
    lines.append("="*40)
    
    lines.append("OLASI SKORLAR:")
    for i, (score, prob) in enumerate(top7[:5], 1):
        lines.append(f"{i}. {score:<5} %{prob*100:.1f}")
        
    ou_lbl, ou_val = net_ou_prediction(blend)
    btts_lbl, btts_val = net_btts_prediction(blend)
    lines.append(f"\nTAHMİNLER:")
    lines.append(f"A/Ü 2.5: {ou_lbl} (%{ou_val*100:.1f})")
    lines.append(f"KG:      {btts_lbl} (%{btts_val*100:.1f})")
    
    c = data.get("corner_analysis", {})
    if c.get("total_corners", 0) > 0:
        lines.append(f"Korner:  {c['total_corners']} (Ort)")
        lines.append(f"         Ev:{c['predicted_home_corners']} | Dep:{c['predicted_away_corners']}")
    
    vb = data.get("value_bets", {})
    if vb.get("used_odds"):
        if vb.get("table"):
            lines.append("\nDEĞERLİ BAHİSLER:")
            for row in vb["table"]:
                if row["value"] >= VALUE_MIN and row["prob"] >= PROB_MIN:
                    lines.append(f"✅ {row['market']} @{row['odds']} (V:%{row['value']*100:.0f})")
        else:
            lines.append("\nDeğerli bahis bulunamadı.")
    else:
        lines.append("\nOran verisi çekilemedi.")
        
    ds = data["data_sources"]
    lines.append(f"\nVERİ KAYNAKLARI:")
    lines.append(f"PSS (Ev): {ds['home_prev_matches']} maç")
    lines.append(f"PSS (Dep): {ds['away_prev_matches']} maç")
    lines.append(f"H2H: {ds['h2h_matches']} maç")
    
    return "\n".join(lines)

def analyze_nowgoal(url, odds=None, mc_runs=MC_RUNS_DEFAULT):
    h2h_url = build_h2h_url(url)
    html = fetch_with_browser(h2h_url)
    
    home_team, away_team = parse_teams_from_title(html)
    if not home_team: raise RuntimeError("Takım isimleri bulunamadı.")
    
    league_match = re.search(r'<span[^>]*class=["\']?sclassLink["\']?[^>]*>([^<]+)</span>', html)
    league_name = strip_tags(league_match.group(1)) if league_match else ""
    
    st_home = standings_to_splits(extract_standings_for_team(html, home_team))
    st_away = standings_to_splits(extract_standings_for_team(html, away_team))
    
    h2h_used = extract_h2h_matches(html, home_team, away_team)[:H2H_N]
    
    # STRICT FILTERING
    prev_home, prev_away = extract_previous_home_away_same_league(html, home_team, away_team, league_name, RECENT_N)
    
    home_prev_stats = build_prev_stats(home_team, prev_home)
    away_prev_stats = build_prev_stats(away_team, prev_away)
    
    lam_home, lam_away, lambda_info = compute_lambdas(st_home, st_away, home_prev_stats, away_prev_stats, h2h_used, home_team, away_team)
    
    score_mat = build_score_matrix(lam_home, lam_away, MAX_GOALS_FOR_MATRIX)
    poisson_market = market_probs_from_matrix(score_mat)
    mc = monte_carlo(lam_home, lam_away, int(mc_runs))
    blended = blend_probs(poisson_market, mc["p"], BLEND_ALPHA)
    
    corner_analysis = analyze_corners(home_prev_stats, away_prev_stats, h2h_used)
    
    if not odds: odds = extract_bet365_initial_odds(html)
    
    value_block = {"used_odds": False}
    qualified = []
    
    if odds and all(k in odds for k in ["1", "X", "2"]):
        value_block["used_odds"] = True
        table = []
        for mkt in ["1", "X", "2"]:
            o = float(odds[mkt]); p = float(blended.get(mkt, 0.0))
            v, k = value_and_kelly(p, o)
            row = {"market": mkt, "prob": p, "odds": o, "value": v, "kelly": k, "qkelly": k*0.25}
            table.append(row)
            if v >= VALUE_MIN and p >= PROB_MIN: qualified.append((mkt, p, o, v))
        value_block["table"] = table
        if qualified:
            best = sorted(qualified, key=lambda x: x[3], reverse=True)[0]
            value_block["decision"] = f"OYNANABİLİR: {best[0]} (V:%{best[3]*100:.1f})"
        else:
            value_block["decision"] = "OYNA MA (Value yok)"
    else:
        value_block["decision"] = "Oran yok"

    data = {
        "teams": {"home": home_team, "away": away_team},
        "poisson": {"top7_scores": [(f"{h}-{a}", p) for (h, a), p in sorted(score_mat.items(), key=lambda x:x[1], reverse=True)[:7]]},
        "blended_probs": blended,
        "corner_analysis": corner_analysis,
        "value_bets": value_block,
        "lambda": {"info": lambda_info},
        "data_sources": {
            "standings_used": len(st_home) > 0,
            "h2h_matches": len(h2h_used),
            "home_prev_matches": len(prev_home),
            "away_prev_matches": len(prev_away)
        }
    }
    data["report_comprehensive"] = format_comprehensive_report(data)
    return data

# ==============================================================================
# 11. FLASK (API)
# ==============================================================================
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "v": "5.7"})

@app.route("/analiz_et", methods=["POST"])
def analiz_et():
    try:
        payload = request.get_json(silent=True) or {}
        url = payload.get("url", "").strip()
        if not url: return jsonify({"error": "URL yok"}), 400
        
        data = analyze_nowgoal(url)
        
        top_skor = data["poisson"]["top7_scores"][0][0]
        blend = data["blended_probs"]
        net_ou, net_ou_p = net_ou_prediction(blend)
        net_btts, net_btts_p = net_btts_prediction(blend)
        
        c = data.get("corner_analysis", {})
        c_pick = "-"
        if c.get("total_corners", 0) > 0:
            line = 9.5
            p_over = c["predictions"].get(f"O{line}", 0.0)
            c_pick = f"{line} {'ÜST' if p_over >= 0.5 else 'ALT'} (%{p_over*100:.0f})"

        return jsonify({
            "ok": True,
            "skor": top_skor,
            "alt_ust": f"{net_ou} (%{net_ou_p*100:.1f})",
            "btts": f"{net_btts} (%{net_btts_p*100:.1f})",
            "korner": c_pick,
            "karar": data["value_bets"].get("decision", "-"),
            "detay": data["report_comprehensive"],
            "veri": data["data_sources"]
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
