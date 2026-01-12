# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - v5.6 (FULL GENİŞLETİLMİŞ VERSİYON)
-----------------------------------------------------------
Bu kod, kullanıcının isteği üzerine hiçbir satır birleştirilmeden,
tamamen açık ve detaylı şekilde yazılmıştır.

İÇERİK:
1. Playwright Entegrasyonu (JavaScript verilerini okumak için)
2. Akıllı PSS Analizi (Ev Sahibi için Ev, Deplasman için Deplasman maçı önceliği)
3. /health Rotası (Render.com deploy hatasını çözmek için)
4. Detaylı Loglama ve Hata Ayıklama
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
# Puan Durumu etkisi
W_ST_BASE = 0.45
# Geçmiş Skorlar (Form) etkisi
W_PSS_BASE = 0.30
# Aralarındaki Maçlar (H2H) etkisi
W_H2H_BASE = 0.25

# İstatistik Birleştirme Oranı (%50 Poisson, %50 Monte Carlo)
BLEND_ALPHA = 0.50

# --- Bahis Filtreleri ---
VALUE_MIN = 0.05       # En az %5 Value
PROB_MIN = 0.55        # En az %55 Olasılık
KELLY_MIN = 0.02       # En az %2 Kelly Oranı
MAX_GOALS_FOR_MATRIX = 5

# Debug (Hata Ayıklama) Modu
DEBUG = os.getenv("NOWGOAL_DEBUG", "0").strip() == "1"

# ==============================================================================
# 2. REGEX (DÜZENLİ İFADELER)
# ==============================================================================

# Tarih formatlarını yakalamak için (dd-mm-yyyy veya yyyy-mm-dd)
DATE_ANY_RE = re.compile(r"\b(\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{2}-\d{2})\b")

# Skorları yakalamak için (Örn: "2-1", "0-0", "(1-0)" gibi)
SCORE_RE = re.compile(
    r"\b(\d{1,2})\s*-\s*(\d{1,2})(?!-\d{2,4})(?:\s*\((\d{1,2})\s*-\s*(\d{1,2})\))?\b"
)

# ==============================================================================
# 3. YARDIMCI FONKSİYONLAR
# ==============================================================================

def dprint(*args):
    """Debug modu açıksa konsola bilgi basar."""
    if DEBUG:
        print(*args)

def norm_key(s: str) -> str:
    """Metinleri karşılaştırmak için temizler (küçük harf, sadece harf/rakam)."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def normalize_date(d: str) -> Optional[str]:
    """Tarih metnini standart dd-mm-yyyy formatına çevirir."""
    if not d:
        return None
    
    d = d.strip()
    m = DATE_ANY_RE.search(d)
    
    if not m:
        return None
    
    val = m.group(1)
    
    # yyyy-mm-dd formatı kontrolü
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
        yyyy, mm, dd = val.split("-")
        return f"{int(dd):02d}-{int(mm):02d}-{yyyy}"
    
    # dd-mm-yyyy formatı kontrolü
    if re.match(r"^\d{1,2}-\d{1,2}-\d{4}$", val):
        dd, mm, yyyy = val.split("-")
        return f"{int(dd):02d}-{int(mm):02d}-{yyyy}"
    
    return None

def parse_date_key(date_str: str) -> Tuple[int, int, int]:
    """Tarihi sıralama yapılabilir bir demet (tuple) haline getirir."""
    if not date_str or not re.match(r"^\d{2}-\d{2}-\d{4}$", date_str):
        return (0, 0, 0)
    
    dd, mm, yyyy = date_str.split("-")
    return (int(yyyy), int(mm), int(dd))

# ==============================================================================
# 4. VERİ SINIFLARI (DATA CLASSES)
# ==============================================================================

@dataclass
class MatchRow:
    """Bir maçın temel verilerini tutan sınıf."""
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
    """Gol Atılan / Gol Yenilen istatistiklerini tutan sınıf."""
    matches: int
    gf: int
    ga: int

    @property
    def gf_pg(self) -> float:
        """Maç başına atılan gol ortalaması"""
        if self.matches:
            return self.gf / self.matches
        return 0.0

    @property
    def ga_pg(self) -> float:
        """Maç başına yenilen gol ortalaması"""
        if self.matches:
            return self.ga / self.matches
        return 0.0

@dataclass
class StandRow:
    """Puan tablosundaki bir satırı temsil eden sınıf."""
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
    """Takımın geçmiş performans özetini tutan sınıf."""
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
# 5. HTML PARSE (AYRIŞTIRMA) YARDIMCILARI
# ==============================================================================

def strip_tags(s: str) -> str:
    """HTML etiketlerini temizler ve saf metni döndürür."""
    # Scriptleri temizle
    s = re.sub(r"<script\b.*?</script>", " ", s, flags=re.IGNORECASE | re.DOTALL)
    # Stilleri temizle
    s = re.sub(r"<style\b.*?</style>", " ", s, flags=re.IGNORECASE | re.DOTALL)
    # Diğer HTML taglerini temizle
    s = re.sub(r"<[^>]+>", " ", s)
    # HTML özel karakterlerini düzelt
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    # Fazla boşlukları tek boşluğa indir
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_tables_html(page_source: str) -> List[str]:
    """Sayfadaki tüm <table>...</table> bloklarını bulur."""
    return [m.group(0) for m in re.finditer(
        r"<table\b[^>]*>.*?</table>", page_source or "", flags=re.IGNORECASE | re.DOTALL
    )]

def extract_table_rows_from_html(table_html: str) -> List[List[str]]:
    """Bir HTML tablosunu satır satır okur ve hücreleri listeye çevirir."""
    rows: List[List[str]] = []
    trs = re.findall(r"<tr\b[^>]*>.*?</tr>", table_html or "", flags=re.IGNORECASE | re.DOTALL)
    
    for tr in trs:
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, flags=re.IGNORECASE | re.DOTALL)
        if not cells:
            continue
        
        cleaned = [strip_tags(c) for c in cells]
        # Boş veya tire olan hücreleri temizle
        cleaned = [c for c in cleaned if c and c not in {"—", "-"}]
        
        if cleaned:
            rows.append(cleaned)
            
    return rows

def section_tables_by_marker(page_source: str, marker: str, max_tables: int = 3) -> List[str]:
    """Belirli bir başlığın (marker) altındaki tabloları bulur."""
    low = (page_source or "").lower()
    pos = low.find(marker.lower())
    
    if pos == -1:
        return []
    
    sub = page_source[pos:]
    tabs = extract_tables_html(sub)
    return tabs[:max_tables]

# ==============================================================================
# 6. VERİ ÇEKME (PLAYWRIGHT AGRESİF MOD)
# ==============================================================================

def fetch_with_browser(url: str) -> str:
    """
    Playwright kullanarak sayfayı açar.
    AGRESİF BEKLEME MODU: Sayfayı yavaşça aşağı kaydırır, belirli noktalarda
    zorunlu beklemeler yapar. Bu, PSS, Oranlar ve Korner verilerinin yüklenmesini
    garanti altına almak içindir.
    """
    print(f"[BROWSER] Sayfa açılıyor (Agresif Mod): {url}")
    try:
        with sync_playwright() as p:
            # Chromium başlat
            browser = p.chromium.launch(headless=True)
            
            # Context oluştur (Ekranı geniş tutuyoruz)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1366, "height": 768}
            )
            page = context.new_page()

            # 1. Sayfaya git ve ilk yükleme için uzun bekle (90sn timeout)
            page.goto(url, timeout=90000, wait_until="domcontentloaded")
            print("[BROWSER] İlk yükleme bekleniyor...")
            page.wait_for_timeout(4000)

            # 2. Sayfanın ortasına kaydır ve bekle
            print("[BROWSER] Ortaya kaydırılıyor...")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            page.wait_for_timeout(3000)

            # 3. Sayfanın en altına kaydır ve bekle (En önemli kısım)
            print("[BROWSER] En alta kaydırılıyor...")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(4000)

            # İçeriği al
            content = page.content()
            
            browser.close()
            print(f"[BROWSER] HTML başarıyla alındı. Boyut: {len(content)} karakter")
            return content

    except Exception as e:
        print(f"[BROWSER ERROR] Hata oluştu: {str(e)}")
        # Hatayı yukarı fırlatıyoruz ki sistem fark etsin
        raise RuntimeError(f"Tarayıcı hatası: {e}")

def extract_match_id(url: str) -> str:
    """URL içinden maç ID'sini çıkarır."""
    m = re.search(r"(?:h2h-|/match/h2h-)(\d+)", url)
    if m:
        return m.group(1)
    
    nums = re.findall(r"\d{6,}", url)
    if not nums:
        raise ValueError("Match ID çıkaramadım")
    
    return nums[-1]

def extract_base_domain(url: str) -> str:
    """URL'nin ana domainini bulur."""
    m = re.match(r"^(https?://[^/]+)", url.strip())
    if m:
        return m.group(1)
    return "https://live3.nowgoal26.com"

def build_h2h_url(url: str) -> str:
    """Verilen maç linkini H2H analiz linkine çevirir."""
    match_id = extract_match_id(url)
    base = extract_base_domain(url)
    return f"{base}/match/h2h-{match_id}"

def parse_teams_from_title(html: str) -> Tuple[str, str]:
    """Sayfa başlığından takım isimlerini bulur."""
    m = re.search(r"<title>\s*(.*?)\s*</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = strip_tags(m.group(1)) if m else ""
    
    # Başlıktaki VS veya vs ayracına göre böl
    mm = re.search(r"(.+?)\s+VS\s+(.+?)(?:\s+-|\s+\||$)", title, flags=re.IGNORECASE)
    if not mm:
        mm = re.search(r"(.+?)\s+vs\s+(.+?)(?:\s+-|\s+\||$)", title, flags=re.IGNORECASE)
    
    if not mm:
        return "", ""
    
    return mm.group(1).strip(), mm.group(2).strip()

# ==============================================================================
# 7. MAÇ VERİSİ AYRIŞTIRMA (MATCH PARSING)
# ==============================================================================

def sort_matches_desc(matches: List[MatchRow]) -> List[MatchRow]:
    """Maçları tarihe göre yeniden eskiye sıralar."""
    return sorted(matches, key=lambda x: parse_date_key(x.date), reverse=True)

def dedupe_matches(matches: List[MatchRow]) -> List[MatchRow]:
    """Aynı maçın birden fazla kez listelenmesini engeller."""
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
    """Bir maçın, bizim analiz ettiğimiz iki takım arasında olup olmadığını kontrol eder."""
    hk = norm_key(home_team)
    ak = norm_key(away_team)
    mh = norm_key(m.home)
    ma = norm_key(m.away)
    
    return (mh == hk and ma == ak) or (mh == ak and ma == hk)

def extract_corners_from_cell(cell: str) -> Optional[Tuple[int, int]]:
    """Hücre içinden korner sayısını çıkarmaya çalışır."""
    if not cell:
        return None
    
    # 1. Yöntem: Parantez içi (8-3)
    m = re.search(r"\((\d{1,2})-(\d{1,2})\)", cell)
    if m:
        return int(m.group(1)), int(m.group(2))
    
    # 2. Yöntem: Düz metin 8-3
    all_pairs = re.findall(r"(\d{1,2})-(\d{1,2})", cell)
    if len(all_pairs) >= 2:
        # Genelde ikinci çift kornerdir
        return int(all_pairs[1][0]), int(all_pairs[1][1])
    
    return None

def parse_match_from_cells(cells: List[str]) -> Optional[MatchRow]:
    """Tablodaki bir satırın hücrelerini alıp MatchRow objesine çevirir."""
    if not cells or len(cells) < 4:
        return None

    # Tarih hücresini bul
    date_idx = None
    date_val = None
    for i, c in enumerate(cells):
        d = normalize_date(c)
        if d:
            date_idx = i
            date_val = d
            break
            
    if not date_val:
        return None

    # Skor hücresini bul
    score_idx = None
    score_m = None
    for i, c in enumerate(cells):
        if i == date_idx:
            continue
        m = SCORE_RE.search((c or "").strip())
        if m:
            score_idx = i
            score_m = m
            break
            
    if not score_m:
        return None

    ft_h = int(score_m.group(1))
    ft_a = int(score_m.group(2))
    ht_h = int(score_m.group(3)) if score_m.group(3) else None
    ht_a = int(score_m.group(4)) if score_m.group(4) else None

    home = None
    away = None

    # Ev sahibi (Skorun solundaki ilk anlamlı metin)
    for i in range(score_idx - 1, -1, -1):
        if cells[i] and cells[i] != date_val:
            home = cells[i].strip()
            break
            
    # Deplasman (Skorun sağındaki ilk anlamlı metin)
    for i in range(score_idx + 1, len(cells)):
        if cells[i]:
            away = cells[i].strip()
            break

    if not home or not away:
        return None

    league = cells[0].strip() if cells[0] and cells[0] != date_val else "—"

    # Korner verisi
    corner_home, corner_away = None, None
    
    # Önce skor hücresine bak
    corners = extract_corners_from_cell(cells[score_idx])
    if corners:
        corner_home, corner_away = corners
        
    # Yoksa sonraki hücrelere bak
    if not corners:
        for i in range(score_idx + 1, min(score_idx + 6, len(cells))):
            corners = extract_corners_from_cell(cells[i])
            if corners:
                corner_home, corner_away = corners
                break

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
    )

def parse_matches_from_table_html(table_html: str) -> List[MatchRow]:
    """Bir HTML tablosunu analiz edip maç listesi döndürür."""
    out: List[MatchRow] = []
    rows = extract_table_rows_from_html(table_html)
    
    for cells in rows:
        m = parse_match_from_cells(cells)
        if m:
            out.append(m)
            
    return sort_matches_desc(dedupe_matches(out))

# ==============================================================================
# 8. PUAN DURUMU (STANDINGS) VE ORANLAR
# ==============================================================================

def _to_int(x: str) -> Optional[int]:
    """String veriyi güvenli şekilde Integer'a çevirir."""
    try:
        x = (x or "").strip()
        if x in {"", "-", "—"}:
            return None
        return int(x)
    except Exception:
        return None

def parse_standings_table_rows(rows: List[List[str]]) -> List[StandRow]:
    """Puan durumu tablosunu işler."""
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
    """Verilen takım için puan durumu tablosunu arayıp bulur."""
    team_key = norm_key(team_name)
    
    for tbl in extract_tables_html(page_source):
        text_low = strip_tags(tbl).lower()
        
        # Puan tablosu anahtar kelimeleri
        if not all(k in text_low for k in ["matches", "win", "draw", "loss"]):
            continue
            
        if team_key not in norm_key(strip_tags(tbl)):
            continue
            
        rows = extract_table_rows_from_html(tbl)
        parsed = parse_standings_table_rows(rows)
        if parsed:
            return parsed
            
    return []

def standings_to_splits(rows: List[StandRow]) -> Dict[str, Optional[SplitGFGA]]:
    """Puan tablosunu Gol Atma/Yeme istatistiklerine dönüştürür."""
    mp = {"Total": None, "Home": None, "Away": None, "Last 6": None}
    for r in rows:
        if r.matches and r.scored is not None and r.conceded is not None:
            mp[r.ft] = SplitGFGA(r.matches, r.scored, r.conceded)
    return mp

def extract_bet365_initial_odds(page_source: str) -> Optional[Dict[str, float]]:
    """Bet365 açılış oranlarını bulmaya çalışır."""
    try:
        # 1. Regex ile ara
        pattern1 = r'Bet365.*?Initial.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)'
        match = re.search(pattern1, page_source, re.DOTALL | re.IGNORECASE)
        if match:
            return {
                "1": float(match.group(1)),
                "X": float(match.group(2)),
                "2": float(match.group(3))
            }
        
        # 2. Tabloları tara
        for table in extract_tables_html(page_source):
            if "bet365" not in table.lower():
                continue
            rows = extract_table_rows_from_html(table)
            for row in rows:
                if len(row) < 4:
                    continue
                if "bet365" not in row[0].lower():
                    continue
                if "initial" in " ".join(row).lower():
                    odds = [float(x) for x in row if re.match(r"^\d+\.\d+$", x)]
                    if len(odds) >= 3:
                        return {"1": odds[0], "X": odds[1], "2": odds[2]}
        return None
    except Exception:
        return None

# ==============================================================================
# 9. KRİTİK BÖLÜM: GURME MODU (PSS FİLTRELEME)
# ==============================================================================

def extract_previous_home_away_same_league(
    page_source: str,
    home_team: str,
    away_team: str,
    league_name: str,
    max_take: int = RECENT_N
) -> Tuple[List[MatchRow], List[MatchRow]]:
    """
    3 AŞAMALI FİLTRELEME MANTIĞI:
    
    1. Aşama (Altın Kural): Ev Sahibi EVİNDE + AYNI LİG & Deplasman DEPLASMANDA + AYNI LİG
    2. Aşama (Gümüş Kural): Ev Sahibi EVİNDE + HERHANGİ LİG & Deplasman DEPLASMANDA + HERHANGİ LİG
    3. Aşama (Bronz Kural): Ev Sahibi HER YERDE + AYNI LİG & Deplasman HER YERDE + AYNI LİG
    """
    hk = norm_key(home_team)
    ak = norm_key(away_team)
    lk = norm_key(league_name) if league_name else ""

    # ÖNCE: Sayfadaki TÜM maçları toplayıp bir havuza atıyoruz.
    all_raw_matches = []
    all_tables = extract_tables_html(page_source)
    
    for tbl in all_tables:
        cand = parse_matches_from_table_html(tbl)
        if cand:
            all_raw_matches.extend(cand)
    
    # Maçları temizle ve sırala
    all_raw_matches = sort_matches_desc(dedupe_matches(all_raw_matches))

    # Takımlara göre havuzları ayır
    home_pool = [m for m in all_raw_matches if norm_key(m.home) == hk or norm_key(m.away) == hk]
    away_pool = [m for m in all_raw_matches if norm_key(m.home) == ak or norm_key(m.away) == ak]

    def filter_matches(pool: List[MatchRow], team_key: str, is_home_mode: bool, league_key: str) -> List[MatchRow]:
        """
        Bu yardımcı fonksiyon, havuzdaki maçları kurallara göre eler.
        is_home_mode=True ise: Takımın ev sahibi olduğu maçları ararız.
        """
        tier1 = [] # Altın: İstediğimiz Yer + Aynı Lig
        tier2 = [] # Gümüş: İstediğimiz Yer + Farklı Lig
        tier3 = [] # Bronz: Herhangi Yer + Aynı Lig
        
        for m in pool:
            is_same_league = (league_key and norm_key(m.league) == league_key)
            
            # Maçta takımımızın rolü (Ev sahibi mi Deplasman mı?)
            is_playing_home = (norm_key(m.home) == team_key)
            is_playing_away = (norm_key(m.away) == team_key)

            # Tier 1: İstediğimiz Yer (Ev/Dep) VE Aynı Lig
            if (is_home_mode and is_playing_home and is_same_league) or (not is_home_mode and is_playing_away and is_same_league):
                tier1.append(m)
            
            # Tier 2: İstediğimiz Yer (Ev/Dep) VE Farklı Lig
            if (is_home_mode and is_playing_home) or (not is_home_mode and is_playing_away):
                tier2.append(m)
            
            # Tier 3: Herhangi Yer VE Aynı Lig
            if is_same_league:
                tier3.append(m)

        # KARAR ANI: Hangi listeyi kullanalım? En iyisi hangisiyse onu ver.
        if len(tier1) >= 3:
            return tier1
        if len(tier2) >= 3:
            return tier2
        if len(tier3) >= 3:
            return tier3
        
        # Hiçbiri yoksa elimizdeki havuzun hepsini ver (Tier 4)
        return pool

    # Filtreleri Uygula
    final_home = filter_matches(home_pool, hk, is_home_mode=True, league_key=lk)
    final_away = filter_matches(away_pool, ak, is_home_mode=False, league_key=lk)

    return final_home[:max_take], final_away[:max_take]

def extract_h2h_matches(page_source: str, home_team: str, away_team: str) -> List[MatchRow]:
    """Aralarındaki maçları (H2H) bulur."""
    all_matches = []
    for tbl in extract_tables_html(page_source):
        cand = parse_matches_from_table_html(tbl)
        if cand:
            all_matches.extend(cand)
            
    h2h_pair = [m for m in all_matches if is_h2h_pair(m, home_team, away_team)]
    return sort_matches_desc(dedupe_matches(h2h_pair))

def build_prev_stats(team: str, matches: List[MatchRow]) -> TeamPrevStats:
    """Maç listesinden istatistiksel özet çıkarır."""
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

# ==============================================================================
# 10. ANALİZ VE MATEMATİKSEL MODELLER
# ==============================================================================

def analyze_corners(home_prev: TeamPrevStats, away_prev: TeamPrevStats, h2h_matches: List[MatchRow]) -> Dict[str, Any]:
    """Korner analizi yapar."""
    h2h_total, h2h_h, h2h_a = [], [], []
    for m in h2h_matches[:H2H_N]:
        if m.corner_home is not None and m.corner_away is not None:
            h2h_total.append(m.corner_home + m.corner_away)
            h2h_h.append(m.corner_home)
            h2h_a.append(m.corner_away)
            
    h2h_total_avg = sum(h2h_total) / len(h2h_total) if h2h_total else 0.0
    h2h_home_avg = sum(h2h_h) / len(h2h_h) if h2h_h else 0.0
    h2h_away_avg = sum(h2h_a) / len(h2h_a) if h2h_a else 0.0
    
    pss_home_for = home_prev.corners_for
    pss_home_against = home_prev.corners_against
    pss_away_for = away_prev.corners_for
    pss_away_against = away_prev.corners_against
    
    has_pss = (pss_home_for > 0 or pss_away_for > 0)
    has_h2h = (h2h_total_avg > 0)

    # Korner Tahmin Formülü
    if has_h2h and has_pss:
        predicted_home = 0.6 * h2h_home_avg + 0.4 * ((pss_home_for + pss_away_against) / 2)
        predicted_away = 0.6 * h2h_away_avg + 0.4 * ((pss_away_for + pss_home_against) / 2)
    elif has_pss:
        predicted_home = (pss_home_for + pss_away_against) / 2
        predicted_away = (pss_away_for + pss_home_against) / 2
    elif has_h2h:
        predicted_home = h2h_home_avg
        predicted_away = h2h_away_avg
    else:
        predicted_home = 0.0
        predicted_away = 0.0

    total = predicted_home + predicted_away
    predictions = {}
    
    for line in [8.5, 9.5, 10.5, 11.5]:
        if total == 0:
            over_prob = 0.0
        else:
            over_prob = 1.0 if total > line else max(0.0, (total - line + 1) / 2)
            over_prob = float(min(1.0, max(0.0, over_prob)))
        predictions[f"O{line}"] = over_prob
        predictions[f"U{line}"] = 1.0 - over_prob
    
    data_points = len(h2h_total) + (1 if has_pss else 0)
    conf = "Yüksek" if data_points >= 8 else ("Orta" if data_points >= 4 else "Düşük")
    
    return {
        "predicted_home_corners": round(predicted_home, 1),
        "predicted_away_corners": round(predicted_away, 1),
        "total_corners": round(total, 1),
        "h2h_avg": round(h2h_total_avg, 1),
        "h2h_data_count": len(h2h_total),
        "pss_data_available": has_pss,
        "predictions": predictions,
        "confidence": conf
    }

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    """Ağırlıkların toplamını 1'e tamamlar."""
    s = sum(max(0.0, v) for v in w.values())
    if s <= 1e-9:
        return {}
    return {k: max(0.0, v) / s for k, v in w.items()}

def compute_component_standings(st_home: Dict[str, Optional[SplitGFGA]], st_away: Dict[str, Optional[SplitGFGA]]) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """Puan durumuna göre gol beklentisi hesaplar."""
    hh = st_home.get("Home") or st_home.get("Total")
    aa = st_away.get("Away") or st_away.get("Total")
    if not hh or not aa or hh.matches < 3 or aa.matches < 3:
        return None
    lam_h = (hh.gf_pg + aa.ga_pg) / 2.0
    lam_a = (aa.gf_pg + hh.ga_pg) / 2.0
    meta = {
        "home_split": {"matches": hh.matches, "gf_pg": hh.gf_pg, "ga_pg": hh.ga_pg},
        "away_split": {"matches": aa.matches, "gf_pg": aa.gf_pg, "ga_pg": aa.ga_pg},
        "formula": "Standing-based lambda"
    }
    return lam_h, lam_a, meta

def compute_component_pss(home_prev: TeamPrevStats, away_prev: TeamPrevStats) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """Geçmiş maçlara göre gol beklentisi hesaplar."""
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
        "home_ga": round(h_ga, 2),
        "away_gf": round(a_gf, 2),
        "away_ga": round(a_ga, 2),
        "formula": "PSS: (home_gf + away_ga) / 2"
    }
    return lam_h, lam_a, meta

def compute_component_h2h(h2h_matches: List[MatchRow], home_team: str, away_team: str) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    """Aralarındaki maçlara göre gol beklentisi hesaplar."""
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
    """Gol beklentilerini makul sınırlar içinde tutar."""
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
    """Tüm bileşenleri birleştirerek nihai gol beklentisini hesaplar."""
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
    if "pss" in comps: w["pss"] = W_PSS_BASE
    if "h2h" in comps: w["h2h"] = W_H2H_BASE
    
    w_norm = normalize_weights(w)
    info["weights_used"] = w_norm
    
    if not w_norm:
        info["warnings"].append("Yetersiz veri → default λ=1.20")
        lh, la = 1.20, 1.20
    else:
        lh = 0.0
        la = 0.0
        for k, wk in w_norm.items():
            ch, ca, meta = comps[k]
            info["components"][k] = {"lam_home": ch, "lam_away": ca, "meta": meta}
            lh += wk * ch
            la += wk * ca
            
    lh, la, clamp_warn = clamp_lambda(lh, la)
    info["warnings"].extend(clamp_warn)
    return lh, la, info

def poisson_pmf(k, lam):
    """Poisson Olasılık Formülü"""
    if lam <= 0: return 1.0 if k == 0 else 0.0
    if k > 170: return 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except:
        return 0.0

def build_score_matrix(lh, la, max_g=5):
    """Skor Olasılık Matrisi Oluşturur"""
    mat = {}
    for h in range(max_g + 1):
        ph = poisson_pmf(h, lh)
        for a in range(max_g + 1):
            mat[(h, a)] = ph * poisson_pmf(a, la)
    return mat

def market_probs_from_matrix(mat):
    """Matristen Bahis Olasılıklarını Çıkarır"""
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

def monte_carlo(lh, la, n, seed=42):
    """Monte Carlo Simülasyonu"""
    rng = np.random.default_rng(seed)
    hg = rng.poisson(lh, size=n)
    ag = rng.poisson(la, size=n)
    total = hg + ag
    
    def p(mask): return float(np.mean(mask))
    
    cnt = Counter(zip(hg.tolist(), ag.tolist()))
    top10 = [(f"{h}-{a}", c / n * 100.0) for (h, a), c in cnt.most_common(10)]
    
    out = {
        "p": {
            "1": p(hg > ag),
            "X": p(hg == ag),
            "2": p(hg < ag),
            "BTTS": p((hg >= 1) & (ag >= 1)),
            "O2.5": p(total >= 3),
            "U2.5": p(total <= 2),
            "O3.5": p(total >= 4),
            "U3.5": p(total <= 3)
        },
        "TOP10": top10
    }
    return out

def model_agreement(p_po, p_mc):
    """İki model arasındaki uyumu ölçer"""
    keys = ["1", "X", "2", "BTTS", "O2.5", "U2.5"]
    diffs = [abs(p_po.get(k, 0) - p_mc.get(k, 0)) for k in keys]
    d = max(diffs) if diffs else 0.0
    if d <= 0.03: return d, "Mükemmel"
    elif d <= 0.06: return d, "İyi"
    elif d <= 0.10: return d, "Orta"
    return d, "Zayıf"

def blend_probs(p1, p2, alpha):
    """Olasılıkları birleştirir"""
    out = {}
    for k in set(list(p1.keys()) + list(p2.keys())):
        if k in p1 and k in p2:
            out[k] = alpha * p1[k] + (1.0 - alpha) * p2[k]
        else:
            out[k] = p1.get(k, p2.get(k, 0.0))
    return out

def value_and_kelly(prob, odds):
    """Value ve Kelly Hesaplar"""
    if odds <= 1.0 or prob <= 0.0 or prob >= 1.0:
        return -1.0, 0.0
    v = odds * prob - 1.0
    b = odds - 1.0
    q = 1.0 - prob
    k = (b * prob - q) / b
    return v, max(0.0, k)

def confidence_label(p):
    if p >= 0.65: return "Yüksek"
    if p >= 0.55: return "Orta"
    return "Düşük"

def net_ou_prediction(probs):
    p_o25 = probs.get("O2.5", 0)
    p_u25 = probs.get("U2.5", 0)
    if p_o25 >= p_u25:
        return "2.5 ÜST", p_o25, confidence_label(p_o25)
    return "2.5 ALT", p_u25, confidence_label(p_u25)

def net_btts_prediction(probs):
    p_btts = probs.get("BTTS", 0)
    p_no = 1.0 - p_btts
    if p_btts >= p_no:
        return "VAR", p_btts, confidence_label(p_btts)
    return "YOK", p_no, confidence_label(p_no)

def final_decision(qualified, diff, diff_label):
    if not qualified:
        return f"OYNAMA (Eşik sağlanmadı, model uyumu: {diff_label})"
    if diff > 0.10:
        return f"TEMKİNLİ (Zayıf model uyumu: {diff_label})"
    best = sorted(qualified, key=lambda x: x[3], reverse=True)[0]
    return f"OYNANABİLİR → {best[0]} (Prob: %{best[1]*100:.1f}, Oran: {best[2]:.2f}, Value: %{best[3]*100:+.1f})"

def format_comprehensive_report(data):
    """Detaylı Analiz Raporu Oluşturur"""
    t = data["teams"]
    blend = data["blended_probs"]
    top7 = data["poisson"]["top7_scores"]
    
    lines = []
    lines.append("=" * 60)
    lines.append(f"  {t['home']} vs {t['away']}")
    lines.append("=" * 60)
    
    lines.append("\nOLASI SKORLAR:")
    for i, (score, prob) in enumerate(top7[:5], 1):
        bar = "█" * int(prob * 50)
        lines.append(f"  {i}. {score:6s} %{prob*100:4.1f}  {bar}")
        
    lines.append("\nNET TAHMİN:")
    lines.append(f"  Ana Skor: {top7[0][0]}")
    lines.append(f"  Alt Skor: {top7[1][0]}, {top7[2][0]}")
    
    net_ou, net_ou_p, _ = net_ou_prediction(blend)
    net_btts, net_btts_p, _ = net_btts_prediction(blend)
    lines.append(f"\nAlt/Üst 2.5: {net_ou} (%{net_ou_p*100:.1f})")
    lines.append(f"KG Var: {net_btts} (%{net_btts_p*100:.1f})")
    
    lines.append("\n1X2 Olasılıkları:")
    lines.append(f"  Ev (1): %{blend.get('1', 0)*100:.1f}")
    lines.append(f"  Ber(X): %{blend.get('X', 0)*100:.1f}")
    lines.append(f"  Dep(2): %{blend.get('2', 0)*100:.1f}")
    
    corners = data.get("corner_analysis", {})
    if corners and corners.get("total_corners", 0) > 0:
        lines.append(f"\nKorner Tahmini (Toplam): {corners['total_corners']}")
        lines.append(f"  (Ev: {corners['predicted_home_corners']} | Dep: {corners['predicted_away_corners']})")
        
        line = 9.5
        p_over = corners["predictions"].get(f"O{line}", 0.0)
        pick = f"{line} ÜST" if p_over >= 0.5 else f"{line} ALT"
        lines.append(f"  Korner Net: {pick} (P(ÜST)≈%{p_over*100:.0f}, Güven: {corners.get('confidence','-')})")
        
    vb = data.get("value_bets", {})
    if vb.get("used_odds"):
        lines.append("\nBAHİS ANALİZİ (Value):")
        has_value = False
        for row in vb.get("table", []):
            if row["value"] >= VALUE_MIN and row["prob"] >= PROB_MIN and row["kelly"] >= KELLY_MIN:
                lines.append(f"  ✅ {row['market']}: Oran {row['odds']:.2f} | Value %{row['value']*100:+.1f} | Kelly %{row['qkelly']*100:.1f}")
                has_value = True
        if not has_value:
            lines.append("  ⚠️  Değerli bahis bulunamadı")
        lines.append(f"\n  KARAR: {vb.get('decision', 'Analiz edilemedi')}")
    else:
        lines.append("\nOran verisi yok - value analizi yapılamadı")
        
    ds = data["data_sources"]
    lines.append("\nKullanılan Veriler:")
    lines.append(f"  Standing: {'✓' if ds['standings_used'] else '✗'}")
    lines.append(f"  PSS (Ev:{ds['home_prev_matches']}|Dep:{ds['away_prev_matches']})")
    lines.append(f"  H2H: {'✓' if ds['h2h_matches']>0 else '✗'} ({ds['h2h_matches']} maç)")
    
    lw = data["lambda"]["info"].get("weights_used", {})
    if lw:
        lines.append("\nAğırlıklar:")
        for k, v in lw.items():
            k_name = {"standing": "Standing", "pss": "PSS", "h2h": "H2H"}.get(k, k)
            lines.append(f"  {k_name}: %{v*100:.0f}")
            
    lines.append("=" * 60)
    return "\n".join(lines)

def analyze_nowgoal(url: str, odds: Optional[Dict[str, float]] = None, mc_runs: int = MC_RUNS_DEFAULT) -> Dict[str, Any]:
    """Tüm analiz sürecini yöneten ana fonksiyon."""
    
    h2h_url = build_h2h_url(url)
    
    # 1. VERİ ÇEKME
    html = fetch_with_browser(h2h_url)
    
    home_team, away_team = parse_teams_from_title(html)
    if not home_team or not away_team:
        raise RuntimeError("Takım isimleri çıkarılamadı")
        
    league_match = re.search(r'<span[^>]*class=["\']?sclassLink["\']?[^>]*>([^<]+)</span>', html)
    league_name = strip_tags(league_match.group(1)) if league_match else ""
    
    # 2. STANDINGS (Puan Durumu)
    st_home_rows = extract_standings_for_team(html, home_team)
    st_away_rows = extract_standings_for_team(html, away_team)
    st_home = standings_to_splits(st_home_rows)
    st_away = standings_to_splits(st_away_rows)
    
    # 3. H2H (Aralarındaki Maçlar)
    h2h_used = extract_h2h_matches(html, home_team, away_team)[:H2H_N]
    
    # 4. PSS (Geçmiş Maçlar) - KADEMELİ FİLTRELEME
    prev_home, prev_away = extract_previous_home_away_same_league(
        page_source=html,
        home_team=home_team,
        away_team=away_team,
        league_name=league_name,
        max_take=RECENT_N,
    )
    
    home_prev_stats = build_prev_stats(home_team, prev_home)
    away_prev_stats = build_prev_stats(away_team, prev_away)
    
    # 5. LAMBDA HESAPLAMA
    lam_home, lam_away, lambda_info = compute_lambdas(
        st_home_s=st_home,
        st_away_s=st_away,
        home_prev=home_prev_stats,
        away_prev=away_prev_stats,
        h2h_used=h2h_used,
        home_team=home_team,
        away_team=away_team
    )
    
    # 6. POISSON & MONTE CARLO
    score_mat = build_score_matrix(lam_home, lam_away, MAX_GOALS_FOR_MATRIX)
    poisson_market = market_probs_from_matrix(score_mat)
    
    mc_runs = int(max(10_000, min(100_000, int(mc_runs))))
    mc = monte_carlo(lam_home, lam_away, n=mc_runs, seed=42)
    
    diff, diff_label = model_agreement(poisson_market, mc["p"])
    blended = blend_probs(poisson_market, mc["p"], BLEND_ALPHA)
    
    # 7. KORNER & VALUE
    corner_analysis = analyze_corners(home_prev_stats, away_prev_stats, h2h_used)
    
    if not odds:
        odds = extract_bet365_initial_odds(html)
        
    value_block = {"used_odds": False, "decision": "Oran Yok"}
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
        value_block["decision"] = final_decision(qualified, diff, diff_label)
        
    data = {
        "url": h2h_url,
        "teams": {"home": home_team, "away": away_team},
        "league": league_name,
        "lambda": {"home": lam_home, "away": lam_away, "info": lambda_info},
        "poisson": {"market_probs": poisson_market, "top7_scores": [(f"{h}-{a}", p) for (h, a), p in sorted(score_mat.items(), key=lambda x:x[1], reverse=True)[:7]]},
        "blended_probs": blended,
        "corner_analysis": corner_analysis,
        "value_bets": value_block,
        "data_sources": {
            "standings_used": len(st_home_rows) > 0,
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

# --- ÖNEMLİ: Render.com Deploy Sorunu İçin /health Rotası ---
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "service": "nowgoal-analyzer-api", "version": "5.6 (Full)"})

@app.route("/analiz_et", methods=["POST"])
def analiz_et():
    try:
        payload = request.get_json(silent=True) or {}
        url = payload.get("url", "").strip()
        
        if not url:
            return jsonify({"ok": False, "error": "URL yok"}), 400
            
        data = analyze_nowgoal(url)
        
        top_skor = data["poisson"]["top7_scores"][0][0]
        blend = data["blended_probs"]
        net_ou, net_ou_p, _ = net_ou_prediction(blend)
        net_btts, net_btts_p, _ = net_btts_prediction(blend)
        
        c_pick = "—"
        corners = data.get("corner_analysis", {})
        if corners.get("total_corners", 0) > 0:
            line = 9.5
            p_over = corners["predictions"].get(f"O{line}", 0.0)
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
        return jsonify({
            "ok": False, 
            "error": str(e), 
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_flag = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_flag)
