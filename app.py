# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - Enhanced Version 5.0 (ROL Logic & Corners)
Flask API with Corner Analysis, Specific Standings Weighting, and Value Betting
"""

import re
import math
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import requests
from flask import Flask, request, jsonify

# ======================
# CONFIG & CONSTANTS
# ======================
MC_RUNS_DEFAULT = 10_000
RECENT_N = 10
H2H_N = 10

# ROL Logic Weights
W_ST_BASE = 0.50      # Standing (Total/Home/Away)
W_FORM_BASE = 0.30    # Form (Recent matches)
W_LAST6_BASE = 0.10   # Last 6 (from Standings)
W_H2H_BASE = 0.10     # Head to Head

# Value Bet Thresholds
VALUE_MIN = 0.05      # Minimum 5% value
PROB_MIN = 0.45       # Minimum 45% probability
KELLY_DIVISOR = 4.0   # Kelly / 4 (Conservative)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
}

# ======================
# DATA STRUCTURES
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
class SplitStats:
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
class TeamStats:
    name: str
    # From Standings Table
    total: Optional[SplitStats] = None
    home: Optional[SplitStats] = None
    away: Optional[SplitStats] = None
    last6: Optional[SplitStats] = None
    # From Previous Matches Analysis
    recent_corner_avg_for: float = 0.0
    recent_corner_avg_against: float = 0.0

# ======================
# UTILITIES
# ======================
def clean_text(s: str) -> str:
    """HTML taglerini temizle ve boşlukları düzelt"""
    s = re.sub(r"<script\b.*?</script>", "", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"<style\b.*?</style>", "", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()

def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def extract_tables(html: str) -> List[str]:
    return [m.group(0) for m in re.finditer(r"<table\b[^>]*>.*?</table>", html, flags=re.IGNORECASE | re.DOTALL)]

def extract_rows(table_html: str) -> List[List[str]]:
    rows = []
    trs = re.findall(r"<tr\b[^>]*>.*?</tr>", table_html, flags=re.IGNORECASE | re.DOTALL)
    for tr in trs:
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, flags=re.IGNORECASE | re.DOTALL)
        if cells:
            cleaned = [clean_text(c) for c in cells]
            rows.append(cleaned)
    return rows

def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

# ======================
# PARSING LOGIC
# ======================

def parse_odds_bet365(html: str) -> Optional[Dict[str, float]]:
    """
    Bet365 Initial oranlarını 1X2 tablosundan çeker.
    HTML içinde 'Bet365' satırını ve altındaki 'Initial' satırını arar.
    """
    # Tablo bazlı arama
    tables = extract_tables(html)
    for tbl in tables:
        if "1X2 Odds" not in tbl and "Bet365" not in tbl:
            continue
        
        rows = extract_rows(tbl)
        found_bet365 = False
        
        for i, row in enumerate(rows):
            row_text = " ".join(row).lower()
            
            # Bet365 başlığını bul
            if "bet365" in row_text:
                found_bet365 = True
                # Aynı satırda Initial varsa (bazen yan yana olur)
                if "initial" in row_text:
                    # Genellikle son 3 sayısal değer orandır, ya da belirli bir column index
                    # Basit regex ile satırdaki float'ları alalım
                    nums = re.findall(r"\d+\.\d{2}", row_text)
                    if len(nums) >= 3:
                        # Genellikle Initial 1-X-2 sırasıyla gelir
                        return {"1": float(nums[0]), "X": float(nums[1]), "2": float(nums[2])}
                continue

            # Bir sonraki satırlarda "Initial" ara
            if found_bet365 and "initial" in row_text:
                nums = re.findall(r"\d+\.\d{2}", row_text)
                if len(nums) >= 3:
                     # Bazen ilk değer handikap vs olabilir, ama 1x2 tablosunda genellikle oranlardır
                     # Güvenlik için büyüklük kontrolü yapılabilir ama şimdilik direkt alıyoruz
                    return {"1": float(nums[0]), "X": float(nums[1]), "2": float(nums[2])}
                found_bet365 = False # Reset if standard flow breaks
                
    # Fallback: Text search
    # Pattern: Bet365 ... Initial ... 2.04 ... 2.98 ... 3.61
    try:
        # Regex ile daha geniş arama
        block = re.search(r"Bet365.*?Initial.*?(?P<h>\d+\.\d+).*?(?P<d>\d+\.\d+).*?(?P<a>\d+\.\d+)", html, re.DOTALL | re.IGNORECASE)
        if block:
            return {"1": float(block.group("h")), "X": float(block.group("d")), "2": float(block.group("a"))}
    except:
        pass
        
    return None

def parse_standings(html: str, team_name: str) -> Dict[str, SplitStats]:
    """
    Takımın Standings tablosundaki Total, Home, Away, Last 6 verilerini çeker.
    """
    stats = {}
    team_key = norm_key(team_name)
    
    tables = extract_tables(html)
    target_table = None
    
    # Standings tablosunu bul (genellikle takım ismini ve Total/Home/Away içerir)
    for tbl in tables:
        if "Standings" in tbl or ("Total" in tbl and "Home" in tbl and "Away" in tbl):
            if team_key in norm_key(clean_text(tbl)):
                target_table = tbl
                break
    
    if not target_table:
        return stats

    rows = extract_rows(target_table)
    # Header indekslerini bulmak zor olabilir, standart pozisyon varsayımı:
    # Genellikle: Type | Match | Win | Draw | Loss | Scored | Conceded | ...
    
    for row in rows:
        if not row: continue
        label = row[0].strip() # Total, Home, Away, Last 6
        
        # Satır verilerini temizle
        vals = [x for x in row if x.replace("-","").isdigit()]
        
        # En az 6 sayısal veri lazım (M, W, D, L, GF, GA)
        if len(vals) < 6:
            continue
            
        try:
            matches = int(vals[0])
            # Win = vals[1], Draw = vals[2], Loss = vals[3]
            gf = int(vals[4])
            ga = int(vals[5])
            
            s = SplitStats(matches, gf, ga)
            
            if "Total" in label: stats["Total"] = s
            elif "Home" in label: stats["Home"] = s
            elif "Away" in label: stats["Away"] = s
            elif "Last" in label and "6" in label: stats["Last 6"] = s
            
        except:
            continue
            
    return stats

def parse_matches(html: str, section_marker: str, filter_league: str = None) -> List[MatchRow]:
    """
    Belirli bir bölümün altındaki maçları çeker.
    Korner formatı: Genellikle skorun yanında veya ayrı sütunda "10-2" veya "10-2(5-1)"
    """
    matches = []
    
    # Bölümü bul
    idx = html.find(section_marker)
    if idx == -1: return []
    
    # Bölümden sonraki ilk tabloyu al
    sub_html = html[idx:]
    tables = extract_tables(sub_html)
    if not tables: return []
    
    # Genellikle ilk tablo maç tablosudur
    rows = extract_rows(tables[0])
    
    for row in rows:
        if len(row) < 5: continue
        
        # Tarih kontrolü
        date_str = ""
        for cell in row:
            if re.match(r"\d{2}-\d{2}-\d{4}", cell) or re.match(r"\d{2}-\d{2}", cell):
                date_str = cell
                break
        if not date_str: continue

        # Lig kontrolü (Eğer filtre varsa)
        league = row[0] # Genellikle ilk sütun
        if filter_league and norm_key(filter_league) != norm_key(league):
            if norm_key(filter_league) not in norm_key(league): # Kapsama kontrolü
                continue

        # Skor ve Takımlar
        # Format: League | Date | Home | Score | Away | ...
        # Score Regex: 2-1 veya 2-1(1-0)
        score_idx = -1
        for i, cell in enumerate(row):
            if re.match(r"^\d+-\d+(\s*\(.*?\))?$", cell):
                score_idx = i
                break
        
        if score_idx == -1 or score_idx < 2: continue
        
        home = row[score_idx - 1]
        away = row[score_idx + 1]
        score_txt = row[score_idx]
        
        # Skor Parse
        m = re.match(r"(\d+)-(\d+)", score_txt)
        if not m: continue
        ft_h, ft_a = int(m.group(1)), int(m.group(2))
        
        # Korner Parse
        # Genellikle Away takımından sonraki sütunlarda
        # Format: 12-4(6-2) -> FT(HT)
        c_h, c_a = None, None
        for i in range(score_idx + 2, len(row)):
            cell = row[i]
            # Korner hücresi skor formatına benzer ama skor hücresi değildir
            # Genellikle parantez içerir
            cm = re.match(r"(\d+)-(\d+)\s*\((\d+)-(\d+)\)", cell)
            if cm:
                c_h, c_a = int(cm.group(1)), int(cm.group(2))
                break
            # Veya sadece FT korner
            cm_simple = re.match(r"(\d+)-(\d+)", cell)
            if cm_simple and i != score_idx: # Skor hücresi değilse
                # Bazen HT skoru ayrı sütunda olur, bunu korner sanmamalıyız
                # Korner genelde HT skorundan sonra gelir. Basitçe kabul edelim.
                c_h, c_a = int(cm_simple.group(1)), int(cm_simple.group(2))
                # HT Score ile karıştırmamak için basit bir kontrol:
                # Genellikle toplam korner sayısı gol sayısından fazladır (her zaman değil)
                break
        
        matches.append(MatchRow(
            league=league, date=date_str, home=home, away=away,
            ft_home=ft_h, ft_away=ft_a, corner_home=c_h, corner_away=c_a
        ))
        
    return matches

# ======================
# CORE ANALYTICS
# ======================

def analyze_match(url: str, override_odds: dict = None) -> Dict[str, Any]:
    
    # 1. Fetch
    try:
        # URL düzeltme (h2h URL'si tam veri içerir)
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return {"error": f"Veri çekilemedi: {str(e)}"}

    # 2. Basic Info
    title = re.search(r"<title>(.*?)</title>", html, re.I)
    title_text = title.group(1) if title else "Unknown vs Unknown"
    
    # Takım isimlerini title'dan al (Genoa vs Cagliari ...)
    if "vs" in title_text.lower():
        parts = re.split(r"\s+vs\.?\s+", title_text, flags=re.I)
        home_team, away_team = parts[0].strip(), parts[1].split("|")[0].strip()
    else:
        # Fallback
        home_team, away_team = "Home", "Away"

    # Lig ismini bul (Genellikle sayfa başında)
    league_match = re.search(r'class="sclassLink"[^>]*>(.*?)</a>', html) or re.search(r'class="LName"[^>]*>(.*?)</span>', html)
    league_name = clean_text(league_match.group(1)) if league_match else ""

    # 3. Parse Odds (Bet365 Initial)
    odds = override_odds if override_odds else parse_odds_bet365(html)
    
    # 4. Parse Standings (ROL Logic needs Total, Home, Away, Last6)
    st_home_raw = parse_standings(html, home_team)
    st_away_raw = parse_standings(html, away_team)
    
    # 5. Parse H2H
    h2h_matches = parse_matches(html, "Head to Head Statistics")
    
    # 6. Parse Previous (Home & Away with Same League Filter)
    # NowGoal'de Previous Scores tablosunda filtreleme JS ile yapılır.
    # HTML'de "Same League" filtresi uygulanmış veri olmayabilir.
    # Bu yüzden tüm listeyi çekip Python tarafında filtreleyeceğiz.
    prev_home_all = parse_matches(html, "Previous Scores Statistics") # Home section
    # Away bölümünü bulmak için ikinci "Previous Scores Statistics" veya tablo sırasına bakmak lazım
    # Ancak parse_matches sadece ilk bulduğunu alır.
    # Manuel split yapalım:
    sections = html.split("Previous Scores Statistics")
    if len(sections) >= 3:
        prev_home_matches = parse_matches("Previous Scores Statistics" + sections[1], "Previous Scores Statistics", filter_league=league_name)
        prev_away_matches = parse_matches("Previous Scores Statistics" + sections[2], "Previous Scores Statistics", filter_league=league_name)
    else:
        # Fallback: Tek tablo varsa
        prev_home_matches = []
        prev_away_matches = []

    # Filter to Home/Away specific logic
    # Home Team's matches: Only where they played Home? OR All matches?
    # ROL Logic: "Ev sahibinde Home ve Same League deplasmanda yine AWAY ve Same league seçili olacak"
    # User instruction says: Home tab -> Home selected. Away tab -> Away selected.
    # We need to filter prev_home_matches to only where home_team was actually Home.
    
    final_home_matches = [m for m in prev_home_matches if norm_key(m.home) == norm_key(home_team)][:RECENT_N]
    final_away_matches = [m for m in prev_away_matches if norm_key(m.away) == norm_key(away_team)][:RECENT_N]

    # ======================
    # CALCULATIONS (ROL LOGIC)
    # ======================
    
    # --- LAMBDA (Expected Goals) ---
    # Formula:
    # 1. Standings (Total/Home/Away) -> Weight 50%
    # 2. Form (Recent filtered) -> Weight 30%
    # 3. Last 6 (Standings) -> Weight 10%
    # 4. H2H -> Weight 10%

    def calc_xg(matches: List[MatchRow], is_home_team: bool) -> float:
        if not matches: return 0.0
        goals = sum(m.ft_home if is_home_team else m.ft_away for m in matches)
        return goals / len(matches)

    def calc_xga(matches: List[MatchRow], is_home_team: bool) -> float:
        if not matches: return 0.0
        conceded = sum(m.ft_away if is_home_team else m.ft_home for m in matches)
        return conceded / len(matches)
        
    lambdas_h = []
    lambdas_a = []
    weights = []
    
    # 1. Standings
    # Home Team Home Stats vs Away Team Away Stats
    sh = st_home_raw.get("Home")
    sa = st_away_raw.get("Away")
    if sh and sa and sh.matches > 0 and sa.matches > 0:
        # Home xG = (Home's Home Scored + Away's Away Conceded) / 2
        l_h = (sh.gf_pg + sa.ga_pg) / 2
        l_a = (sa.gf_pg + sh.ga_pg) / 2
        lambdas_h.append(l_h)
        lambdas_a.append(l_a)
        weights.append(W_ST_BASE)
    
    # 2. Form (Filtered Matches)
    if final_home_matches and final_away_matches:
        f_h_xg = calc_xg(final_home_matches, True)
        f_a_xga = calc_xga(final_away_matches, False) # Away team playing away
        
        l_h = (f_h_xg + f_a_xga) / 2 # Basitleştirilmiş formül
        
        f_a_xg = calc_xg(final_away_matches, False)
        f_h_xga = calc_xga(final_home_matches, True)
        
        l_a = (f_a_xg + f_h_xga) / 2
        
        lambdas_h.append(l_h)
        lambdas_a.append(l_a)
        weights.append(W_FORM_BASE)
        
    # 3. Last 6
    sl6_h = st_home_raw.get("Last 6")
    sl6_a = st_away_raw.get("Last 6")
    if sl6_h and sl6_a:
        l_h = (sl6_h.gf_pg + sl6_a.ga_pg) / 2
        l_a = (sl6_a.gf_pg + sl6_h.ga_pg) / 2
        lambdas_h.append(l_h)
        lambdas_a.append(l_a)
        weights.append(W_LAST6_BASE)
        
    # 4. H2H
    if h2h_matches:
        h2h_h_goals = sum(m.ft_home if norm_key(m.home) == norm_key(home_team) else m.ft_away for m in h2h_matches)
        h2h_a_goals = sum(m.ft_away if norm_key(m.home) == norm_key(home_team) else m.ft_home for m in h2h_matches)
        l_h = h2h_h_goals / len(h2h_matches)
        l_a = h2h_a_goals / len(h2h_matches)
        lambdas_h.append(l_h)
        lambdas_a.append(l_a)
        weights.append(W_H2H_BASE)
        
    # Weighted Average
    if sum(weights) > 0:
        lambda_home = sum(l * w for l, w in zip(lambdas_h, weights)) / sum(weights)
        lambda_away = sum(l * w for l, w in zip(lambdas_a, weights)) / sum(weights)
    else:
        lambda_home, lambda_away = 1.0, 1.0 # Default
        
    # --- CORNER ANALYSIS ---
    # H2H Avg
    h2h_corners = [m.corner_home + m.corner_away for m in h2h_matches if m.corner_home is not None]
    avg_h2h_corner = sum(h2h_corners) / len(h2h_corners) if h2h_corners else 0
    
    # Recent Home Avg (Home playing Home)
    rc_h_list = [m.corner_home + m.corner_away for m in final_home_matches if m.corner_home is not None]
    avg_rc_h = sum(rc_h_list) / len(rc_h_list) if rc_h_list else 0
    
    # Recent Away Avg (Away playing Away)
    rc_a_list = [m.corner_home + m.corner_away for m in final_away_matches if m.corner_home is not None]
    avg_rc_a = sum(rc_a_list) / len(rc_a_list) if rc_a_list else 0
    
    # Prediction: Weighted (40% H2H + 30% Home + 30% Away)
    # Eğer H2H yoksa sadece form
    if avg_h2h_corner > 0:
        pred_corner = (avg_h2h_corner * 0.4) + (avg_rc_h * 0.3) + (avg_rc_a * 0.3)
    elif avg_rc_h > 0 and avg_rc_a > 0:
        pred_corner = (avg_rc_h + avg_rc_a) / 2
    else:
        pred_corner = 9.5 # Fallback
        
    # --- SIMULATION (Poisson + Monte Carlo) ---
    def poisson(k, lam):
        return (lam**k * math.exp(-lam)) / math.factorial(k)
        
    probs = {"1": 0, "X": 0, "2": 0, "O2.5": 0, "U2.5": 0, "BTTS": 0}
    score_matrix = {}
    
    for h in range(6):
        for a in range(6):
            p = poisson(h, lambda_home) * poisson(a, lambda_away)
            score_matrix[f"{h}-{a}"] = p
            
            if h > a: probs["1"] += p
            elif h == a: probs["X"] += p
            else: probs["2"] += p
            
            if h + a > 2.5: probs["O2.5"] += p
            else: probs["U2.5"] += p
            
            if h > 0 and a > 0: probs["BTTS"] += p
            
    top_scores = sorted(score_matrix.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # --- VALUE BET & DECISION ---
    decision_text = "Oran Yok"
    value_list = []
    
    if odds:
        for outcome in ["1", "X", "2"]:
            p = probs[outcome]
            o = odds.get(outcome, 0)
            if o > 1.0:
                value = (p * o) - 1
                kelly = ((p * o - 1) / (o - 1)) / KELLY_DIVISOR
                
                if value > VALUE_MIN and p > PROB_MIN:
                    value_list.append(f"{outcome} Value! (Prob: %{p*100:.1f}, Odds: {o}, Val: %{value*100:.1f})")
                    
        if value_list:
            decision_text = " | ".join(value_list)
        else:
            # Fallback Decision based on raw Prob
            if probs["1"] > 0.60: decision_text = "MS 1 Güçlü"
            elif probs["2"] > 0.60: decision_text = "MS 2 Güçlü"
            elif probs["O2.5"] > 0.65: decision_text = "2.5 ÜST"
            else: decision_text = "Riskli / Pas"
            
    # Format Output
    net_ou = "ÜST" if probs["O2.5"] > probs["U2.5"] else "ALT"
    net_ou_conf = max(probs["O2.5"], probs["U2.5"])
    
    net_btts = "VAR" if probs["BTTS"] > 0.55 else "YOK"
    
    comprehensive_report = f"""
    ANALİZ RAPORU (ROL KRİTERLERİ)
    ------------------------------
    Maç: {home_team} vs {away_team}
    Lig: {league_name}
    
    HESAPLANAN GÜÇ (Lambda):
    Ev Sahibi Gol Beklentisi: {lambda_home:.2f}
    Deplasman Gol Beklentisi: {lambda_away:.2f}
    
    TAHMİNLER:
    Skor: {top_scores[0][0]} (%{top_scores[0][1]*100:.1f}) - Alternatif: {top_scores[1][0]}
    Alt/Üst: 2.5 {net_ou} (%{net_ou_conf*100:.1f})
    KG (BTTS): {net_btts} (%{probs['BTTS']*100:.1f})
    
    KORNER ANALİZİ:
    H2H Ort: {avg_h2h_corner:.1f}
    Ev (İç Saha) Ort: {avg_rc_h:.1f}
    Dep (Dış Saha) Ort: {avg_rc_a:.1f}
    Tahmini Korner: {pred_corner:.1f} ({'9.5 ÜST' if pred_corner > 9.5 else '9.5 ALT'})
    
    VALUE BET DURUMU (Bet365 Initial):
    Oranlar: {odds}
    Karar: {decision_text}
    """
    
    return {
        "ok": True,
        "skor": top_scores[0][0],
        "alt_ust": f"2.5 {net_ou} (%{net_ou_conf*100:.0f})",
        "btts": f"{net_btts} (%{probs['BTTS']*100:.0f})",
        "karar": decision_text,
        "detay": comprehensive_report,
        "raw_data": {
            "lambda": {"h": lambda_home, "a": lambda_away},
            "corners": {"pred": pred_corner},
            "odds": odds
        }
    }

# ======================
# FLASK ROUTES
# ======================
app = Flask(__name__)

@app.route('/analiz_et', methods=['POST'])
def analyze_endpoint():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({"ok": False, "error": "URL eksik"})
        
    try:
        result = analyze_match(url)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)})

if __name__ == '__main__':
    print("NowGoal Analyzer v5.0 Started...")
    app.run(host='0.0.0.0', port=5000)
