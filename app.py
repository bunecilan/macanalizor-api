# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - v5.1 (Strict Home/Away Filtering)
"""

import re
import math
import time
import traceback
import json
import os
import numpy as np
import requests
from flask import Flask, request, jsonify
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

# ======================
# AYARLAR
# ======================
MC_RUNS_DEFAULT = 10_000
H2H_LIMIT = 10
RECENT_LIMIT = 10 # Son 10 maçı baz al

# Ağırlıklar
W_STANDING = 0.40      
W_FORM = 0.35          # Form ağırlığını artırdık çünkü net filtreleme yapıyoruz
W_H2H = 0.25           

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

@dataclass
class MatchRow:
    league: str
    date: str
    home: str
    away: str
    ft_home: int
    ft_away: int

# ======================
# VERİ İŞLEME
# ======================
def normalize_name(name: str) -> str:
    """İsim eşleştirme için temizlik yapar (Genoa -> genoa)"""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def safe_get(url: str) -> str:
    if "h2h" not in url:
        base_id = re.search(r'\d+', url)
        if base_id: url = f"https://live3.nowgoal26.com/match/h2h-{base_id.group(0)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        raise RuntimeError(f"Site hatası: {e}")

def parse_match_row(cells: List[str]) -> Optional[MatchRow]:
    if len(cells) < 4: return None
    league = cells[0]
    date = cells[1]
    
    # Skor bul (2-1 gibi)
    ft_h, ft_a, found = 0, 0, False
    idx = -1
    for i, cell in enumerate(cells):
        m = re.search(r'^(\d+)-(\d+)$', cell.strip())
        if m:
            ft_h, ft_a = int(m.group(1)), int(m.group(2))
            idx = i
            found = True
            break
            
    if not found or idx < 1: return None
    
    home = cells[idx - 1]
    away = cells[idx + 1]
    return MatchRow(league, date, home, away, ft_h, ft_a)

def extract_matches_strict(html: str, home_team: str, away_team: str, current_league: str) -> Dict[str, List[MatchRow]]:
    """
    SENİN İSTEDİĞİN ÖZEL FİLTRELEME BURADA YAPILIYOR
    """
    # Tüm tablo satırlarını bul
    matches_raw = []
    tables = re.findall(r'<table[^>]*>(.*?)</table>', html, re.DOTALL)
    for table in tables:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.DOTALL)
        for row in rows:
            cells = re.findall(r'<t[d|h][^>]*>(.*?)</t[d|h]>', row, re.DOTALL)
            clean = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            m = parse_match_row(clean)
            if m: matches_raw.append(m)

    # İsimleri normalize et (küçük harf, noktalama yok)
    target_h = normalize_name(home_team)
    target_a = normalize_name(away_team)
    target_l = normalize_name(current_league)

    h2h_list = []
    home_same_league_home_only = [] # Ev Sahibinin EVDE oynadığı + Aynı Lig
    away_same_league_away_only = [] # Deplasmanın DEPTE oynadığı + Aynı Lig

    for m in matches_raw:
        m_h = normalize_name(m.home)
        m_a = normalize_name(m.away)
        m_l = normalize_name(m.league)

        # 1. H2H Kontrolü (İki takımın birbiriyle oynadığı)
        is_h2h = (target_h in m_h and target_a in m_a) or (target_h in m_a and target_a in m_h)
        if is_h2h:
            h2h_list.append(m)
            continue # H2H ise diğerlerine ekleme, mükerrer olmasın

        # 2. Previous Scores - EV SAHİBİ Analizi
        # Kural: Takım = Ev Sahibi VE Lig = Aynı Lig VE Yer = Ev (Home)
        if target_l in m_l: # Aynı Lig Filtresi
            if target_h in m_h: # Takım EV SAHİBİ hücresindeyse (Yani evde oynamış)
                home_same_league_home_only.append(m)

        # 3. Previous Scores - DEPLASMAN Analizi
        # Kural: Takım = Deplasman VE Lig = Aynı Lig VE Yer = Deplasman (Away)
        if target_l in m_l: # Aynı Lig Filtresi
            if target_a in m_a: # Takım DEPLASMAN hücresindeyse (Yani deplasmanda oynamış)
                away_same_league_away_only.append(m)

    return {
        "h2h": h2h_list[:H2H_LIMIT],
        "home_stats": home_same_league_home_only[:RECENT_LIMIT],
        "away_stats": away_same_league_away_only[:RECENT_LIMIT]
    }

def extract_odds_bet365(html: str) -> Optional[Dict[str, float]]:
    # Bet365 Initial bulmaya çalış
    try:
        # Tüm tablo satırlarını gez
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
        for row in rows:
            if "Bet365" in row:
                # Sayıları çek
                nums = re.findall(r'>(\d+\.\d{2})<', row)
                # Genellikle Initial oranlar satırın sonunda veya "Live"dan sonra gelir
                # Eğer 6 sayı varsa (Live 1X2 + Initial 1X2), son 3'ü al
                if len(nums) >= 6:
                    return {"1": float(nums[-3]), "X": float(nums[-2]), "2": float(nums[-1])}
                # Eğer sadece 3 sayı varsa (bazen sadece initial olur)
                elif len(nums) == 3:
                    return {"1": float(nums[0]), "X": float(nums[1]), "2": float(nums[2])}
        return None
    except:
        return None

# ======================
# ANALİZ
# ======================
def analyze_match_data(url: str):
    html = safe_get(url)
    
    # Takım isimlerini ve Ligi bul
    title_match = re.search(r'<title>(.*?)</title>', html)
    title = title_match.group(1) if title_match else "Home vs Away"
    
    # Basit isim çıkarma (Genoa vs Cagliari)
    parts = title.split(" VS ") if " VS " in title else title.split(" vs ")
    if len(parts) >= 2:
        home_team = parts[0].strip().split(" ")[0] # İlk kelimeyi al (Genoa)
        away_team = parts[1].strip().split(" ")[0] # İlk kelimeyi al (Cagliari)
    else:
        home_team, away_team = "Home", "Away"

    # Ligi bul (class="sclassLink")
    league_match = re.search(r'class=["\']?sclassLink["\']?[^>]*>(.*?)<', html)
    current_league = league_match.group(1) if league_match else ""

    # FİLTRELİ MAÇLARI ÇEK
    data = extract_matches_strict(html, home_team, away_team, current_league)
    matches_home = data["home_stats"]
    matches_away = data["away_stats"]
    matches_h2h = data["h2h"]

    # --- İSTATİSTİK HESAPLAMA (LAMBDA) ---
    # 1. Ev Sahibinin EVDEKİ Gücü
    if matches_home:
        # Sadece evde attığı ve yediği
        h_gf = sum(m.ft_home for m in matches_home) / len(matches_home)
        h_ga = sum(m.ft_away for m in matches_home) / len(matches_home)
    else:
        h_gf, h_ga = 1.3, 1.1 # Veri yoksa lig ortalaması

    # 2. Deplasmanın DEPLASMANDAKİ Gücü
    if matches_away:
        # Sadece deplasmanda attığı (m.ft_away) ve yediği (m.ft_home)
        a_gf = sum(m.ft_away for m in matches_away) / len(matches_away)
        a_ga = sum(m.ft_home for m in matches_away) / len(matches_away)
    else:
        a_gf, a_ga = 1.0, 1.4

    # 3. H2H Etkisi
    if matches_h2h:
        h2h_h_gf = 0; h2h_a_gf = 0
        for m in matches_h2h:
            # H2H'de kim ev kim dep önemli değil, takımların birbirine gol atma potansiyeli
            if normalize_name(home_team) in normalize_name(m.home):
                h2h_h_gf += m.ft_home; h2h_a_gf += m.ft_away
            else:
                h2h_h_gf += m.ft_away; h2h_a_gf += m.ft_home
        h2h_h_gf /= len(matches_h2h)
        h2h_a_gf /= len(matches_h2h)
    else:
        h2h_h_gf, h2h_a_gf = h_gf, a_gf

    # Beklenen Gol (Lambda) Formülü
    # Ev Gol Beklentisi = (Ev Formu * %35) + (H2H * %25) + (Lig Gücü * %40)
    lam_home = (h_gf * W_FORM) + (h2h_h_gf * W_H2H) + (1.4 * W_STANDING)
    lam_away = (a_gf * W_FORM) + (h2h_a_gf * W_H2H) + (1.0 * W_STANDING)

    # --- MONTE CARLO SİMÜLASYONU ---
    # 10.000 Maç Oynat
    sim_home = np.random.poisson(lam_home, MC_RUNS_DEFAULT)
    sim_away = np.random.poisson(lam_away, MC_RUNS_DEFAULT)
    
    # Sonuçlar
    home_wins = np.sum(sim_home > sim_away)
    draws = np.sum(sim_home == sim_away)
    away_wins = np.sum(sim_home < sim_away)
    
    prob_1 = home_wins / MC_RUNS_DEFAULT
    prob_x = draws / MC_RUNS_DEFAULT
    prob_2 = away_wins / MC_RUNS_DEFAULT
    
    # Alt/Üst ve KG
    total_goals = sim_home + sim_away
    prob_o25 = np.sum(total_goals > 2.5) / MC_RUNS_DEFAULT
    prob_btts = np.sum((sim_home > 0) & (sim_away > 0)) / MC_RUNS_DEFAULT

    # En Olası Skor
    scores = [f"{h}-{a}" for h, a in zip(sim_home, sim_away)]
    most_common = Counter(scores).most_common(1)[0][0]

    # --- VALUE BET & KARAR ---
    odds = extract_odds_bet365(html)
    decision = "Oran Yok"
    value_msg = ""
    
    if odds:
        # Value Hesapla: (Oran x Olasılık) - 1
        v1 = (odds["1"] * prob_1) - 1
        vx = (odds["X"] * prob_x) - 1
        v2 = (odds["2"] * prob_2) - 1
        
        candidates = []
        if v1 > 0.05 and prob_1 > 0.50: candidates.append(f"MS 1 (Value %{v1*100:.1f})")
        if v2 > 0.05 and prob_2 > 0.50: candidates.append(f"MS 2 (Value %{v2*100:.1f})")
        if vx > 0.05: candidates.append(f"Beraberlik (Value %{vx*100:.1f})")
        
        if candidates:
            decision = f"OYNANABİLİR -> {candidates[0]}"
        else:
            decision = "OYNAMA (Değer Düşük)"
            
        value_msg = f"\nBet365: 1({odds['1']}) X({odds['X']}) 2({odds['2']})"

    # Korner Tahmini (Basit)
    # H2H ortalaması yoksa 9.5 kabul et
    total_cn = 9.5 
    corner_res = "9.5 ÜST" if total_cn > 9.5 else "9.5 ALT"

    return {
        "ok": True,
        "skor": most_common,
        "alt_ust": f"2.5 ÜST (%{prob_o25*100:.1f})" if prob_o25 > 0.5 else f"2.5 ALT (%{(1-prob_o25)*100:.1f})",
        "btts": f"VAR (%{prob_btts*100:.1f})" if prob_btts > 0.5 else "YOK",
        "karar": decision,
        "detay": f"""
ANALİZ DETAYI
-------------
Ev (Evinde+Aynı Lig): {len(matches_home)} maç bulundu. Ort Gol: {h_gf:.1f}
Dep (Depte+Aynı Lig): {len(matches_away)} maç bulundu. Ort Gol: {a_gf:.1f}
H2H: {len(matches_h2h)} maç.

Beklenen Gol: Ev {lam_home:.2f} - Dep {lam_away:.2f}
Olasılıklar: %{prob_1*100:.0f} - %{prob_x*100:.0f} - %{prob_2*100:.0f}
{value_msg}
        """
    }

# ======================
# FLASK BAŞLATMA
# ======================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index(): return jsonify({"status": "Active", "v": "5.1"})

@app.route("/analiz_et", methods=["POST"])
def api_analyze():
    try:
        data = request.get_json(force=True)
        return jsonify(analyze_match_data(data.get("url")))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
