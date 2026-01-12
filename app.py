# -*- coding: utf-8 -*-
"""
NowGoal Match Analyzer - Ultimate Version 5.0
Flask API with Fixed Weight System & Render.com Deployment
"""

import os
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
from flask_cors import CORS  # CORS için

# ======================
# CONFIG - SABİT AĞIRLIKLAR
# ======================
WEIGHT_STANDINGS = 0.50    # %50 Standing verisi
WEIGHT_H2H = 0.30          # %30 H2H verisi
WEIGHT_PREVIOUS = 0.20     # %20 Previous Scores Statistics

MC_RUNS_DEFAULT = 10_000
RECENT_N = 10
H2H_N = 10

# Render için optimize edilmiş timeout
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
}

# Render free tier için cache sistemi
ANALYSIS_CACHE = {}
CACHE_TIMEOUT = 300  # 5 dakika

# ======================
# UTILITY FUNCTIONS
# ======================
def norm_key(s: str) -> str:
    """Normalize team names for comparison"""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def parse_corners(corner_str: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse corner string like "12-1(3-1)" into:
    total_home, total_away
    """
    if not corner_str:
        return None, None
    
    corner_str = corner_str.strip()
    total_match = re.match(r'(\d+)-(\d+)', corner_str)
    if total_match:
        home_corners = int(total_match.group(1))
        away_corners = int(total_match.group(2))
        return home_corners, away_corners
    
    return None, None

def safe_get_url(url: str, timeout: int = 15):
    """Safe URL fetching with retry for Render"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"URL fetch error: {e}")
        return ""

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
    corner_home: Optional[int] = None
    corner_away: Optional[int] = None

@dataclass
class StandRow:
    ft: str
    matches: int
    win: int
    draw: int
    loss: int
    scored: int
    conceded: int
    pts: int
    rank: int
    rate: str = ""

@dataclass
class TeamStats:
    name: str
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: float = 0.0
    goals_against: float = 0.0
    corners_for: float = 0.0
    corners_against: float = 0.0
    clean_sheets: int = 0
    btts: int = 0

# ======================
# HTML PARSING FUNCTIONS (Simplified for Render)
# ======================
def extract_tables_html(html: str) -> List[str]:
    """Extract all table tags from HTML"""
    if not html:
        return []
    return re.findall(r'<table[^>]*>.*?</table>', html, re.DOTALL | re.IGNORECASE)

def extract_table_rows(table_html: str) -> List[List[str]]:
    """Extract rows and cells from table HTML"""
    rows = []
    try:
        row_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
        
        for row in row_matches:
            row_clean = re.sub(r'<img[^>]*>', '', row)
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_clean, re.DOTALL | re.IGNORECASE)
            if cells:
                cleaned_cells = []
                for cell in cells:
                    text = re.sub(r'<[^>]+>', '', cell)
                    text = re.sub(r'&nbsp;', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    if text:  # Skip empty cells
                        cleaned_cells.append(text)
                if cleaned_cells and len(cleaned_cells) >= 3:
                    rows.append(cleaned_cells)
    except Exception as e:
        print(f"Table row extraction error: {e}")
    return rows

def find_section(html: str, section_name: str) -> str:
    """Find a specific section in HTML"""
    try:
        pattern = rf'(?i){re.escape(section_name)}.*?(?=<h\d>|$)'
        match = re.search(pattern, html, re.DOTALL)
        return match.group(0) if match else ""
    except:
        return ""

def extract_match_from_row(row: List[str]) -> Optional[MatchRow]:
    """Extract match data from a table row"""
    if len(row) < 6:
        return None
    
    try:
        # Find score
        score_match = None
        score_idx = -1
        
        for idx, cell in enumerate(row):
            if re.search(r'\d+\s*-\s*\d+', cell):
                score_match = re.search(r'(\d+)\s*-\s*(\d+)', cell)
                if score_match:
                    score_idx = idx
                    break
        
        if not score_match or score_idx == -1:
            return None
        
        home_score = int(score_match.group(1))
        away_score = int(score_match.group(2))
        
        # Get teams (assuming teams are around the score)
        if score_idx > 0 and score_idx < len(row) - 1:
            home_team = row[score_idx - 1].strip()
            away_team = row[score_idx + 1].strip()
        else:
            return None
        
        # Find corners
        corner_home = corner_away = None
        for cell in row:
            corners_match = re.search(r'(\d+)\s*-\s*(\d+)\s*(?:\([^)]+\))?', cell)
            if corners_match and cell != f"{home_score}-{away_score}":
                corner_home = int(corners_match.group(1))
                corner_away = int(corners_match.group(2))
                break
        
        # Date (usually in first column)
        match_date = ""
        for cell in row[:3]:
            if re.match(r'\d{2}-\d{2}-\d{4}', cell):
                match_date = cell
                break
        
        league = row[1] if len(row) > 1 and not re.match(r'\d{2}-\d{2}-\d{4}', row[1]) else "Unknown"
        
        return MatchRow(
            league=league,
            date=match_date,
            home=home_team,
            away=away_team,
            ft_home=home_score,
            ft_away=away_score,
            corner_home=corner_home,
            corner_away=corner_away
        )
    except Exception as e:
        return None

# ======================
# DATA EXTRACTION FUNCTIONS
# ======================
def extract_standings_data(html: str, team_name: str) -> Dict[str, StandRow]:
    """Extract standings data for a specific team"""
    standings = {}
    
    standings_section = find_section(html, "Standings")
    if not standings_section:
        return standings
    
    tables = extract_tables_html(standings_section)
    
    for table in tables:
        rows = extract_table_rows(table)
        if len(rows) < 3:
            continue
        
        header_text = ' '.join(rows[0]).lower() if rows else ""
        
        if any(keyword in header_text for keyword in ['matches', 'win', 'draw', 'loss', 'pts']):
            for row in rows[1:]:
                if len(row) >= 9:
                    try:
                        row_text = ' '.join(row).lower()
                        team_key = norm_key(team_name)
                        
                        if team_key in norm_key(row_text) or any(team_key in norm_key(cell) for cell in row[:3]):
                            stand_row = StandRow(
                                ft=row[0],
                                matches=int(row[1]) if row[1].isdigit() else 0,
                                win=int(row[2]) if row[2].isdigit() else 0,
                                draw=int(row[3]) if row[3].isdigit() else 0,
                                loss=int(row[4]) if row[4].isdigit() else 0,
                                scored=int(row[5]) if row[5].isdigit() else 0,
                                conceded=int(row[6]) if row[6].isdigit() else 0,
                                pts=int(row[7]) if row[7].isdigit() else 0,
                                rank=int(row[8]) if len(row) > 8 and row[8].isdigit() else 0,
                                rate=row[9] if len(row) > 9 else ""
                            )
                            standings[stand_row.ft] = stand_row
                    except (ValueError, IndexError):
                        continue
    
    return standings

def extract_h2h_matches(html: str) -> List[MatchRow]:
    """Extract Head-to-Head matches"""
    matches = []
    
    for section_name in ["Head to Head Statistics", "H2H Statistics", "Head to Head"]:
        h2h_section = find_section(html, section_name)
        if h2h_section:
            tables = extract_tables_html(h2h_section)
            for table in tables[:2]:  # İlk 2 tablo
                rows = extract_table_rows(table)
                for row in rows:
                    match_data = extract_match_from_row(row)
                    if match_data:
                        matches.append(match_data)
            if matches:
                break
    
    return matches[:10]  # Max 10 maç

def extract_previous_matches(html: str, team_type: str = "home") -> List[MatchRow]:
    """
    Extract previous matches for home or away team
    team_type: "home" for Home+Same League, "away" for Away+Same League
    """
    matches = []
    
    prev_section = find_section(html, "Previous Scores Statistics")
    if not prev_section:
        return matches
    
    tables = extract_tables_html(prev_section)
    
    if team_type == "home":
        table_idx = 0
    else:
        table_idx = 1 if len(tables) > 1 else 0
    
    if table_idx < len(tables):
        rows = extract_table_rows(tables[table_idx])
        for row in rows:
            match_data = extract_match_from_row(row)
            if match_data:
                matches.append(match_data)
    
    return matches[:8]  # Max 8 maç

def extract_bet365_odds(html: str) -> Dict[str, float]:
    """Extract Bet365 Initial 1X2 odds"""
    odds = {}
    
    patterns = [
        r'Bet365.*?Initial.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)',
        r'bet365.*?initial.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)',
        r'1\s*[/:]\s*(\d+\.\d+).*?X\s*[/:]\s*(\d+\.\d+).*?2\s*[/:]\s*(\d+\.\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                odds = {
                    "1": float(match.group(1)),
                    "X": float(match.group(2)),
                    "2": float(match.group(3))
                }
                return odds
            except (ValueError, IndexError):
                continue
    
    return odds

def extract_teams_from_html(html: str) -> Tuple[str, str]:
    """Extract team names from HTML"""
    # Title'dan çıkar
    title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
    if title_match:
        title = title_match.group(1)
        vs_match = re.search(r'([^VS]+)\s+VS\s+([^<|]+)', title, re.IGNORECASE)
        if vs_match:
            return vs_match.group(1).strip(), vs_match.group(2).strip()
    
    return "Ev Sahibi", "Deplasman"

# ======================
# STATISTICAL ANALYSIS - SABİT AĞIRLIK SİSTEMİ
# ======================
def calculate_team_stats(matches: List[MatchRow], team_name: str) -> TeamStats:
    """Calculate statistics for a team from matches"""
    stats = TeamStats(name=team_name)
    team_key = norm_key(team_name)
    
    if not matches:
        return stats
    
    for match in matches:
        stats.matches += 1
        
        is_home = norm_key(match.home) == team_key
        
        if is_home:
            goals_for = match.ft_home
            goals_against = match.ft_away
            corners_for = match.corner_home or 0
            corners_against = match.corner_away or 0
        else:
            goals_for = match.ft_away
            goals_against = match.ft_home
            corners_for = match.corner_away or 0
            corners_against = match.corner_home or 0
        
        stats.goals_for += goals_for
        stats.goals_against += goals_against
        stats.corners_for += corners_for
        stats.corners_against += corners_against
        
        if goals_for > goals_against:
            stats.wins += 1
        elif goals_for == goals_against:
            stats.draws += 1
        else:
            stats.losses += 1
        
        if goals_against == 0:
            stats.clean_sheets += 1
        
        if goals_for > 0 and goals_against > 0:
            stats.btts += 1
    
    if stats.matches > 0:
        stats.goals_for = stats.goals_for / stats.matches
        stats.goals_against = stats.goals_against / stats.matches
        stats.corners_for = stats.corners_for / stats.matches
        stats.corners_against = stats.corners_against / stats.matches
    
    return stats

def calculate_expected_goals(home_stats: TeamStats, away_stats: TeamStats,
                           h2h_matches: List[MatchRow], home_standings: Dict,
                           away_standings: Dict) -> Tuple[float, float]:
    """
    Beklenen golleri SABİT AĞIRLIKLARLA hesapla:
    %50 Standing + %30 H2H + %20 Previous Scores
    """
    
    # 1. STANDING BİLEŞENİ (%50 Ağırlık)
    standing_home_xg = standing_away_xg = 0.0
    
    if "Home" in home_standings and "Away" in away_standings:
        home_stand = home_standings["Home"]
        away_stand = away_standings["Away"]
        
        if home_stand.matches > 5 and away_stand.matches > 5:
            home_gf_avg = home_stand.scored / home_stand.matches
            home_ga_avg = home_stand.conceded / home_stand.matches
            away_gf_avg = away_stand.scored / away_stand.matches
            away_ga_avg = away_stand.conceded / away_stand.matches
            
            standing_home_xg = (home_gf_avg + away_ga_avg) / 2
            standing_away_xg = (away_gf_avg + home_ga_avg) / 2
    
    # 2. H2H BİLEŞENİ (%30 Ağırlık)
    h2h_home_xg = h2h_away_xg = 0.0
    
    if h2h_matches and len(h2h_matches) >= 3:
        h2h_home_goals = []
        h2h_away_goals = []
        
        for match in h2h_matches[:6]:
            h2h_home_goals.append(match.ft_home)
            h2h_away_goals.append(match.ft_away)
        
        h2h_home_xg = sum(h2h_home_goals) / len(h2h_home_goals)
        h2h_away_xg = sum(h2h_away_goals) / len(h2h_away_goals)
    
    # 3. PREVIOUS SCORES BİLEŞENİ (%20 Ağırlık)
    prev_home_xg = home_stats.goals_for if home_stats.matches > 0 else 0
    prev_away_xg = away_stats.goals_for if away_stats.matches > 0 else 0
    
    # SABİT AĞIRLIKLARLA BİRLEŞTİR
    home_xg = (standing_home_xg * WEIGHT_STANDINGS) + (h2h_home_xg * WEIGHT_H2H) + (prev_home_xg * WEIGHT_PREVIOUS)
    away_xg = (standing_away_xg * WEIGHT_STANDINGS) + (h2h_away_xg * WEIGHT_H2H) + (prev_away_xg * WEIGHT_PREVIOUS)
    
    # Minimum ve maksimum sınırlar
    home_xg = max(0.2, min(3.5, home_xg))
    away_xg = max(0.2, min(3.0, away_xg))
    
    return round(home_xg, 2), round(away_xg, 2)

def calculate_corner_predictions(home_stats: TeamStats, away_stats: TeamStats,
                               h2h_matches: List[MatchRow]) -> Dict[str, Any]:
    """Calculate corner predictions"""
    
    home_corners = (home_stats.corners_for + away_stats.corners_against) / 2
    away_corners = (away_stats.corners_for + home_stats.corners_against) / 2
    
    if h2h_matches:
        h2h_home_corners = []
        h2h_away_corners = []
        
        for match in h2h_matches:
            if match.corner_home and match.corner_away:
                h2h_home_corners.append(match.corner_home)
                h2h_away_corners.append(match.corner_away)
        
        if h2h_home_corners:
            h2h_home_avg = sum(h2h_home_corners) / len(h2h_home_corners)
            home_corners = (home_corners * 0.6) + (h2h_home_avg * 0.4)
        
        if h2h_away_corners:
            h2h_away_avg = sum(h2h_away_corners) / len(h2h_away_corners)
            away_corners = (away_corners * 0.6) + (h2h_away_avg * 0.4)
    
    total_corners = home_corners + away_corners
    
    predictions = {}
    corner_lines = [8.5, 9.5, 10.5]
    
    for line in corner_lines:
        predictions[f"O{line}"] = "Evet" if total_corners > line else "Hayır"
        predictions[f"U{line}"] = "Hayır" if total_corners > line else "Evet"
    
    confidence = "Yüksek" if len(h2h_matches) >= 5 else "Orta" if len(h2h_matches) >= 3 else "Düşük"
    
    return {
        "home_corners": round(home_corners, 1),
        "away_corners": round(away_corners, 1),
        "total_corners": round(total_corners, 1),
        "predictions": predictions,
        "confidence": confidence
    }

# ======================
# PREDICTION ENGINE
# ======================
def poisson_probability(k: int, lam: float) -> float:
    """Calculate Poisson probability"""
    if lam <= 0:
        return 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except:
        return 0.0

def calculate_score_probabilities(home_xg: float, away_xg: float) -> Dict[str, Any]:
    """Calculate score probabilities using Poisson distribution"""
    max_goals = 4  # Render için daha az hesaplama
    score_probs = {}
    
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = poisson_probability(home_goals, home_xg) * poisson_probability(away_goals, away_xg)
            score_probs[f"{home_goals}-{away_goals}"] = round(prob * 100, 2)
    
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    
    home_win_prob = sum(prob for score, prob in score_probs.items() 
                       if int(score.split('-')[0]) > int(score.split('-')[1]))
    draw_prob = sum(prob for score, prob in score_probs.items() 
                   if int(score.split('-')[0]) == int(score.split('-')[1]))
    away_win_prob = sum(prob for score, prob in score_probs.items() 
                       if int(score.split('-')[0]) < int(score.split('-')[1]))
    
    over_25_prob = sum(prob for score, prob in score_probs.items() 
                      if sum(map(int, score.split('-'))) > 2.5)
    
    btts_prob = sum(prob for score, prob in score_probs.items() 
                   if int(score.split('-')[0]) > 0 and int(score.split('-')[1]) > 0)
    
    return {
        "expected_score": sorted_scores[0][0] if sorted_scores else "1-1",
        "top_scores": sorted_scores,
        "probabilities": {
            "home_win": round(home_win_prob, 1),
            "draw": round(draw_prob, 1),
            "away_win": round(away_win_prob, 1),
            "over_2.5": round(over_25_prob, 1),
            "under_2.5": round(100 - over_25_prob, 1),
            "btts": round(btts_prob, 1)
        }
    }

def calculate_value_bets(predicted_probs: Dict[str, float], odds: Dict[str, float]) -> List[Dict]:
    """Calculate value bets using Kelly Criterion"""
    value_bets = []
    
    for market, prob in predicted_probs.items():
        if market in odds and odds[market] > 0:
            decimal_prob = prob / 100
            odds_value = odds[market]
            
            value = (decimal_prob * odds_value) - 1
            
            if odds_value > 1:
                kelly = max(0, (decimal_prob * odds_value - 1) / (odds_value - 1))
            else:
                kelly = 0
            
            if value > 0.05 and decimal_prob > 0.55 and kelly > 0.02:
                recommendation = "GÜÇLÜ" if value > 0.15 else "ORTA" if value > 0.08 else "HAFİF"
                
                value_bets.append({
                    "market": market,
                    "probability": prob,
                    "odds": odds_value,
                    "value": round(value * 100, 1),
                    "kelly": round(kelly * 100, 1),
                    "recommendation": recommendation
                })
    
    return sorted(value_bets, key=lambda x: x["value"], reverse=True)

# ======================
# MAIN ANALYSIS FUNCTION
# ======================
def analyze_nowgoal_match(url: str) -> Dict[str, Any]:
    """Main analysis function for NowGoal matches"""
    
    # Cache kontrolü
    cache_key = hash(url)
    current_time = time.time()
    if cache_key in ANALYSIS_CACHE:
        cached_data, timestamp = ANALYSIS_CACHE[cache_key]
        if current_time - timestamp < CACHE_TIMEOUT:
            print(f"📦 Cache'ten yüklendi: {url}")
            return cached_data
    
    print(f"🔍 Analiz başlatılıyor: {url}")
    
    try:
        print("1. Sayfa yükleniyor...")
        html = safe_get_url(url)
        
        if not html:
            return {
                "success": False,
                "error": "Sayfa yüklenemedi",
                "match_info": {
                    "url": url,
                    "analysis_date": time.strftime("%d-%m-%Y %H:%M:%S")
                }
            }
        
        print("2. Takım isimleri çıkarılıyor...")
        home_team, away_team = extract_teams_from_html(html)
        print(f"   Takımlar: {home_team} vs {away_team}")
        
        print("3. Standing verileri çıkarılıyor...")
        home_standings = extract_standings_data(html, home_team)
        away_standings = extract_standings_data(html, away_team)
        
        print("4. H2H maçları çıkarılıyor...")
        h2h_matches = extract_h2h_matches(html)
        print(f"   {len(h2h_matches)} H2H maçı bulundu")
        
        print("5. Önceki maçlar çıkarılıyor...")
        home_previous = extract_previous_matches(html, "home")
        away_previous = extract_previous_matches(html, "away")
        
        print("6. Bet365 oranları çıkarılıyor...")
        odds = extract_bet365_odds(html)
        
        print("7. Takım istatistikleri hesaplanıyor...")
        home_stats = calculate_team_stats(home_previous, home_team)
        away_stats = calculate_team_stats(away_previous, away_team)
        
        print("8. Beklenen goller hesaplanıyor...")
        home_xg, away_xg = calculate_expected_goals(
            home_stats, away_stats, h2h_matches, 
            home_standings, away_standings
        )
        
        print("9. Skor olasılıkları hesaplanıyor...")
        score_predictions = calculate_score_probabilities(home_xg, away_xg)
        
        print("10. Korner tahminleri hesaplanıyor...")
        corner_predictions = calculate_corner_predictions(
            home_stats, away_stats, h2h_matches
        )
        
        print("11. Value bet'ler hesaplanıyor...")
        value_bets = []
        if odds:
            value_bets = calculate_value_bets(
                score_predictions["probabilities"],
                odds
            )
        
        print("12. Rapor oluşturuluyor...")
        report = {
            "success": True,
            "match_info": {
                "home_team": home_team,
                "away_team": away_team,
                "url": url,
                "analysis_date": time.strftime("%d-%m-%Y %H:%M:%S"),
                "weights_used": {
                    "standings": f"%{WEIGHT_STANDINGS*100:.0f}",
                    "h2h": f"%{WEIGHT_H2H*100:.0f}",
                    "previous": f"%{WEIGHT_PREVIOUS*100:.0f}"
                }
            },
            "data_summary": {
                "h2h_matches": len(h2h_matches),
                "home_previous_matches": len(home_previous),
                "away_previous_matches": len(away_previous),
                "odds_available": bool(odds)
            },
            "expected_goals": {
                "home": home_xg,
                "away": away_xg,
                "total": round(home_xg + away_xg, 2)
            },
            "predictions": {
                "main_score": score_predictions["expected_score"],
                "alternative_scores": score_predictions["top_scores"],
                "probabilities": score_predictions["probabilities"]
            },
            "corner_analysis": corner_predictions,
            "value_bets": value_bets
        }
        
        report["summary"] = generate_summary(report)
        
        # Cache'e kaydet
        ANALYSIS_CACHE[cache_key] = (report, time.time())
        
        print("✅ Analiz tamamlandı!")
        return report
        
    except Exception as e:
        error_msg = f"❌ Analiz hatası: {str(e)}"
        print(error_msg)
        return {
            "success": False,
            "error": str(e),
            "match_info": {
                "url": url,
                "analysis_date": time.strftime("%d-%m-%Y %H:%M:%S")
            }
        }

def generate_summary(report: Dict) -> str:
    """Generate human-readable summary"""
    pred = report["predictions"]
    corners = report["corner_analysis"]
    value_bets = report["value_bets"]
    data = report["data_summary"]
    
    lines = []
    lines.append("=" * 60)
    lines.append("📊 NOWGOAL MAÇ ANALİZ RAPORU")
    lines.append("=" * 60)
    
    lines.append(f"\n⚽ TAKIMLAR: {report['match_info']['home_team']} vs {report['match_info']['away_team']}")
    lines.append(f"📅 Analiz Tarihi: {report['match_info']['analysis_date']}")
    
    lines.append(f"\n⚖️  AĞIRLIKLAR: %50 Standing, %30 H2H, %20 Previous")
    
    lines.append(f"\n🎯 TAHMİN EDİLEN SKOR: {pred['main_score']}")
    
    lines.append(f"\n📈 OLASILIKLAR:")
    lines.append(f"   • Ev Kazanır: %{pred['probabilities']['home_win']:.1f}")
    lines.append(f"   • Beraberlik: %{pred['probabilities']['draw']:.1f}")
    lines.append(f"   • Deplasman Kazanır: %{pred['probabilities']['away_win']:.1f}")
    lines.append(f"   • 2.5 Üst: %{pred['probabilities']['over_2.5']:.1f}")
    lines.append(f"   • BTTS: %{pred['probabilities']['btts']:.1f}")
    
    lines.append(f"\n🔮 ALTERNATİF SKORLAR:")
    for i, (score, prob) in enumerate(pred['alternative_scores'][:3], 1):
        lines.append(f"   {i}. {score}: %{prob:.1f}")
    
    lines.append(f"\n⚽ KORNER TAHMİNLERİ:")
    lines.append(f"   • Ev Sahibi: {corners['home_corners']:.1f}")
    lines.append(f"   • Deplasman: {corners['away_corners']:.1f}")
    lines.append(f"   • Toplam: {corners['total_corners']:.1f}")
    lines.append(f"   • Güven: {corners['confidence']}")
    
    if value_bets:
        lines.append(f"\n💰 DEĞERLİ BAHİSLER:")
        for bet in value_bets[:2]:
            lines.append(f"   • {bet['market']}: Oran {bet['odds']:.2f}, "
                        f"Value %{bet['value']:+.1f} ({bet['recommendation']})")
    else:
        lines.append(f"\nℹ️  Değerli bahis bulunamadı")
    
    lines.append(f"\n" + "=" * 60)
    lines.append("✅ NET TAVSİYE:")
    
    home_prob = pred['probabilities']['home_win']
    draw_prob = pred['probabilities']['draw']
    away_prob = pred['probabilities']['away_win']
    
    if home_prob > 45 and home_prob > away_prob + 10:
        lines.append(f"EV SAHİBİ KAZANIR - {pred['main_score']}")
    elif away_prob > 45 and away_prob > home_prob + 10:
        lines.append(f"DEPLASMAN KAZANIR - {pred['main_score']}")
    elif draw_prob > 35:
        lines.append(f"BERABERLİK - {pred['main_score']}")
    else:
        lines.append(f"BELİRSİZ - En olası skor: {pred['main_score']}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

# ======================
# FLASK API
# ======================
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "service": "NowGoal Match Analyzer v5.0",
        "weights": {
            "standings": f"%{WEIGHT_STANDINGS*100:.0f}",
            "h2h": f"%{WEIGHT_H2H*100:.0f}",
            "previous": f"%{WEIGHT_PREVIOUS*100:.0f}"
        },
        "endpoints": {
            "/analyze": "POST - Maç analizi yap",
            "/health": "GET - Sağlık kontrolü"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "cache_size": len(ANALYSIS_CACHE)
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint for Render.com"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                "success": False,
                "error": "URL gereklidir",
                "example": {"url": "https://live3.nowgoal26.com/match/h2h-2784675"}
            }), 400
        
        url = data['url'].strip()
        if not url.startswith('http'):
            return jsonify({"success": False, "error": "Geçersiz URL formatı"}), 400
        
        print(f"📩 API İsteği: {url}")
        result = analyze_nowgoal_match(url)
        
        result["processing_time"] = round(time.time() - start_time, 2)
        
        if result.get("success"):
            return jsonify(result)
        else:
            return jsonify(result), 500
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "processing_time": round(time.time() - start_time, 2)
        }), 500

# ======================
# MAIN EXECUTION
# ======================
if __name__ == '__main__':
    print("=" * 60)
    print("NOWGOAL MAÇ ANALİZ SİSTEMİ v5.0")
    print("SABİT AĞIRLIK SİSTEMİ: %50 Standing, %30 H2H, %20 Previous")
    print("RENDER.COM OPTİMİZE EDİLMİŞ")
    print("=" * 60)
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
