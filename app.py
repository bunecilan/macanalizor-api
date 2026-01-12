# -*- coding: utf-8 -*-
"""
FUTBOL MAÇ ANALİZ API - RENDER DEPLOYMENT (FIXED)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import math
import requests
import time
import traceback
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

# Flask App
app = Flask(__name__)
CORS(app)

# ============================================================
# AYARLAR
# ============================================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
}

W_STANDING = 0.45
W_PREVIOUS = 0.30
W_H2H = 0.25

RECENT_MATCHES = 10
H2H_MATCHES = 10
MC_SIMULATIONS = 10000

VALUE_MIN = 0.05
PROB_MIN = 0.55

# ============================================================
# VERİ YAPILARI
# ============================================================
@dataclass
class MatchData:
    league: str
    date: str
    home: str
    away: str
    score_home: int
    score_away: int
    ht_home: Optional[int] = None
    ht_away: Optional[int] = None
    corner_home: Optional[int] = None
    corner_away: Optional[int] = None

@dataclass
class TeamStats:
    name: str
    goals_scored: float = 0.0
    goals_conceded: float = 0.0
    matches: int = 0
    goals_scored_home: float = 0.0
    goals_conceded_home: float = 0.0
    matches_home: int = 0
    goals_scored_away: float = 0.0
    goals_conceded_away: float = 0.0
    matches_away: int = 0
    corners_for: float = 0.0
    corners_against: float = 0.0
    clean_sheets: int = 0

# ============================================================
# HTML PARSE
# ============================================================
def strip_html_tags(text: str) -> str:
    text = re.sub(r'<script.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_tables(html: str) -> List[str]:
    return re.findall(r'<table.*?</table>', html, flags=re.DOTALL | re.IGNORECASE)

def parse_table(table_html: str) -> List[List[str]]:
    rows = []
    trs = re.findall(r'<tr.*?</tr>', table_html, flags=re.DOTALL | re.IGNORECASE)
    for tr in trs:
        cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', tr, flags=re.DOTALL | re.IGNORECASE)
        cleaned = [strip_html_tags(c) for c in cells if c]
        if cleaned:
            rows.append(cleaned)
    return rows

# ============================================================
# VERİ ÇEKME
# ============================================================
def get_page(url: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    except Exception as e:
        raise RuntimeError(f"Sayfa yüklenemedi: {str(e)}")

def extract_team_names(html: str) -> Tuple[str, str]:
    match = re.search(r'<title>\s*(.*?)\s*</title>', html, flags=re.IGNORECASE)
    if not match:
        return "", ""
    title = strip_html_tags(match.group(1))
    vs_match = re.search(r'(.+?)\s+(?:vs|VS)\s+(.+?)(?:\s+-|\||$)', title, flags=re.IGNORECASE)
    if vs_match:
        return vs_match.group(1).strip(), vs_match.group(2).strip()
    return "", ""

def extract_bet365_odds(html: str) -> Optional[Dict[str, float]]:
    try:
        tables = extract_tables(html)
        for table in tables:
            if 'bet365' not in table.lower():
                continue
            rows = parse_table(table)
            for row in rows:
                row_text = ' '.join(row).lower()
                if 'bet365' in row_text and 'initial' in row_text:
                    odds = []
                    for cell in row:
                        if re.match(r'^\d+\.\d+$', cell.strip()):
                            try:
                                odds.append(float(cell))
                            except:
                                pass
                    if len(odds) >= 3:
                        return {"1": odds[0], "X": odds[1], "2": odds[2]}
        return None
    except:
        return None

def extract_standings(html: str, team_name: str) -> Dict[str, Any]:
    team_key = team_name.lower().replace(' ', '')
    standings = {}
    tables = extract_tables(html)
    for table in tables:
        table_text = strip_html_tags(table).lower()
        if team_key not in table_text:
            continue
        if 'matches' not in table_text or 'scored' not in table_text:
            continue
        rows = parse_table(table)
        for row in rows:
            if len(row) < 7:
                continue
            row_type = row[0].strip()
            if row_type not in ['Total', 'Home', 'Away', 'Last 6']:
                continue
            try:
                standings[row_type] = {
                    'matches': int(row[1]) if len(row) > 1 and row[1].isdigit() else 0,
                    'scored': int(row[5]) if len(row) > 5 and row[5].isdigit() else 0,
                    'conceded': int(row[6]) if len(row) > 6 and row[6].isdigit() else 0,
                }
            except:
                pass
    return standings

def parse_match_row(cells: List[str]) -> Optional[MatchData]:
    if len(cells) < 5:
        return None
    score_pattern = r'(\d{1,2})\s*-\s*(\d{1,2})(?:\s*\((\d{1,2})\s*-\s*(\d{1,2})\))?'
    score_idx = None
    score_match = None
    for i, cell in enumerate(cells):
        match = re.search(score_pattern, cell)
        if match:
            score_idx = i
            score_match = match
            break
    if not score_match or score_idx is None or score_idx == 0 or score_idx >= len(cells) - 1:
        return None
    home = cells[score_idx - 1].strip()
    away = cells[score_idx + 1].strip()
    if not home or not away:
        return None
    ft_home = int(score_match.group(1))
    ft_away = int(score_match.group(2))
    corner_home, corner_away = None, None
    for i in range(score_idx + 2, len(cells)):
        corner_match = re.search(r'(\d{1,2})\s*-\s*(\d{1,2})', cells[i])
        if corner_match:
            corner_home = int(corner_match.group(1))
            corner_away = int(corner_match.group(2))
            break
    league = cells[0].strip() if cells else ""
    date = ""
    return MatchData(
        league=league, date=date, home=home, away=away,
        score_home=ft_home, score_away=ft_away,
        corner_home=corner_home, corner_away=corner_away
    )

def extract_h2h_matches(html: str, home_team: str, away_team: str) -> List[MatchData]:
    home_key = home_team.lower().replace(' ', '')
    away_key = away_team.lower().replace(' ', '')
    matches = []
    tables = extract_tables(html)
    for table in tables:
        rows = parse_table(table)
        for row in rows:
            match = parse_match_row(row)
            if not match:
                continue
            match_home_key = match.home.lower().replace(' ', '')
            match_away_key = match.away.lower().replace(' ', '')
            is_h2h = (match_home_key == home_key and match_away_key == away_key) or \
                     (match_home_key == away_key and match_away_key == home_key)
            if is_h2h:
                matches.append(match)
    return matches[:H2H_MATCHES]

def extract_previous_matches(html: str, team_name: str, is_home: bool) -> List[MatchData]:
    team_key = team_name.lower().replace(' ', '')
    matches = []
    tables = extract_tables(html)
    for table in tables:
        rows = parse_table(table)
        for row in rows:
            match = parse_match_row(row)
            if not match:
                continue
            match_home_key = match.home.lower().replace(' ', '')
            match_away_key = match.away.lower().replace(' ', '')
            is_team_match = (match_home_key == team_key) or (match_away_key == team_key)
            if not is_team_match:
                continue
            if is_home and match_home_key != team_key:
                continue
            if not is_home and match_away_key != team_key:
                continue
            matches.append(match)
    return matches[:RECENT_MATCHES]

# ============================================================
# İSTATİSTİK
# ============================================================
def calculate_team_stats(team_name: str, matches: List[MatchData]) -> TeamStats:
    team_key = team_name.lower().replace(' ', '')
    stats = TeamStats(name=team_name)
    if not matches:
        return stats
    goals_scored = []
    goals_conceded = []
    corners_for = []
    corners_against = []
    for match in matches:
        match_home_key = match.home.lower().replace(' ', '')
        if match_home_key == team_key:
            goals_scored.append(match.score_home)
            goals_conceded.append(match.score_away)
            if match.corner_home:
                corners_for.append(match.corner_home)
            if match.corner_away:
                corners_against.append(match.corner_away)
        else:
            goals_scored.append(match.score_away)
            goals_conceded.append(match.score_home)
            if match.corner_away:
                corners_for.append(match.corner_away)
            if match.corner_home:
                corners_against.append(match.corner_home)
    stats.matches = len(matches)
    stats.goals_scored = sum(goals_scored) / len(goals_scored) if goals_scored else 0.0
    stats.goals_conceded = sum(goals_conceded) / len(goals_conceded) if goals_conceded else 0.0
    stats.corners_for = sum(corners_for) / len(corners_for) if corners_for else 0.0
    stats.corners_against = sum(corners_against) / len(corners_against) if corners_against else 0.0
    return stats

# ============================================================
# LAMBDA HESAPLAMA
# ============================================================
def calculate_lambda(home_stats: TeamStats, away_stats: TeamStats,
                     home_standing: Dict, away_standing: Dict,
                     h2h_matches: List[MatchData]) -> Tuple[float, float]:
    lambda_home_st = 0.0
    lambda_away_st = 0.0
    if home_standing.get('Home') and away_standing.get('Away'):
        h_st = home_standing['Home']
        a_st = away_standing['Away']
        if h_st['matches'] > 0:
            lambda_home_st = h_st['scored'] / h_st['matches']
        if a_st['matches'] > 0:
            lambda_away_st = a_st['scored'] / a_st['matches']

    lambda_home_prev = home_stats.goals_scored
    lambda_away_prev = away_stats.goals_scored

    lambda_home_h2h = 0.0
    lambda_away_h2h = 0.0
    if h2h_matches:
        home_team_key = home_stats.name.lower().replace(' ', '')
        h2h_home_goals = []
        h2h_away_goals = []
        for match in h2h_matches:
            match_home_key = match.home.lower().replace(' ', '')
            if match_home_key == home_team_key:
                h2h_home_goals.append(match.score_home)
                h2h_away_goals.append(match.score_away)
            else:
                h2h_home_goals.append(match.score_away)
                h2h_away_goals.append(match.score_home)
        if h2h_home_goals:
            lambda_home_h2h = sum(h2h_home_goals) / len(h2h_home_goals)
        if h2h_away_goals:
            lambda_away_h2h = sum(h2h_away_goals) / len(h2h_away_goals)

    total_weight = 0.0
    lambda_home_final = 0.0
    if lambda_home_st > 0:
        lambda_home_final += lambda_home_st * W_STANDING
        total_weight += W_STANDING
    if lambda_home_prev > 0:
        lambda_home_final += lambda_home_prev * W_PREVIOUS
        total_weight += W_PREVIOUS
    if lambda_home_h2h > 0:
        lambda_home_final += lambda_home_h2h * W_H2H
        total_weight += W_H2H
    lambda_home_final = lambda_home_final / total_weight if total_weight > 0 else 1.0

    total_weight = 0.0
    lambda_away_final = 0.0
    if lambda_away_st > 0:
        lambda_away_final += lambda_away_st * W_STANDING
        total_weight += W_STANDING
    if lambda_away_prev > 0:
        lambda_away_final += lambda_away_prev * W_PREVIOUS
        total_weight += W_PREVIOUS
    if lambda_away_h2h > 0:
        lambda_away_final += lambda_away_h2h * W_H2H
        total_weight += W_H2H
    lambda_away_final = lambda_away_final / total_weight if total_weight > 0 else 1.0

    return lambda_home_final, lambda_away_final

# ============================================================
# POİSSON & MONTE CARLO
# ============================================================
def poisson_prob(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def create_score_matrix(lambda_home: float, lambda_away: float) -> np.ndarray:
    matrix = np.zeros((6, 6))
    for h in range(6):
        for a in range(6):
            matrix[h, a] = poisson_prob(h, lambda_home) * poisson_prob(a, lambda_away)
    return matrix

def calculate_market_probabilities(score_matrix: np.ndarray) -> Dict[str, float]:
    probs = {}
    probs['1'] = float(np.sum(np.tril(score_matrix, -1)))
    probs['X'] = float(np.sum(np.diag(score_matrix)))
    probs['2'] = float(np.sum(np.triu(score_matrix, 1)))
    over25 = 0.0
    for h in range(6):
        for a in range(6):
            if h + a > 2.5:
                over25 += score_matrix[h, a]
    probs['O2.5'] = float(over25)
    probs['U2.5'] = float(1.0 - over25)
    btts_yes = 0.0
    for h in range(1, 6):
        for a in range(1, 6):
            btts_yes += score_matrix[h, a]
    probs['BTTS_Yes'] = float(btts_yes)
    probs['BTTS_No'] = float(1.0 - btts_yes)
    return probs

def get_top_scores(score_matrix: np.ndarray) -> List[Tuple[str, float]]:
    scores = []
    for h in range(6):
        for a in range(6):
            scores.append((f'{h}-{a}', float(score_matrix[h, a])))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:7]

def analyze_corners(home_stats: TeamStats, away_stats: TeamStats, h2h_matches: List[MatchData]) -> Dict:
    h2h_corners = []
    for match in h2h_matches:
        if match.corner_home and match.corner_away:
            h2h_corners.append(match.corner_home + match.corner_away)
    h2h_avg = sum(h2h_corners) / len(h2h_corners) if h2h_corners else 0.0
    predicted_home = ((home_stats.corners_for + away_stats.corners_against) / 2) if (home_stats.corners_for + away_stats.corners_against) > 0 else 0.0
    predicted_away = ((away_stats.corners_for + home_stats.corners_against) / 2) if (away_stats.corners_for + home_stats.corners_against) > 0 else 0.0
    total_corners = predicted_home + predicted_away
    return {
        'home': round(predicted_home, 1),
        'away': round(predicted_away, 1),
        'total': round(total_corners, 1),
        'h2h_avg': round(h2h_avg, 1),
        'confidence': 'Yüksek' if len(h2h_corners) >= 5 else 'Orta' if len(h2h_corners) >= 3 else 'Düşük'
    }

def analyze_value_bets(probs: Dict[str, float], odds: Optional[Dict[str, float]]) -> Dict:
    if not odds:
        return {'decision': 'Oran verisi yok', 'bets': []}
    bets = []
    for market in ['1', 'X', '2']:
        if market in probs and market in odds:
            prob = probs[market]
            odd = odds[market]
            value = (odd * prob) - 1
            kelly = ((odd * prob) - 1) / (odd - 1) if odd > 1 else 0
            bets.append({
                'market': market,
                'prob': round(prob, 4),
                'odd': round(odd, 2),
                'value': round(value, 4),
                'playable': value >= VALUE_MIN and prob >= PROB_MIN
            })
    playable_bets = [b for b in bets if b['playable']]
    if playable_bets:
        best = max(playable_bets, key=lambda x: x['value'])
        market_name = {'1': 'Ev', 'X': 'Beraberlik', '2': 'Deplasman'}[best['market']]
        decision = f"OYNA: {market_name} - Value: {best['value']*100:+.1f}%"
    else:
        decision = "OYNAMA - Değerli bahis yok"
    return {'decision': decision, 'bets': bets}

# ============================================================
# ANA ANALİZ
# ============================================================
def analyze_match(url: str) -> Dict[str, Any]:
    try:
        html = get_page(url)
        home_team, away_team = extract_team_names(html)
        if not home_team or not away_team:
            return {"error": "Takım isimleri bulunamadı", "success": False}

        league_match = re.search(r'Italian Serie A|Premier League|La Liga|Bundesliga|Ligue 1', html, re.IGNORECASE)
        league_name = league_match.group(0) if league_match else ""

        odds = extract_bet365_odds(html)
        home_standing = extract_standings(html, home_team)
        away_standing = extract_standings(html, away_team)
        h2h_matches = extract_h2h_matches(html, home_team, away_team)
        home_prev = extract_previous_matches(html, home_team, is_home=True)
        away_prev = extract_previous_matches(html, away_team, is_home=False)

        home_stats = calculate_team_stats(home_team, home_prev)
        away_stats = calculate_team_stats(away_team, away_prev)

        lambda_home, lambda_away = calculate_lambda(
            home_stats, away_stats, home_standing, away_standing, h2h_matches
        )

        score_matrix = create_score_matrix(lambda_home, lambda_away)
        poisson_probs = calculate_market_probabilities(score_matrix)
        top_scores = get_top_scores(score_matrix)
        corner_analysis = analyze_corners(home_stats, away_stats, h2h_matches)
        value_analysis = analyze_value_bets(poisson_probs, odds)

        return {
            "success": True,
            "match_info": {
                "home_team": home_team,
                "away_team": away_team,
                "league": league_name
            },
            "expected_goals": {
                "home": round(lambda_home, 2),
                "away": round(lambda_away, 2),
                "total": round(lambda_home + lambda_away, 2)
            },
            "top_scores": [
                {"score": score, "probability": round(prob * 100, 1)}
                for score, prob in top_scores[:5]
            ],
            "predictions": {
                "main_score": top_scores[0][0],
                "alt_scores": [top_scores[1][0], top_scores[2][0]],
                "over_under_2_5": {
                    "prediction": "ÜST" if poisson_probs['O2.5'] > poisson_probs['U2.5'] else "ALT",
                    "over_prob": round(poisson_probs['O2.5'] * 100, 1),
                    "under_prob": round(poisson_probs['U2.5'] * 100, 1)
                },
                "btts": {
                    "prediction": "VAR" if poisson_probs['BTTS_Yes'] > poisson_probs['BTTS_No'] else "YOK",
                    "yes_prob": round(poisson_probs['BTTS_Yes'] * 100, 1),
                    "no_prob": round(poisson_probs['BTTS_No'] * 100, 1)
                },
                "match_result": {
                    "home_win": round(poisson_probs['1'] * 100, 1),
                    "draw": round(poisson_probs['X'] * 100, 1),
                    "away_win": round(poisson_probs['2'] * 100, 1)
                }
            },
            "corners": corner_analysis,
            "value_bets": {
                "decision": value_analysis['decision'],
                "odds": odds,
                "analysis": value_analysis['bets']
            },
            "data_sources": {
                "standings": bool(home_standing and away_standing),
                "previous_home": len(home_prev),
                "previous_away": len(away_prev),
                "h2h": len(h2h_matches)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ============================================================
# FLASK ROUTES
# ============================================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "service": "Futbol Maç Analiz API",
        "version": "1.1-fixed",
        "endpoints": {
            "analyze": {
                "method": "POST",
                "url": "/analyze",
                "body": {"url": "https://live3.nowgoal26.com/match/h2h-XXXXXX"}
            }
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                "success": False,
                "error": "URL gerekli"
            }), 400

        url = data['url']
        result = analyze_match(url)

        if result.get('success'):
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ============================================================
# RENDER START
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
