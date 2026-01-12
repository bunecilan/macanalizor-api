# -*- coding: utf-8 -*-
"""
FUTBOL MAÇ ANALİZ API - RENDER DEPLOYMENT
Endpoint: POST /analyze
Body: {"url": "https://live3.nowgoal26.com/match/h2h-2784675"}
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import math
import requests
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

app = Flask(__name__)
CORS(app)  # Tüm originlere izin ver

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
KELLY_MIN = 0.02

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
        response = requests.get(url, headers=HEADERS, timeout=25)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    except Exception as e:
        raise RuntimeError(f"Sayfa yüklenemedi: {url} - {e}")

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
                    'win': int(row[2]) if len(row) > 2 and row[2].isdigit() else 0,
                    'draw': int(row[3]) if len(row) > 3 and row[3].isdigit() else 0,
                    'loss': int(row[4]) if len(row) > 4 and row[4].isdigit() else 0,
                    'scored': int(row[5]) if len(row) > 5 and row[5].isdigit() else 0,
                    'conceded': int(row[6]) if len(row) > 6 and row[6].isdigit() else 0,
                    'pts': int(row[7]) if len(row) > 7 and row[7].isdigit() else 0,
                    'rank': int(row[8]) if len(row) > 8 and row[8].isdigit() else 0,
                    'rate': row[9] if len(row) > 9 else ''
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
    if not score_match or score_idx is None:
        return None
    if score_idx == 0 or score_idx >= len(cells) - 1:
        return None
    home = cells[score_idx - 1].strip()
    away = cells[score_idx + 1].strip()
    if not home or not away:
        return None
    ft_home = int(score_match.group(1))
    ft_away = int(score_match.group(2))
    ht_home = int(score_match.group(3)) if score_match.group(3) else None
    ht_away = int(score_match.group(4)) if score_match.group(4) else None
    corner_home, corner_away = None, None
    for i in range(score_idx + 2, len(cells)):
        corner_match = re.search(r'(\d{1,2})\s*-\s*(\d{1,2})', cells[i])
        if corner_match:
            corner_home = int(corner_match.group(1))
            corner_away = int(corner_match.group(2))
            break
    date = ""
    date_match = re.search(r'(\d{1,2}-\d{1,2}-\d{4})', ' '.join(cells))
    if date_match:
        date = date_match.group(1)
    league = cells[0].strip() if cells else ""
    return MatchData(
        league=league, date=date, home=home, away=away,
        score_home=ft_home, score_away=ft_away,
        ht_home=ht_home, ht_away=ht_away,
        corner_home=corner_home, corner_away=corner_away
    )

def extract_h2h_matches(html: str, home_team: str, away_team: str) -> List[MatchData]:
    home_key = home_team.lower().replace(' ', '')
    away_key = away_team.lower().replace(' ', '')
    h2h_markers = ['head to head statistics', 'head to head', 'h2h statistics']
    h2h_section = ""
    html_lower = html.lower()
    for marker in h2h_markers:
        pos = html_lower.find(marker)
        if pos != -1:
            h2h_section = html[pos:pos+50000]
            break
    if not h2h_section:
        h2h_section = html
    matches = []
    tables = extract_tables(h2h_section)
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

def extract_previous_matches(html: str, team_name: str, league_name: str, is_home: bool) -> List[MatchData]:
    team_key = team_name.lower().replace(' ', '')
    league_key = league_name.lower().replace(' ', '') if league_name else ''
    prev_markers = ['previous scores statistics', 'previous scores', 'recent matches']
    prev_section = ""
    html_lower = html.lower()
    for marker in prev_markers:
        pos = html_lower.find(marker)
        if pos != -1:
            prev_section = html[pos:pos+50000]
            break
    if not prev_section:
        prev_section = html
    matches = []
    tables = extract_tables(prev_section)
    for table in tables:
        table_text = strip_html_tags(table).lower()
        if team_key not in table_text:
            continue
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
            if league_key:
                match_league_key = match.league.lower().replace(' ', '')
                if league_key not in match_league_key:
                    continue
            matches.append(match)
    return matches[:RECENT_MATCHES]

# ============================================================
# İSTATİSTİK
# ============================================================
def calculate_team_stats(team_name: str, matches: List[MatchData], is_home: bool) -> TeamStats:
    team_key = team_name.lower().replace(' ', '')
    stats = TeamStats(name=team_name)
    if not matches:
        return stats
    goals_scored = []
    goals_conceded = []
    corners_for = []
    corners_against = []
    clean_sheets = 0
    home_matches = []
    away_matches = []
    for match in matches:
        match_home_key = match.home.lower().replace(' ', '')
        match_away_key = match.away.lower().replace(' ', '')
        if match_home_key == team_key:
            goals_scored.append(match.score_home)
            goals_conceded.append(match.score_away)
            if match.corner_home:
                corners_for.append(match.corner_home)
            if match.corner_away:
                corners_against.append(match.corner_away)
            if match.score_away == 0:
                clean_sheets += 1
            home_matches.append(match)
        elif match_away_key == team_key:
            goals_scored.append(match.score_away)
            goals_conceded.append(match.score_home)
            if match.corner_away:
                corners_for.append(match.corner_away)
            if match.corner_home:
                corners_against.append(match.corner_home)
            if match.score_home == 0:
                clean_sheets += 1
            away_matches.append(match)
    stats.matches = len(matches)
    stats.goals_scored = sum(goals_scored) / len(goals_scored) if goals_scored else 0.0
    stats.goals_conceded = sum(goals_conceded) / len(goals_conceded) if goals_conceded else 0.0
    stats.corners_for = sum(corners_for) / len(corners_for) if corners_for else 0.0
    stats.corners_against = sum(corners_against) / len(corners_against) if corners_against else 0.0
    stats.clean_sheets = clean_sheets
    if home_matches:
        stats.matches_home = len(home_matches)
        home_scored = [m.score_home for m in home_matches]
        home_conceded = [m.score_away for m in home_matches]
        stats.goals_scored_home = sum(home_scored) / len(home_scored)
        stats.goals_conceded_home = sum(home_conceded) / len(home_conceded)
    if away_matches:
        stats.matches_away = len(away_matches)
        away_scored = [m.score_away for m in away_matches]
        away_conceded = [m.score_home for m in away_matches]
        stats.goals_scored_away = sum(away_scored) / len(away_scored)
        stats.goals_conceded_away = sum(away_conceded) / len(away_conceded)
    return stats

# ============================================================
# LAMBDA HESAPLAMA
# ============================================================
def calculate_lambda(home_stats: TeamStats, away_stats: TeamStats,
                     home_standing: Dict, away_standing: Dict,
                     h2h_matches: List[MatchData]) -> Tuple[float, float, Dict]:
    info = {'methods': {}, 'weights': {}, 'final': {}}
    lambda_home_st = 0.0
    lambda_away_st = 0.0
    if home_standing.get('Home') and away_standing.get('Away'):
        h_st = home_standing['Home']
        a_st = away_standing['Away']
        if h_st['matches'] > 0:
            lambda_home_st = h_st['scored'] / h_st['matches']
        if a_st['matches'] > 0:
            lambda_away_st = a_st['scored'] / a_st['matches']
        info['methods']['standing'] = {
            'home': lambda_home_st,
            'away': lambda_away_st,
            'weight': W_STANDING
        }
    lambda_home_prev = home_stats.goals_scored_home if home_stats.matches_home > 0 else home_stats.goals_scored
    lambda_away_prev = away_stats.goals_scored_away if away_stats.matches_away > 0 else away_stats.goals_scored
    info['methods']['previous'] = {
        'home': lambda_home_prev,
        'away': lambda_away_prev,
        'weight': W_PREVIOUS
    }
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
        info['methods']['h2h'] = {
            'home': lambda_home_h2h,
            'away': lambda_away_h2h,
            'weight': W_H2H
        }
    total_weight = 0.0
    lambda_home_final = 0.0
    lambda_away_final = 0.0
    if lambda_home_st > 0:
        lambda_home_final += lambda_home_st * W_STANDING
        total_weight += W_STANDING
    if lambda_home_prev > 0:
        lambda_home_final += lambda_home_prev * W_PREVIOUS
        if lambda_home_st == 0:
            total_weight += W_STANDING + W_PREVIOUS
        else:
            total_weight += W_PREVIOUS
    if lambda_home_h2h > 0:
        lambda_home_final += lambda_home_h2h * W_H2H
        total_weight += W_H2H
    if total_weight > 0:
        lambda_home_final /= total_weight
    else:
        lambda_home_final = 1.0
    total_weight = 0.0
    if lambda_away_st > 0:
        lambda_away_final += lambda_away_st * W_STANDING
        total_weight += W_STANDING
    if lambda_away_prev > 0:
        lambda_away_final += lambda_away_prev * W_PREVIOUS
        if lambda_away_st == 0:
            total_weight += W_STANDING + W_PREVIOUS
        else:
            total_weight += W_PREVIOUS
    if lambda_away_h2h > 0:
        lambda_away_final += lambda_away_h2h * W_H2H
        total_weight += W_H2H
    if total_weight > 0:
        lambda_away_final /= total_weight
    else:
        lambda_away_final = 1.0
    info['final'] = {
        'lambda_home': lambda_home_final,
        'lambda_away': lambda_away_final
    }
    return lambda_home_final, lambda_away_final, info

# ============================================================
# POİSSON
# ============================================================
def poisson_prob(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def create_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 5) -> np.ndarray:
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[h, a] = poisson_prob(h, lambda_home) * poisson_prob(a, lambda_away)
    return matrix

def calculate_market_probabilities(score_matrix: np.ndarray) -> Dict[str, float]:
    probs = {}
    probs['1'] = np.sum(np.tril(score_matrix, -1))
    probs['X'] = np.sum(np.diag(score_matrix))
    probs['2'] = np.sum(np.triu(score_matrix, 1))
    for line in [0.5, 1.5, 2.5, 3.5]:
        over = 0.0
        for h in range(score_matrix.shape[0]):
            for a in range(score_matrix.shape[1]):
                if h + a > line:
                    over += score_matrix[h, a]
        probs[f'O{line}'] = over
        probs[f'U{line}'] = 1.0 - over
    btts_yes = 0.0
    for h in range(1, score_matrix.shape[0]):
        for a in range(1, score_matrix.shape[1]):
            btts_yes += score_matrix[h, a]
    probs['BTTS_Yes'] = btts_yes
    probs['BTTS_No'] = 1.0 - btts_yes
    return probs

def get_top_scores(score_matrix: np.ndarray, top_n: int = 7) -> List[Tuple[str, float]]:
    scores = []
    for h in range(score_matrix.shape[0]):
        for a in range(score_matrix.shape[1]):
            scores.append((f'{h}-{a}', score_matrix[h, a]))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# ============================================================
# MONTE CARLO
# ============================================================
def monte_carlo_simulation(lambda_home: float, lambda_away: float, n_sims: int = 10000) -> Dict:
    np.random.seed(42)
    home_goals = np.random.poisson(lambda_home, n_sims)
    away_goals = np.random.poisson(lambda_away, n_sims)
    results = {
        '1': float(np.sum(home_goals > away_goals) / n_sims),
        'X': float(np.sum(home_goals == away_goals) / n_sims),
        '2': float(np.sum(home_goals < away_goals) / n_sims),
        'O2.5': float(np.sum((home_goals + away_goals) > 2.5) / n_sims),
        'U2.5': float(np.sum((home_goals + away_goals) <= 2.5) / n_sims),
        'BTTS_Yes': float(np.sum((home_goals > 0) & (away_goals > 0)) / n_sims),
        'BTTS_No': float(np.sum((home_goals == 0) | (away_goals == 0)) / n_sims),
    }
    return results

# ============================================================
# KORNER
# ============================================================
def analyze_corners(home_stats: TeamStats, away_stats: TeamStats, h2h_matches: List[MatchData]) -> Dict:
    h2h_corners = []
    for match in h2h_matches:
        if match.corner_home and match.corner_away:
            h2h_corners.append(match.corner_home + match.corner_away)
    h2h_avg = sum(h2h_corners) / len(h2h_corners) if h2h_corners else 0.0
    home_for = home_stats.corners_for
    home_against = home_stats.corners_against
    away_for = away_stats.corners_for
    away_against = away_stats.corners_against
    if h2h_avg > 0:
        predicted_home = 0.6 * (h2h_avg / 2) + 0.4 * ((home_for + away_against) / 2)
        predicted_away = 0.6 * (h2h_avg / 2) + 0.4 * ((away_for + home_against) / 2)
    else:
        predicted_home = (home_for + away_against) / 2 if (home_for + away_against) > 0 else 0.0
        predicted_away = (away_for + home_against) / 2 if (away_for + home_against) > 0 else 0.0
    total_corners = predicted_home + predicted_away
    return {
        'home': round(predicted_home, 1),
        'away': round(predicted_away, 1),
        'total': round(total_corners, 1),
        'h2h_avg': round(h2h_avg, 1),
        'confidence': 'Yüksek' if len(h2h_corners) >= 5 else 'Orta' if len(h2h_corners) >= 3 else 'Düşük'
    }

# ============================================================
# VALUE BET
# ============================================================
def analyze_value_bets(probs: Dict[str, float], odds: Dict[str, float]) -> Dict:
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
                'kelly': round(kelly, 4),
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
        home_prev = extract_previous_matches(html, home_team, league_name, is_home=True)
        away_prev = extract_previous_matches(html, away_team, league_name, is_home=False)

        home_stats = calculate_team_stats(home_team, home_prev, is_home=True)
        away_stats = calculate_team_stats(away_team, away_prev, is_home=False)

        lambda_home, lambda_away, lambda_info = calculate_lambda(
            home_stats, away_stats, home_standing, away_standing, h2h_matches
        )

        score_matrix = create_score_matrix(lambda_home, lambda_away)
        poisson_probs = calculate_market_probabilities(score_matrix)
        top_scores = get_top_scores(score_matrix)
        mc_probs = monte_carlo_simulation(lambda_home, lambda_away, MC_SIMULATIONS)
        corner_analysis = analyze_corners(home_stats, away_stats, h2h_matches)
        value_analysis = analyze_value_bets(poisson_probs, odds)

        # JSON response
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
            },
            "weights": {
                "standing": f"{W_STANDING*100:.0f}%",
                "previous": f"{W_PREVIOUS*100:.0f}%",
                "h2h": f"{W_H2H*100:.0f}%"
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
        "version": "1.0",
        "endpoints": {
            "analyze": {
                "method": "POST",
                "url": "/analyze",
                "body": {"url": "https://live3.nowgoal26.com/match/h2h-XXXXXX"}
            },
            "health": {
                "method": "GET",
                "url": "/health"
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
                "error": "URL gerekli. Body: {\"url\": \"https://live3.nowgoal26.com/match/h2h-XXXXXX\"}"
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
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# ============================================================
# RENDER DEPLOYMENT
# ============================================================
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
