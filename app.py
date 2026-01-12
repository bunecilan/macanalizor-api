# -*- coding: utf-8 -*-
"""
FUTBOL MAÇ ANALİZ PROGRAMI
NowGoal sitesinden veri çekerek olasılık modelleme ve bahis analizi yapar
"""

import re
import math
import requests
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

# ============================================================
# AYARLAR
# ============================================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
}

# Veri kaynağı ağırlıkları (EN ÖNEMLİDEN EN AZ ÖNEMLİYE)
W_STANDING = 0.45    # Lig sıralaması ve istatistikleri
W_PREVIOUS = 0.30    # Son maç performansları (Same League)
W_H2H = 0.25         # İki takımın geçmiş karşılaşmaları

# Analiz ayarları
RECENT_MATCHES = 10   # Son kaç maç analiz edilsin
H2H_MATCHES = 10      # Kaç H2H maç analiz edilsin
MC_SIMULATIONS = 10000  # Monte Carlo simülasyon sayısı

# Bahis eşikleri
VALUE_MIN = 0.05     # Minimum %5 değer
PROB_MIN = 0.55      # Minimum %55 olasılık
KELLY_MIN = 0.02     # Minimum %2 Kelly

# ============================================================
# VERİ YAPILARI
# ============================================================
@dataclass
class MatchData:
    """Bir maçın verileri"""
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
    """Bir takımın istatistikleri"""
    name: str
    # Gol istatistikleri
    goals_scored: float = 0.0
    goals_conceded: float = 0.0
    matches: int = 0
    # İç/Dış saha
    goals_scored_home: float = 0.0
    goals_conceded_home: float = 0.0
    matches_home: int = 0
    goals_scored_away: float = 0.0
    goals_conceded_away: float = 0.0
    matches_away: int = 0
    # Korner
    corners_for: float = 0.0
    corners_against: float = 0.0
    # Temiz çarşaf
    clean_sheets: int = 0

# ============================================================
# HTML PARSE FONKSİYONLARI
# ============================================================
def strip_html_tags(text: str) -> str:
    """HTML taglerini temizle"""
    text = re.sub(r'<script.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_tables(html: str) -> List[str]:
    """HTML'den tüm tabloları çıkar"""
    return re.findall(r'<table.*?</table>', html, flags=re.DOTALL | re.IGNORECASE)

def parse_table(table_html: str) -> List[List[str]]:
    """Bir tabloyu satır ve hücrelere ayır"""
    rows = []
    trs = re.findall(r'<tr.*?</tr>', table_html, flags=re.DOTALL | re.IGNORECASE)
    for tr in trs:
        cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', tr, flags=re.DOTALL | re.IGNORECASE)
        cleaned = [strip_html_tags(c) for c in cells if c]
        if cleaned:
            rows.append(cleaned)
    return rows

# ============================================================
# VERİ ÇEKME FONKSİYONLARI
# ============================================================
def get_page(url: str) -> str:
    """URL'den sayfa içeriğini çek"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=25)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    except Exception as e:
        raise RuntimeError(f"Sayfa yüklenemedi: {url} - {e}")

def extract_team_names(html: str) -> Tuple[str, str]:
    """Sayfa başlığından takım isimlerini çıkar"""
    match = re.search(r'<title>\s*(.*?)\s*</title>', html, flags=re.IGNORECASE)
    if not match:
        return "", ""
    title = strip_html_tags(match.group(1))

    # "Genoa vs Cagliari" formatı
    vs_match = re.search(r'(.+?)\s+(?:vs|VS)\s+(.+?)(?:\s+-|\||$)', title, flags=re.IGNORECASE)
    if vs_match:
        return vs_match.group(1).strip(), vs_match.group(2).strip()
    return "", ""

def extract_bet365_odds(html: str) -> Optional[Dict[str, float]]:
    """
    Bet365 Initial 1X2 oranlarını çıkar
    PDF'te işaretli alan: Live Odds Comparison > Bet365 > Initial
    """
    try:
        # Bet365 tablosunu bul
        tables = extract_tables(html)
        for table in tables:
            if 'bet365' not in table.lower():
                continue

            rows = parse_table(table)
            for row in rows:
                # "Initial" satırını ara
                row_text = ' '.join(row).lower()
                if 'bet365' in row_text and 'initial' in row_text:
                    # Oran değerlerini çıkar (1X2 formatında)
                    odds = []
                    for cell in row:
                        if re.match(r'^\d+\.\d+$', cell.strip()):
                            try:
                                odds.append(float(cell))
                            except:
                                pass

                    # En az 3 oran bulmalıyız (1-X-2)
                    if len(odds) >= 3:
                        return {"1": odds[0], "X": odds[1], "2": odds[2]}

        return None
    except Exception as e:
        print(f"⚠️  Oran çıkarma hatası: {e}")
        return None

def extract_standings(html: str, team_name: str) -> Dict[str, Any]:
    """
    STANDING verilerini çıkar
    PDF'te işaretli alan: Standings tablosu
    Çıkarılacak: FT Matches Win Draw Loss Scored Conceded Pts Rank Rate
    """
    team_key = team_name.lower().replace(' ', '')
    standings = {}

    tables = extract_tables(html)
    for table in tables:
        table_text = strip_html_tags(table).lower()

        # Bu takımın standing tablosu mu?
        if team_key not in table_text:
            continue
        if 'matches' not in table_text or 'scored' not in table_text:
            continue

        rows = parse_table(table)
        for row in rows:
            if len(row) < 7:
                continue

            # Satır tipi: Total, Home, Away, Last 6
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
    """Tablo satırından maç verisini çıkar"""
    if len(cells) < 5:
        return None

    # Skor formatı: "2-1" veya "2-1(1-0)"
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

    # Takım isimleri skor''un iki yanında
    if score_idx == 0 or score_idx >= len(cells) - 1:
        return None

    home = cells[score_idx - 1].strip()
    away = cells[score_idx + 1].strip()

    if not home or not away:
        return None

    # Skorlar
    ft_home = int(score_match.group(1))
    ft_away = int(score_match.group(2))
    ht_home = int(score_match.group(3)) if score_match.group(3) else None
    ht_away = int(score_match.group(4)) if score_match.group(4) else None

    # Korner (genelde skordan sonra)
    corner_home, corner_away = None, None
    for i in range(score_idx + 2, len(cells)):
        corner_match = re.search(r'(\d{1,2})\s*-\s*(\d{1,2})', cells[i])
        if corner_match:
            corner_home = int(corner_match.group(1))
            corner_away = int(corner_match.group(2))
            break

    # Tarih
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
    """
    H2H (Head to Head) maçlarını çıkar
    PDF'te işaretli alan: Head to Head Statistics
    """
    home_key = home_team.lower().replace(' ', '')
    away_key = away_team.lower().replace(' ', '')

    # H2H bölümünü bul
    h2h_markers = ['head to head statistics', 'head to head', 'h2h statistics']
    h2h_section = ""

    html_lower = html.lower()
    for marker in h2h_markers:
        pos = html_lower.find(marker)
        if pos != -1:
            h2h_section = html[pos:pos+50000]  # H2H bölümünün bir kısmını al
            break

    if not h2h_section:
        h2h_section = html  # Bulamazsa tüm sayfada ara

    matches = []
    tables = extract_tables(h2h_section)

    for table in tables:
        rows = parse_table(table)
        for row in rows:
            match = parse_match_row(row)
            if not match:
                continue

            # Bu iki takım arasında mı?
            match_home_key = match.home.lower().replace(' ', '')
            match_away_key = match.away.lower().replace(' ', '')

            is_h2h = (match_home_key == home_key and match_away_key == away_key) or \
                     (match_home_key == away_key and match_away_key == home_key)

            if is_h2h:
                matches.append(match)

    return matches[:H2H_MATCHES]

def extract_previous_matches(html: str, team_name: str, league_name: str, is_home: bool) -> List[MatchData]:
    """
    Previous Scores Statistics verilerini çıkar
    PDF'te işaretli alan: Previous Scores Statistics
    - Ev sahibi için: Home + Same League seçili
    - Deplasman için: Away + Same League seçili
    """
    team_key = team_name.lower().replace(' ', '')
    league_key = league_name.lower().replace(' ', '') if league_name else ''

    # Previous Scores bölümünü bul
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

        # Bu takımın tablosu mu?
        if team_key not in table_text:
            continue

        rows = parse_table(table)
        for row in rows:
            match = parse_match_row(row)
            if not match:
                continue

            match_home_key = match.home.lower().replace(' ', '')
            match_away_key = match.away.lower().replace(' ', '')

            # Bu takımın maçı mı?
            is_team_match = (match_home_key == team_key) or (match_away_key == team_key)
            if not is_team_match:
                continue

            # İç/Dış saha kontrolü
            if is_home and match_home_key != team_key:
                continue
            if not is_home and match_away_key != team_key:
                continue

            # Same League kontrolü
            if league_key:
                match_league_key = match.league.lower().replace(' ', '')
                if league_key not in match_league_key:
                    continue

            matches.append(match)

    return matches[:RECENT_MATCHES]

# ============================================================
# İSTATİSTİK HESAPLAMA
# ============================================================
def calculate_team_stats(team_name: str, matches: List[MatchData], is_home: bool) -> TeamStats:
    """Bir takımın maç istatistiklerini hesapla"""
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
            # Bu takım ev sahibi
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
            # Bu takım deplasman
            goals_scored.append(match.score_away)
            goals_conceded.append(match.score_home)
            if match.corner_away:
                corners_for.append(match.corner_away)
            if match.corner_home:
                corners_against.append(match.corner_home)
            if match.score_home == 0:
                clean_sheets += 1
            away_matches.append(match)

    # Genel istatistikler
    stats.matches = len(matches)
    stats.goals_scored = sum(goals_scored) / len(goals_scored) if goals_scored else 0.0
    stats.goals_conceded = sum(goals_conceded) / len(goals_conceded) if goals_conceded else 0.0
    stats.corners_for = sum(corners_for) / len(corners_for) if corners_for else 0.0
    stats.corners_against = sum(corners_against) / len(corners_against) if corners_against else 0.0
    stats.clean_sheets = clean_sheets

    # İç saha
    if home_matches:
        stats.matches_home = len(home_matches)
        home_scored = [m.score_home for m in home_matches]
        home_conceded = [m.score_away for m in home_matches]
        stats.goals_scored_home = sum(home_scored) / len(home_scored)
        stats.goals_conceded_home = sum(home_conceded) / len(home_conceded)

    # Dış saha
    if away_matches:
        stats.matches_away = len(away_matches)
        away_scored = [m.score_away for m in away_matches]
        away_conceded = [m.score_home for m in away_matches]
        stats.goals_scored_away = sum(away_scored) / len(away_scored)
        stats.goals_conceded_away = sum(away_conceded) / len(away_conceded)

    return stats

# ============================================================
# LAMBDA HESAPLAMA (Beklenen Gol)
# ============================================================
def calculate_lambda(home_stats: TeamStats, away_stats: TeamStats,
                     home_standing: Dict, away_standing: Dict,
                     h2h_matches: List[MatchData]) -> Tuple[float, float, Dict]:
    """
    λ (lambda) hesaplama - Beklenen gol sayısı
    3 veri kaynağını ağırlıklandırarak birleştirir
    """
    info = {
        'methods': {},
        'weights': {},
        'final': {}
    }

    # 1) STANDING verilerinden
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

    # 2) PREVIOUS SCORES verilerinden (Same League)
    lambda_home_prev = home_stats.goals_scored_home if home_stats.matches_home > 0 else home_stats.goals_scored
    lambda_away_prev = away_stats.goals_scored_away if away_stats.matches_away > 0 else away_stats.goals_scored

    info['methods']['previous'] = {
        'home': lambda_home_prev,
        'away': lambda_away_prev,
        'weight': W_PREVIOUS
    }

    # 3) H2H verilerinden
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

    # Ağırlıklı ortalama hesapla
    total_weight = 0.0
    lambda_home_final = 0.0
    lambda_away_final = 0.0

    if lambda_home_st > 0:
        lambda_home_final += lambda_home_st * W_STANDING
        total_weight += W_STANDING

    if lambda_home_prev > 0:
        lambda_home_final += lambda_home_prev * W_PREVIOUS
        if lambda_home_st == 0:  # Standing yoksa weight'i artır
            total_weight += W_STANDING + W_PREVIOUS
        else:
            total_weight += W_PREVIOUS

    if lambda_home_h2h > 0:
        lambda_home_final += lambda_home_h2h * W_H2H
        total_weight += W_H2H

    if total_weight > 0:
        lambda_home_final /= total_weight
    else:
        lambda_home_final = 1.0  # Fallback

    # Away için aynı işlem
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
# POİSSON MODELİ
# ============================================================
def poisson_prob(k: int, lam: float) -> float:
    """Poisson olasılık fonksiyonu: P(X=k) = (λ^k * e^(-λ)) / k!"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def create_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 5) -> np.ndarray:
    """Skor olasılık matrisi oluştur"""
    matrix = np.zeros((max_goals + 1, max_goals + 1))

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[h, a] = poisson_prob(h, lambda_home) * poisson_prob(a, lambda_away)

    return matrix

def calculate_market_probabilities(score_matrix: np.ndarray) -> Dict[str, float]:
    """Skor matrisinden bahis piyasası olasılıklarını hesapla"""
    probs = {}

    # 1X2
    probs['1'] = np.sum(np.tril(score_matrix, -1))  # Ev kazanır
    probs['X'] = np.sum(np.diag(score_matrix))       # Beraberlik
    probs['2'] = np.sum(np.triu(score_matrix, 1))    # Deplasman kazanır

    # Total Goals (Alt/Üst)
    for line in [0.5, 1.5, 2.5, 3.5]:
        over = 0.0
        for h in range(score_matrix.shape[0]):
            for a in range(score_matrix.shape[1]):
                if h + a > line:
                    over += score_matrix[h, a]
        probs[f'O{line}'] = over
        probs[f'U{line}'] = 1.0 - over

    # BTTS (Her İki Takım da Gol Atar)
    btts_yes = 0.0
    for h in range(1, score_matrix.shape[0]):
        for a in range(1, score_matrix.shape[1]):
            btts_yes += score_matrix[h, a]
    probs['BTTS_Yes'] = btts_yes
    probs['BTTS_No'] = 1.0 - btts_yes

    return probs

def get_top_scores(score_matrix: np.ndarray, top_n: int = 7) -> List[Tuple[str, float]]:
    """En olası skorları bul"""
    scores = []
    for h in range(score_matrix.shape[0]):
        for a in range(score_matrix.shape[1]):
            scores.append((f'{h}-{a}', score_matrix[h, a]))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# ============================================================
# MONTE CARLO SİMÜLASYONU
# ============================================================
def monte_carlo_simulation(lambda_home: float, lambda_away: float, n_sims: int = 10000) -> Dict:
    """Monte Carlo simülasyonu ile olasılıkları hesapla"""
    np.random.seed(42)

    home_goals = np.random.poisson(lambda_home, n_sims)
    away_goals = np.random.poisson(lambda_away, n_sims)

    results = {
        '1': np.sum(home_goals > away_goals) / n_sims,
        'X': np.sum(home_goals == away_goals) / n_sims,
        '2': np.sum(home_goals < away_goals) / n_sims,
        'O2.5': np.sum((home_goals + away_goals) > 2.5) / n_sims,
        'U2.5': np.sum((home_goals + away_goals) <= 2.5) / n_sims,
        'BTTS_Yes': np.sum((home_goals > 0) & (away_goals > 0)) / n_sims,
        'BTTS_No': np.sum((home_goals == 0) | (away_goals == 0)) / n_sims,
    }

    return results

# ============================================================
# KORNER ANALİZİ
# ============================================================
def analyze_corners(home_stats: TeamStats, away_stats: TeamStats, h2h_matches: List[MatchData]) -> Dict:
    """Korner analizi ve tahmini"""

    # H2H korner ortalaması
    h2h_corners = []
    for match in h2h_matches:
        if match.corner_home and match.corner_away:
            h2h_corners.append(match.corner_home + match.corner_away)

    h2h_avg = sum(h2h_corners) / len(h2h_corners) if h2h_corners else 0.0

    # Takım ortalamaları
    home_for = home_stats.corners_for
    home_against = home_stats.corners_against
    away_for = away_stats.corners_for
    away_against = away_stats.corners_against

    # Tahmini korner sayısı
    if h2h_avg > 0:
        # H2H verisi varsa %60 H2H, %40 form
        predicted_home = 0.6 * (h2h_avg / 2) + 0.4 * ((home_for + away_against) / 2)
        predicted_away = 0.6 * (h2h_avg / 2) + 0.4 * ((away_for + home_against) / 2)
    else:
        # H2H yoksa sadece form
        predicted_home = (home_for + away_against) / 2
        predicted_away = (away_for + home_against) / 2

    total_corners = predicted_home + predicted_away

    return {
        'home': round(predicted_home, 1),
        'away': round(predicted_away, 1),
        'total': round(total_corners, 1),
        'h2h_avg': round(h2h_avg, 1),
        'confidence': 'Yüksek' if len(h2h_corners) >= 5 else 'Orta' if len(h2h_corners) >= 3 else 'Düşük'
    }

# ============================================================
# VALUE BET ANALİZİ
# ============================================================
def analyze_value_bets(probs: Dict[str, float], odds: Dict[str, float]) -> Dict:
    """
    Value Bet analizi
    Value = (Oran × Olasılık) - 1
    Kelly Criterion = (Oran × Olasılık - 1) / (Oran - 1)
    """
    if not odds:
        return {'decision': 'Oran verisi yok - analiz yapılamadı', 'bets': []}

    bets = []

    for market in ['1', 'X', '2']:
        if market in probs and market in odds:
            prob = probs[market]
            odd = odds[market]

            value = (odd * prob) - 1
            kelly = ((odd * prob) - 1) / (odd - 1) if odd > 1 else 0

            bets.append({
                'market': market,
                'prob': prob,
                'odd': odd,
                'value': value,
                'kelly': kelly,
                'playable': value >= VALUE_MIN and prob >= PROB_MIN
            })

    # Karar
    playable_bets = [b for b in bets if b['playable']]

    if playable_bets:
        best = max(playable_bets, key=lambda x: x['value'])
        decision = f"✅ OYNA: {best['market']} - Value: {best['value']*100:+.1f}%"
    else:
        decision = "⚠️  OYNAMA - Değerli bahis bulunamadı"

    return {'decision': decision, 'bets': bets}

# ============================================================
# ANA ANALİZ FONKSİYONU
# ============================================================
def analyze_match(url: str) -> str:
    """
    Ana analiz fonksiyonu - Tüm işlemleri yönetir
    """
    try:
        # 1) Sayfa içeriğini çek
        print("🔄 Sayfa yükleniyor...")
        html = get_page(url)

        # 2) Takım isimlerini bul
        home_team, away_team = extract_team_names(html)
        if not home_team or not away_team:
            return "❌ HATA: Takım isimleri bulunamadı"

        print(f"⚽ {home_team} vs {away_team}")

        # Lig ismi
        league_match = re.search(r'Italian Serie A|Premier League|La Liga|Bundesliga|Ligue 1', html, re.IGNORECASE)
        league_name = league_match.group(0) if league_match else ""

        # 3) Verileri çek
        print("📊 Veriler çekiliyor...")

        # Bet365 oranları
        odds = extract_bet365_odds(html)

        # Standing verileri
        home_standing = extract_standings(html, home_team)
        away_standing = extract_standings(html, away_team)

        # H2H maçları
        h2h_matches = extract_h2h_matches(html, home_team, away_team)

        # Previous maçlar (Same League)
        home_prev = extract_previous_matches(html, home_team, league_name, is_home=True)
        away_prev = extract_previous_matches(html, away_team, league_name, is_home=False)

        # 4) İstatistikleri hesapla
        print("🧮 İstatistikler hesaplanıyor...")
        home_stats = calculate_team_stats(home_team, home_prev, is_home=True)
        away_stats = calculate_team_stats(away_team, away_prev, is_home=False)

        # 5) Lambda (Beklenen gol) hesapla
        lambda_home, lambda_away, lambda_info = calculate_lambda(
            home_stats, away_stats,
            home_standing, away_standing,
            h2h_matches
        )

        # 6) Poisson modeli
        print("📈 Olasılıklar hesaplanıyor...")
        score_matrix = create_score_matrix(lambda_home, lambda_away)
        poisson_probs = calculate_market_probabilities(score_matrix)
        top_scores = get_top_scores(score_matrix)

        # 7) Monte Carlo
        mc_probs = monte_carlo_simulation(lambda_home, lambda_away, MC_SIMULATIONS)

        # 8) Korner analizi
        corner_analysis = analyze_corners(home_stats, away_stats, h2h_matches)

        # 9) Value Bet analizi
        value_analysis = analyze_value_bets(poisson_probs, odds) if odds else {
            'decision': 'Oran verisi yok', 'bets': []
        }

        # 10) Sonuçları formatla
        return format_results(
            home_team, away_team, league_name,
            lambda_home, lambda_away,
            poisson_probs, mc_probs, top_scores,
            corner_analysis, value_analysis, odds,
            home_standing, away_standing,
            len(home_prev), len(away_prev), len(h2h_matches)
        )

    except Exception as e:
        return f"❌ HATA: {str(e)}\n\n{traceback.format_exc()}"

# ============================================================
# SONUÇ FORMATLAMA (SADE VE NET)
# ============================================================
def format_results(home: str, away: str, league: str,
                   lam_h: float, lam_a: float,
                   poisson: Dict, mc: Dict, top_scores: List,
                   corners: Dict, value: Dict, odds: Optional[Dict],
                   home_st: Dict, away_st: Dict,
                   n_home_prev: int, n_away_prev: int, n_h2h: int) -> str:
    """SADE VE NET ÇIKTI - Sadece önemli bilgiler"""

    lines = []
    lines.append("=" * 60)
    lines.append(f"  {home} vs {away}")
    if league:
        lines.append(f"  {league}")
    lines.append("=" * 60)

    # EN OLASI SKORLAR
    lines.append("\n🎯 EN OLASI SKORLAR:")
    for i, (score, prob) in enumerate(top_scores[:5], 1):
        bar = "█" * int(prob * 50)
        lines.append(f"  {i}. {score:6s} %{prob*100:4.1f}  {bar}")

    # NET TAHMİN
    lines.append("\n📋 NET TAHMİN:")
    lines.append(f"  Ana Skor: {top_scores[0][0]}")
    lines.append(f"  Alt Skorlar: {top_scores[1][0]}, {top_scores[2][0]}")

    # BEKLENEN GOL
    lines.append("\n⚽ BEKLENEN GOL:")
    lines.append(f"  {home}: {lam_h:.2f} gol")
    lines.append(f"  {away}: {lam_a:.2f} gol")
    lines.append(f"  Toplam: {lam_h + lam_a:.2f} gol")

    # ALT/ÜST 2.5
    o25 = poisson.get('O2.5', 0)
    u25 = poisson.get('U2.5', 0)
    prediction_ou = "ÜST 2.5" if o25 > u25 else "ALT 2.5"
    lines.append(f"\n📊 ALT/ÜST 2.5: {prediction_ou} (%{max(o25, u25)*100:.1f})")
    lines.append(f"  Üst 2.5: %{o25*100:.1f}")
    lines.append(f"  Alt 2.5: %{u25*100:.1f}")

    # BTTS
    btts_yes = poisson.get('BTTS_Yes', 0)
    btts_no = poisson.get('BTTS_No', 0)
    prediction_btts = "VAR" if btts_yes > btts_no else "YOK"
    lines.append(f"\n⚽ KG VAR: {prediction_btts} (%{max(btts_yes, btts_no)*100:.1f})")
    lines.append(f"  Var: %{btts_yes*100:.1f}")
    lines.append(f"  Yok: %{btts_no*100:.1f}")

    # 1X2
    lines.append("\n🏆 1X2 OLASILIKLAR:")
    lines.append(f"  Ev (1): %{poisson.get('1', 0)*100:.1f}")
    lines.append(f"  Ber(X): %{poisson.get('X', 0)*100:.1f}")
    lines.append(f"  Dep(2): %{poisson.get('2', 0)*100:.1f}")

    # KORNER
    if corners['total'] > 0:
        lines.append(f"\n🚩 KORNER TAHMİNİ: {corners['total']} (Güven: {corners['confidence']})")
        lines.append(f"  {home}: {corners['home']}")
        lines.append(f"  {away}: {corners['away']}")
        if corners['h2h_avg'] > 0:
            lines.append(f"  H2H Ort: {corners['h2h_avg']}")

    # VALUE BET ANALİZİ
    lines.append("\n💰 BAHIS ANALİZİ:")
    if odds:
        lines.append(f"  Oranlar: 1:{odds.get('1', 0):.2f} X:{odds.get('X', 0):.2f} 2:{odds.get('2', 0):.2f}")

        for bet in value['bets']:
            market_name = {'1': 'Ev', 'X': 'Beraberlik', '2': 'Deplasman'}[bet['market']]
            status = "✅" if bet['playable'] else "  "
            lines.append(f"  {status} {market_name}: Value %{bet['value']*100:+.1f} | Olasılık %{bet['prob']*100:.1f}")

        lines.append(f"\n  {value['decision']}")
    else:
        lines.append("  ⚠️  Oran verisi bulunamadı")

    # VERİ KAYNAKLARI
    lines.append("\n📂 KULLANILAN VERİLER:")
    lines.append(f"  Standing: {'✓' if home_st and away_st else '✗'}")
    lines.append(f"  PSS (Same League): ✓ (Ev:{n_home_prev} | Dep:{n_away_prev})" if n_home_prev > 0 or n_away_prev > 0 else "  PSS: ✗")
    lines.append(f"  H2H: ✓ ({n_h2h} maç)" if n_h2h > 0 else "  H2H: ✗")

    # AĞIRLIKLAR
    lines.append("\n⚖️  AĞIRLIKLAR:")
    lines.append(f"  Standing: %{W_STANDING*100:.0f}")
    lines.append(f"  PSS (Same League): %{W_PREVIOUS*100:.0f}")
    lines.append(f"  H2H: %{W_H2H*100:.0f}")

    lines.append("=" * 60)

    return "\n".join(lines)

# ============================================================
# ÇALIŞTIRMA
# ============================================================
if __name__ == "__main__":
    # Test URL
    test_url = "https://live3.nowgoal26.com/match/h2h-2784675"

    print("\n" + "=" * 60)
    print("  FUTBOL MAÇ ANALİZ PROGRAMI")
    print("=" * 60)

    result = analyze_match(test_url)
    print("\n" + result)
