# -*- coding: utf-8 -*-
"""
FUTBOL MAÇ ANALİZ API - SYNTAX FIXED VERSION
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

app = Flask(__name__)
CORS(app)

# ============================================================
# AYARLAR
# ============================================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ============================================================
# VERİ YAPILARI
# ============================================================
@dataclass
class MatchData:
    home: str
    away: str
    score_home: int
    score_away: int
    corner_home: Optional[int] = None
    corner_away: Optional[int] = None

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def log_info(message: str):
    """Log mesajı"""
    print(f"[INFO] {message}")

def log_error(message: str, error: Exception = None):
    """Hata logu"""
    print(f"[ERROR] {message}")
    if error:
        print(f"[ERROR] {str(error)}")

# ============================================================
# HTML PARSE
# ============================================================
def strip_html(text: str) -> str:
    """HTML taglerini temizle"""
    try:
        text = re.sub(r'<script.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
        return re.sub(r'\s+', ' ', text).strip()
    except:
        return str(text)

def extract_tables(html: str) -> List[str]:
    """HTML'den tabloları çıkar"""
    try:
        return re.findall(r'<table.*?</table>', html, flags=re.DOTALL | re.IGNORECASE)
    except:
        return []

def parse_table(table_html: str) -> List[List[str]]:
    """Tabloyu parse et"""
    rows = []
    try:
        trs = re.findall(r'<tr.*?</tr>', table_html, flags=re.DOTALL | re.IGNORECASE)
        for tr in trs:
            cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', tr, flags=re.DOTALL | re.IGNORECASE)
            cleaned = [strip_html(c) for c in cells if c and strip_html(c)]
            if cleaned and len(cleaned) >= 3:
                rows.append(cleaned)
    except Exception as e:
        log_error("Table parse error", e)
    return rows

# ============================================================
# VERİ ÇEKME
# ============================================================
def get_page(url: str, timeout: int = 15) -> Optional[str]:
    """Sayfa içeriğini çek"""
    try:
        log_info(f"Fetching: {url}")
        response = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        response.encoding = 'utf-8'
        log_info(f"Page fetched: {len(response.text)} chars")
        return response.text
    except requests.exceptions.Timeout:
        log_error("Timeout error")
        return None
    except requests.exceptions.RequestException as e:
        log_error("Request error", e)
        return None
    except Exception as e:
        log_error("Unexpected error in get_page", e)
        return None

def extract_team_names(html: str) -> Tuple[str, str]:
    """Takım isimlerini çıkar"""
    try:
        # Yöntem 1: Title'dan
        match = re.search(r'<title>\s*(.*?)\s*</title>', html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            title = strip_html(match.group(1))
            # vs, VS, v, V ile ayır
            vs_match = re.search(r'(.+?)\s+(?:vs|VS|v|V)\s+(.+?)(?:\s+-|\s+\||$)', title, flags=re.IGNORECASE)
            if vs_match:
                home = vs_match.group(1).strip()
                away = vs_match.group(2).strip()
                # İlk 3 kelimeyi al (çok uzunsa)
                home = ' '.join(home.split()[:3])
                away = ' '.join(away.split()[:3])
                if home and away and len(home) > 2 and len(away) > 2:
                    log_info(f"Teams found: {home} vs {away}")
                    return home, away

        log_error("Could not extract team names")
        return "", ""
    except Exception as e:
        log_error("Error extracting team names", e)
        return "", ""

def extract_odds(html: str) -> Optional[Dict[str, float]]:
    """Bet365 oranlarını çıkar"""
    try:
        # Basitleştirilmiş yaklaşım
        bet365_section = ""
        bet365_pos = html.lower().find('bet365')
        if bet365_pos != -1:
            bet365_section = html[bet365_pos:bet365_pos+5000]

        if bet365_section:
            initial_pos = bet365_section.lower().find('initial')
            if initial_pos != -1:
                odds_section = bet365_section[initial_pos:initial_pos+500]
                # X.XX formatında sayıları bul
                odds_pattern = r'\b([1-9]\d*\.\d{2})\b'
                odds = re.findall(odds_pattern, odds_section)
                if len(odds) >= 3:
                    result = {
                        "1": float(odds[0]),
                        "X": float(odds[1]),
                        "2": float(odds[2])
                    }
                    # Makul aralıkta mı
                    if all(1.01 <= v <= 50.0 for v in result.values()):
                        log_info(f"Odds found: {result}")
                        return result

        log_info("Odds not found")
        return None
    except Exception as e:
        log_error("Error extracting odds", e)
        return None

def parse_match_row(cells: List[str]) -> Optional[MatchData]:
    """Maç satırını parse et"""
    try:
        if len(cells) < 5:
            return None

        # Skor formatı: "2-1" veya "2-1(1-0)"
        score_pattern = r'(\d{1,2})\s*[-:]\s*(\d{1,2})'

        for i, cell in enumerate(cells):
            match = re.search(score_pattern, cell)
            if match and i > 0 and i < len(cells) - 1:
                home = strip_html(cells[i-1])
                away = strip_html(cells[i+1])

                if home and away and len(home) > 2 and len(away) > 2:
                    ft_home = int(match.group(1))
                    ft_away = int(match.group(2))

                    # Korner ara
                    corner_home, corner_away = None, None
                    for j in range(i+2, min(i+5, len(cells))):
                        corner_match = re.search(r'(\d{1,2})\s*[-:]\s*(\d{1,2})', cells[j])
                        if corner_match:
                            ch = int(corner_match.group(1))
                            ca = int(corner_match.group(2))
                            if 0 <= ch <= 20 and 0 <= ca <= 20:
                                corner_home = ch
                                corner_away = ca
                                break

                    return MatchData(
                        home=home, away=away,
                        score_home=ft_home, score_away=ft_away,
                        corner_home=corner_home, corner_away=corner_away
                    )
        return None
    except:
        return None

def extract_matches(html: str, team1: str, team2: str, match_type: str = "all") -> List[MatchData]:
    """Maçları çıkar"""
    matches = []
    try:
        team1_key = team1.lower().replace(' ', '')
        team2_key = team2.lower().replace(' ', '') if team2 else ""

        tables = extract_tables(html)
        log_info(f"Found {len(tables)} tables")

        for table in tables:
            rows = parse_table(table)
            for row in rows:
                match = parse_match_row(row)
                if not match:
                    continue

                match_home_key = match.home.lower().replace(' ', '')
                match_away_key = match.away.lower().replace(' ', '')

                if match_type == "h2h":
                    is_h2h = (match_home_key == team1_key and match_away_key == team2_key) or \
                             (match_home_key == team2_key and match_away_key == team1_key)
                    if is_h2h:
                        matches.append(match)
                elif match_type == "home":
                    if match_home_key == team1_key:
                        matches.append(match)
                elif match_type == "away":
                    if match_away_key == team1_key:
                        matches.append(match)
                else:
                    if team1_key in match_home_key or team1_key in match_away_key:
                        matches.append(match)

        log_info(f"Extracted {len(matches)} matches (type: {match_type})")
        return matches[:10]
    except Exception as e:
        log_error(f"Error extracting matches ({match_type})", e)
        return []

# ============================================================
# İSTATİSTİK
# ============================================================
def calculate_stats(team_name: str, matches: List[MatchData]) -> Dict[str, float]:
    """Takım istatistiklerini hesapla"""
    try:
        if not matches:
            return {"goals_scored": 1.0, "goals_conceded": 1.0, "corners": 5.0}

        team_key = team_name.lower().replace(' ', '')
        goals_scored = []
        goals_conceded = []
        corners = []

        for match in matches:
            match_home_key = match.home.lower().replace(' ', '')
            if team_key in match_home_key:
                goals_scored.append(match.score_home)
                goals_conceded.append(match.score_away)
                if match.corner_home:
                    corners.append(match.corner_home)
            else:
                goals_scored.append(match.score_away)
                goals_conceded.append(match.score_home)
                if match.corner_away:
                    corners.append(match.corner_away)

        avg_scored = sum(goals_scored) / len(goals_scored) if goals_scored else 1.0
        avg_conceded = sum(goals_conceded) / len(goals_conceded) if goals_conceded else 1.0
        avg_corners = sum(corners) / len(corners) if corners else 5.0

        return {
            "goals_scored": round(avg_scored, 2),
            "goals_conceded": round(avg_conceded, 2),
            "corners": round(avg_corners, 1)
        }
    except Exception as e:
        log_error("Error calculating stats", e)
        return {"goals_scored": 1.0, "goals_conceded": 1.0, "corners": 5.0}

def calculate_lambda(home_stats: Dict, away_stats: Dict, h2h_matches: List[MatchData]) -> Tuple[float, float]:
    """Lambda hesapla"""
    try:
        lambda_home = home_stats["goals_scored"] * 0.6 + away_stats["goals_conceded"] * 0.4
        lambda_away = away_stats["goals_scored"] * 0.6 + home_stats["goals_conceded"] * 0.4

        if h2h_matches:
            h2h_home_goals = sum(m.score_home for m in h2h_matches)
            h2h_away_goals = sum(m.score_away for m in h2h_matches)
            h2h_avg_home = h2h_home_goals / len(h2h_matches)
            h2h_avg_away = h2h_away_goals / len(h2h_matches)

            lambda_home = lambda_home * 0.7 + h2h_avg_home * 0.3
            lambda_away = lambda_away * 0.7 + h2h_avg_away * 0.3

        lambda_home = max(0.3, min(3.5, lambda_home))
        lambda_away = max(0.3, min(3.5, lambda_away))

        log_info(f"Lambda: Home={lambda_home:.2f}, Away={lambda_away:.2f}")
        return lambda_home, lambda_away
    except Exception as e:
        log_error("Error calculating lambda", e)
        return 1.2, 1.0

# ============================================================
# POİSSON
# ============================================================
def poisson_prob(k: int, lam: float) -> float:
    """Poisson olasılığı"""
    try:
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    except:
        return 0.0

def calculate_predictions(lambda_home: float, lambda_away: float) -> Dict[str, Any]:
    """Tahminleri hesapla"""
    try:
        scores = []
        for h in range(6):
            for a in range(6):
                prob = poisson_prob(h, lambda_home) * poisson_prob(a, lambda_away)
                scores.append((f"{h}-{a}", prob))
        scores.sort(key=lambda x: x[1], reverse=True)

        prob_1 = sum(poisson_prob(h, lambda_home) * poisson_prob(a, lambda_away) 
                     for h in range(6) for a in range(6) if h > a)
        prob_X = sum(poisson_prob(k, lambda_home) * poisson_prob(k, lambda_away) 
                     for k in range(6))
        prob_2 = sum(poisson_prob(h, lambda_home) * poisson_prob(a, lambda_away) 
                     for h in range(6) for a in range(6) if h < a)

        prob_over = sum(poisson_prob(h, lambda_home) * poisson_prob(a, lambda_away) 
                        for h in range(6) for a in range(6) if h + a > 2.5)
        prob_under = 1.0 - prob_over

        prob_btts_yes = sum(poisson_prob(h, lambda_home) * poisson_prob(a, lambda_away) 
                            for h in range(1, 6) for a in range(1, 6))
        prob_btts_no = 1.0 - prob_btts_yes

        return {
            "top_scores": [{"score": s, "prob": round(p*100, 1)} for s, p in scores[:5]],
            "main_score": scores[0][0],
            "alt_scores": [scores[1][0], scores[2][0]],
            "match_result": {
                "home_win": round(prob_1 * 100, 1),
                "draw": round(prob_X * 100, 1),
                "away_win": round(prob_2 * 100, 1)
            },
            "over_under": {
                "prediction": "ÜST" if prob_over > prob_under else "ALT",
                "over_prob": round(prob_over * 100, 1),
                "under_prob": round(prob_under * 100, 1)
            },
            "btts": {
                "prediction": "VAR" if prob_btts_yes > prob_btts_no else "YOK",
                "yes_prob": round(prob_btts_yes * 100, 1),
                "no_prob": round(prob_btts_no * 100, 1)
            }
        }
    except Exception as e:
        log_error("Error calculating predictions", e)
        return {
            "top_scores": [{"score": "1-1", "prob": 15.0}],
            "main_score": "1-1",
            "alt_scores": ["1-0", "0-1"]
        }

def analyze_value_bets(predictions: Dict, odds: Optional[Dict]) -> Dict:
    """Value bet analizi"""
    try:
        if not odds:
            return {"decision": "Oran verisi yok", "bets": []}

        probs = {
            "1": predictions["match_result"]["home_win"] / 100,
            "X": predictions["match_result"]["draw"] / 100,
            "2": predictions["match_result"]["away_win"] / 100
        }

        bets = []
        for market in ["1", "X", "2"]:
            if market in probs and market in odds:
                prob = probs[market]
                odd = odds[market]
                value = (odd * prob) - 1

                bets.append({
                    "market": market,
                    "prob": round(prob, 3),
                    "odd": odd,
                    "value": round(value, 3),
                    "playable": value >= 0.05 and prob >= 0.30
                })

        playable = [b for b in bets if b["playable"]]
        if playable:
            best = max(playable, key=lambda x: x["value"])
            names = {"1": "Ev", "X": "Beraberlik", "2": "Deplasman"}
            decision = f"OYNA: {names[best['market']]} - Value: {best['value']*100:+.1f}%"
        else:
            decision = "OYNAMA - Değerli bahis yok"

        return {"decision": decision, "bets": bets}
    except Exception as e:
        log_error("Error analyzing value bets", e)
        return {"decision": "Analiz hatası", "bets": []}

# ============================================================
# ANA ANALİZ
# ============================================================
def analyze_match(url: str) -> Dict[str, Any]:
    """Ana analiz fonksiyonu"""
    try:
        log_info(f"Starting analysis for: {url}")

        html = get_page(url)
        if not html:
            return {
                "success": False,
                "error": "Sayfa yüklenemedi. NowGoal sitesi erişilebilir değil veya timeout oluştu."
            }

        home_team, away_team = extract_team_names(html)
        if not home_team or not away_team:
            return {
                "success": False,
                "error": "Takım isimleri bulunamadı. URL formatı doğru mu kontrol edin."
            }

        h2h_matches = extract_matches(html, home_team, away_team, "h2h")
        home_matches = extract_matches(html, home_team, "", "home")
        away_matches = extract_matches(html, away_team, "", "away")

        log_info(f"Matches found - H2H: {len(h2h_matches)}, Home: {len(home_matches)}, Away: {len(away_matches)}")

        home_stats = calculate_stats(home_team, home_matches)
        away_stats = calculate_stats(away_team, away_matches)

        lambda_home, lambda_away = calculate_lambda(home_stats, away_stats, h2h_matches)

        predictions = calculate_predictions(lambda_home, lambda_away)

        odds = extract_odds(html)

        value_analysis = analyze_value_bets(predictions, odds)

        corner_total = (home_stats["corners"] + away_stats["corners"]) / 2

        return {
            "success": True,
            "match_info": {
                "home_team": home_team,
                "away_team": away_team
            },
            "expected_goals": {
                "home": round(lambda_home, 2),
                "away": round(lambda_away, 2),
                "total": round(lambda_home + lambda_away, 2)
            },
            "predictions": predictions,
            "corners": {
                "total": round(corner_total, 1),
                "confidence": "Orta"
            },
            "value_bets": {
                "decision": value_analysis["decision"],
                "odds": odds,
                "analysis": value_analysis["bets"]
            },
            "data_sources": {
                "h2h_matches": len(h2h_matches),
                "home_matches": len(home_matches),
                "away_matches": len(away_matches)
            }
        }

    except Exception as e:
        log_error("Fatal error in analyze_match", e)
        return {
            "success": False,
            "error": f"Analiz hatası: {str(e)}",
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
        "version": "1.3-syntax-fixed",
        "health": "healthy"
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

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
        log_info(f"API request received for: {url}")

        result = analyze_match(url)

        if result.get('success'):
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        log_error("API endpoint error", e)
        return jsonify({
            "success": False,
            "error": f"API hatası: {str(e)}"
        }), 500

# ============================================================
# START
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    log_info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
