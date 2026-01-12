# -*- coding: utf-8 -*-
"""
API TEST DOSYASI
Render'daki API'yi test etmek için
"""

import requests
import json

# Render URL'in
API_URL = "https://macanalizor-api.onrender.com"

def test_home():
    """Ana sayfayı test et"""
    print("\n" + "="*70)
    print("TEST 1: Ana Sayfa")
    print("="*70)

    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_health():
    """Health check test et"""
    print("\n" + "="*70)
    print("TEST 2: Health Check")
    print("="*70)

    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_analyze(url):
    """Maç analizini test et"""
    print("\n" + "="*70)
    print("TEST 3: Maç Analizi")
    print("="*70)
    print(f"URL: {url}")
    print("Analiz yapılıyor... (30-60 saniye sürebilir)")

    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"url": url},
            timeout=120  # 2 dakika timeout
        )

        print(f"\nStatus: {response.status_code}")

        result = response.json()

        if result.get('success'):
            print("\n✅ ANALİZ BAŞARILI!")
            print(f"\nMaç: {result['match_info']['home_team']} vs {result['match_info']['away_team']}")
            print(f"Lig: {result['match_info']['league']}")
            print(f"\nBeklenen Gol: {result['expected_goals']['home']} - {result['expected_goals']['away']}")
            print(f"\nTahmin: {result['predictions']['main_score']}")
            print(f"Alt Skorlar: {', '.join(result['predictions']['alt_scores'])}")
            print(f"\nAlt/Üst 2.5: {result['predictions']['over_under_2_5']['prediction']} ({result['predictions']['over_under_2_5']['over_prob']}% - {result['predictions']['over_under_2_5']['under_prob']}%)")
            print(f"KG Var: {result['predictions']['btts']['prediction']} ({result['predictions']['btts']['yes_prob']}% - {result['predictions']['btts']['no_prob']}%)")
            print(f"\n1X2: Ev %{result['predictions']['match_result']['home_win']} | Ber %{result['predictions']['match_result']['draw']} | Dep %{result['predictions']['match_result']['away_win']}")

            if result['corners']['total'] > 0:
                print(f"\nKorner: {result['corners']['total']} ({result['corners']['confidence']} güven)")

            print(f"\nBahis Kararı: {result['value_bets']['decision']}")

            print(f"\nVeri Kaynakları:")
            print(f"  Standing: {'✓' if result['data_sources']['standings'] else '✗'}")
            print(f"  Previous (Home): {result['data_sources']['previous_home']} maç")
            print(f"  Previous (Away): {result['data_sources']['previous_away']} maç")
            print(f"  H2H: {result['data_sources']['h2h']} maç")

        else:
            print(f"\n❌ HATA: {result.get('error')}")
            if 'traceback' in result:
                print(f"\nDetay:\n{result['traceback']}")

    except requests.exceptions.Timeout:
        print("\n⚠️  TIMEOUT: İstek 2 dakikayı aştı. Render Free plan cold start olabilir.")
    except Exception as e:
        print(f"\n❌ HATA: {e}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FUTBOL MAÇ ANALİZ API - TEST")
    print("="*70)

    # Test 1: Ana sayfa
    test_home()

    # Test 2: Health check
    test_health()

    # Test 3: Maç analizi
    test_url = "https://live3.nowgoal26.com/match/h2h-2784675"
    test_analyze(test_url)

    print("\n" + "="*70)
    print("TESTLER TAMAMLANDI!")
    print("="*70)
