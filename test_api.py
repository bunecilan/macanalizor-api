# -*- coding: utf-8 -*-
"""
LOKALde TEST ET - Render'a yüklemeden önce çalıştır
"""

import requests
import json

# Test 1: Local test (python app.py çalıştır)
LOCAL_URL = "http://localhost:10000"

# Test 2: Render test
RENDER_URL = "https://macanalizor-api.onrender.com"

def test_api(base_url):
    print(f"\n{'='*70}")
    print(f"Testing: {base_url}")
    print('='*70)

    # Test 1: Health check
    try:
        print("\n[1/3] Health check...")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return

    # Test 2: Ana sayfa
    try:
        print("\n[2/3] Home page...")
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Home page failed: {e}")

    # Test 3: Maç analizi
    try:
        print("\n[3/3] Match analysis...")
        print("⏳ Bu 30-60 saniye sürebilir...")

        test_url = "https://live3.nowgoal26.com/match/h2h-2784675"

        response = requests.post(
            f"{base_url}/analyze",
            json={"url": test_url},
            timeout=120
        )

        print(f"\n✅ Status: {response.status_code}")

        result = response.json()

        if result.get('success'):
            print("\n🎯 ANALİZ BAŞARILI!")
            print(f"\nMaç: {result['match_info']['home_team']} vs {result['match_info']['away_team']}")
            print(f"Beklenen Gol: {result['expected_goals']['home']} - {result['expected_goals']['away']}")
            print(f"\nTahmin: {result['predictions']['main_score']}")
            print(f"Alt Skorlar: {', '.join(result['predictions']['alt_scores'])}")
            print(f"\n1X2: Ev %{result['predictions']['match_result']['home_win']} | Ber %{result['predictions']['match_result']['draw']} | Dep %{result['predictions']['match_result']['away_win']}")
            print(f"Alt/Üst 2.5: {result['predictions']['over_under']['prediction']} ({result['predictions']['over_under']['over_prob']}% - {result['predictions']['over_under']['under_prob']}%)")
            print(f"KG Var: {result['predictions']['btts']['prediction']} ({result['predictions']['btts']['yes_prob']}% - {result['predictions']['btts']['no_prob']}%)")
            print(f"\nKorner: {result['corners']['total']}")
            print(f"\nBahis Kararı: {result['value_bets']['decision']}")

            print(f"\nVeri Kaynakları:")
            print(f"  H2H: {result['data_sources']['h2h_matches']} maç")
            print(f"  Home: {result['data_sources']['home_matches']} maç")
            print(f"  Away: {result['data_sources']['away_matches']} maç")

            # Full response'u kaydet
            with open('test_response.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Full response saved to: test_response.json")
        else:
            print(f"\n❌ ANALİZ BAŞARISIZ!")
            print(f"Error: {result.get('error')}")
            if 'traceback' in result:
                print(f"\nDetay:\n{result['traceback']}")

    except requests.exceptions.Timeout:
        print("\n❌ TIMEOUT: 2 dakikayı aştı")
    except Exception as e:
        print(f"\n❌ HATA: {e}")

if __name__ == "__main__":
    import sys

    print("\n" + "🧪"*35)
    print("FUTBOL MAÇ ANALİZ API - TEST")
    print("🧪"*35)

    if len(sys.argv) > 1 and sys.argv[1] == "local":
        print("\n📍 LOCAL TEST MODE")
        print("Önce başka bir terminalde çalıştır: python app.py")
        input("\nEnter tuşuna basarak devam et...")
        test_api(LOCAL_URL)
    else:
        print("\n🌐 RENDER TEST MODE")
        test_api(RENDER_URL)

    print("\n" + "="*70)
    print("TEST TAMAMLANDI!")
    print("="*70)
