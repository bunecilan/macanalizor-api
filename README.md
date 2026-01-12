⚽ FUTBOL MAÇ ANALİZ API
Bu API, NowGoal sitesinden futbol maçı verilerini çekerek, Poisson olasılık modeli ve Monte Carlo simülasyonu ile maç tahminleri yapar.

🚀 RENDER DEPLOYMENT
1. GitHub'a Yükle
bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/KULLANICI_ADI/REPO_ADI.git
git push -u origin main
2. Render'da Ayarla
Render Dashboard 'a git

"New +" → "Web Service" seç

GitHub repo'nu bağla

Ayarlar:

Name: macanalizor-api

Environment: Python 3

Build Command: pip install -r requirements.txt

Start Command: gunicorn app:app

Region: Frankfurt (veya en yakın)

Plan: Free

"Create Web Service" tıkla

Deploy tamamlanınca URL'ni al: https://macanalizor-api.onrender.com

📡 API KULLANIMI
Base URL
text
https://macanalizor-api.onrender.com
Endpoint 1: Ana Sayfa (GET)
bash
curl https://macanalizor-api.onrender.com/
Response:

json
{
  "status": "online",
  "service": "Futbol Maç Analiz API",
  "version": "1.0"
}
Endpoint 2: Maç Analizi (POST)
bash
curl -X POST https://macanalizor-api.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://live3.nowgoal26.com/match/h2h-2784675"}'
Request Body:

json
{
  "url": "https://live3.nowgoal26.com/match/h2h-2784675"
}
Response Örneği:

json
{
  "success": true,
  "match_info": {
    "home_team": "Genoa",
    "away_team": "Cagliari",
    "league": "Italian Serie A"
  },
  "expected_goals": {
    "home": 1.05,
    "away": 0.95,
    "total": 2.00
  },
  "top_scores": [
    {"score": "1-1", "probability": 12.5},
    {"score": "1-0", "probability": 11.8},
    {"score": "0-1", "probability": 10.3}
  ],
  "predictions": {
    "main_score": "1-1",
    "alt_scores": ["1-0", "0-1"],
    "over_under_2_5": {
      "prediction": "ALT",
      "over_prob": 41.7,
      "under_prob": 58.3
    },
    "btts": {
      "prediction": "VAR",
      "yes_prob": 52.1,
      "no_prob": 47.9
    },
    "match_result": {
      "home_win": 35.2,
      "draw": 31.8,
      "away_win": 33.0
    }
  },
  "corners": {
    "home": 5.5,
    "away": 5.0,
    "total": 10.5,
    "h2h_avg": 8.0,
    "confidence": "Yüksek"
  },
  "value_bets": {
    "decision": "OYNA: Deplasman - Value: +12.2%",
    "odds": {"1": 2.25, "X": 3.00, "2": 3.40},
    "analysis": [
      {
        "market": "1",
        "prob": 0.352,
        "odd": 2.25,
        "value": -0.208,
        "kelly": -0.166,
        "playable": false
      },
      {
        "market": "2",
        "prob": 0.330,
        "odd": 3.40,
        "value": 0.122,
        "kelly": 0.051,
        "playable": true
      }
    ]
  },
  "data_sources": {
    "standings": true,
    "previous_home": 10,
    "previous_away": 10,
    "h2h": 10
  },
  "weights": {
    "standing": "45%",
    "previous": "30%",
    "h2h": "25%"
  }
}
Endpoint 3: Health Check (GET)
bash
curl https://macanalizor-api.onrender.com/health
Response:

json
{
  "status": "healthy",
  "timestamp": 1736654280.5
}
🧪 TEST ETME
Python ile:
python
import requests

url = "https://macanalizor-api.onrender.com/analyze"
data = {"url": "https://live3.nowgoal26.com/match/h2h-2784675"}

response = requests.post(url, json=data)
print(response.json())
JavaScript ile:
javascript
fetch('https://macanalizor-api.onrender.com/analyze', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    url: 'https://live3.nowgoal26.com/match/h2h-2784675'
  })
})
.then(res => res.json())
.then(data => console.log(data));
cURL ile:
bash
curl -X POST https://macanalizor-api.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -d '{"url":"https://live3.nowgoal26.com/match/h2h-2784675"}'
📊 NASIL ÇALIŞIR?
1. Veri Toplama
Bet365 Oranları: Initial 1X2 oranları

Standing: Lig sıralaması ve istatistikler

H2H: İki takımın geçmiş karşılaşmaları

Previous Scores: Son maç performansları (Same League)

Korner Verileri: Korner istatistikleri

2. Lambda Hesaplama (Beklenen Gol)
text
λ = (Standing × 45%) + (Previous × 30%) + (H2H × 25%)
3. Poisson Modeli
text
P(k gol) = (λ^k × e^(-λ)) / k!
Her skor için olasılık hesaplanır.

4. Monte Carlo Simülasyonu
10,000 maç simüle edilir ve sonuçlar doğrulanır.

5. Value Bet Analizi
text
Value = (Oran × Olasılık) - 1
Kelly = ((Oran × Olasılık) - 1) / (Oran - 1)
Karar Eşiği:
✅ OYNA: Value ≥ %5 VE Olasılık ≥ %55

⚠️ OYNAMA: Diğer durumlar

🔧 LOKAL ÇALIŞTIRMA
bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Uygulamayı başlat
python app.py
Tarayıcıda aç: http://localhost:5000

📁 DOSYA YAPISI
text
macanalizor-api/
│
├── app.py                  # Ana Flask uygulaması
├── requirements.txt        # Python bağımlılıkları
├── Procfile               # Render start komutu
└── README.md              # Bu dosya
⚠️ NOTLAR
Free Plan Sınırları:

Render Free plan 15 dakika inaktivite sonrası uyur

İlk istek 30-60 saniye sürebilir (cold start)

750 saat/ay kullanım limiti

Rate Limiting:

NowGoal sitesi çok fazla istek engelleyebilir

Dakikada en fazla 10-15 istek önerilir

Timeout:

Render Free plan: 30 saniyelik timeout

Karmaşık analizler zaman alabilir

🛠️ SORUN GİDERME
"Application Error" hatası
Render loglarını kontrol et: Dashboard → Logs

gunicorn doğru çalışıyor mu?

Timeout hatası
NowGoal sitesi yavaş olabilir

Render Free plan limitine takılmış olabilir

Import hatası
requirements.txt eksik paket var mı?

Python 3.9+ kullanıldığından emin ol

📞 DESTEK
Herhangi bir sorun için GitHub Issues kullanın.

📜 LİSANS
MIT License - İstediğiniz gibi kullanabilirsiniz.
