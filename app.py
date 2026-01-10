from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get("/health")
def health():
    return "ok", 200

@app.post("/analiz_et")
def analiz_et():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"skor": "-", "detay": "URL boş geldi"}), 400

    # Şimdilik örnek cevap. Buraya kendi analiz kodunu koyacağız.
    return jsonify({
        "skor": "TEST",
        "detay": f"Gelen link: {url}"
    }), 200
