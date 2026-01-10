from flask import Flask, request, jsonify
from urllib.parse import urlparse
import ipaddress
import socket
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# 1) Hangi sitelere izin veriyoruz? (Güvenlik için)
ALLOWED_HOSTS = {
    "live3.nowgoal26.com",
    "nowgoal26.com",
    "www.nowgoal26.com",
}

def is_safe_url(url: str) -> (bool, str):
    """Basit güvenlik: sadece izinli domain + iç ağ IP engeli."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False, "URL http/https ile başlamalı"
        if not p.hostname:
            return False, "URL içinde domain yok"

        host = p.hostname.lower()

        if host not in ALLOWED_HOSTS:
            return False, f"Bu siteye izin yok: {host}"

        # Domain'i IP'ye çevirip yerel ağ/localhost engelle
        ip_str = socket.gethostbyname(host)
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
            return False, "Güvenlik: yerel ağ/özel IP engellendi"

        return True, "ok"
    except Exception as e:
        return False, f"URL kontrol hatası: {e}"

@app.get("/")
def home():
    return "macanalizor-api is running", 200

@app.get("/health")
def health():
    return "ok", 200

@app.post("/analiz_et")
def analiz_et():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"skor": "-", "detay": "URL boş geldi"}), 400

    ok, msg = is_safe_url(url)
    if not ok:
        return jsonify({"skor": "-", "detay": msg}), 400

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (MacanalizorBot/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return jsonify({
                "skor": "-",
                "detay": f"Site cevap vermedi. HTTP: {r.status_code}"
            }), 502

        html = r.text
        soup = BeautifulSoup(html, "lxml")

        # Basit örnek: sayfa başlığını yakalayalım
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        if not title:
            title = "Başlık bulunamadı (sayfa JS ile yükleniyor olabilir)"

        # Şimdilik skor alanına "OK" yazıp, detayda başlığı gösterelim
        return jsonify({
            "skor": "OK",
            "detay": f"Sayfa başlığı: {title}"
        }), 200

    except requests.exceptions.Timeout:
        return jsonify({"skor": "-", "detay": "Zaman aşımı (timeout). Site geç yanıt verdi."}), 504
    except Exception as e:
        return jsonify({"skor": "-", "detay": f"Hata: {e}"}), 500
