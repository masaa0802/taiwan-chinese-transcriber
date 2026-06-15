import os
import re
import sys
import json
import tempfile
import threading
import subprocess
import webbrowser
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer

# Check dependencies
try:
    import whisper
except ImportError:
    print("Installing whisper...")
    subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper", "--break-system-packages", "-q"])
    import whisper

OLLAMA_URL = "http://localhost:11434"
model_cache = {}

def load_whisper_model(size="base"):
    if size not in model_cache:
        print(f"Loading Whisper model: {size}")
        model_cache[size] = whisper.load_model(size)
    return model_cache[size]

def get_gdrive_direct_url(url):
    """Convert Google Drive share URL to direct download URL"""
    # https://drive.google.com/file/d/FILE_ID/view
    m = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    # https://drive.google.com/open?id=FILE_ID
    m = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    return url

def download_video(url, dest_path):
    """Download video from URL"""
    if "drive.google.com" in url:
        url = get_gdrive_direct_url(url)
    
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    with urllib.request.urlopen(req, timeout=120) as response:
        with open(dest_path, 'wb') as f:
            while True:
                chunk = response.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                f.write(chunk)

def fmt_time(t):
    m = int(t // 60)
    s = t % 60
    return f"{m:02d}:{s:06.3f}"

def has_kana(text):
    return any('ぁ' <= c <= 'ゟ' or '゠' <= c <= 'ヿ' for c in text)

def transcribe_mixed(file_path, model):
    """中国語・日本語それぞれで文字起こしし、セグメントごとに最適な方を選択する"""
    print("混合モード: 中国語で文字起こし中...")
    result_zh = model.transcribe(file_path, language="zh", verbose=False)
    print("混合モード: 日本語で文字起こし中...")
    result_ja = model.transcribe(file_path, language="ja", verbose=False)

    ja_segs = result_ja["segments"]
    segments = []

    for zh_seg in result_zh["segments"]:
        zh_start = zh_seg["start"]
        zh_end   = zh_seg["end"]
        zh_text  = zh_seg["text"].strip()

        # 時間的に最も重なるjaセグメントを探す
        best_ja_text = ""
        best_overlap = 0
        for ja_seg in ja_segs:
            overlap = max(0, min(zh_end, ja_seg["end"]) - max(zh_start, ja_seg["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ja_text = ja_seg["text"].strip()

        # ja出力にかなが含まれ、zh出力にかなが含まれない → 日本語セグメント
        if has_kana(best_ja_text) and not has_kana(zh_text):
            text, lang = best_ja_text, "ja"
        else:
            text, lang = zh_text, "zh"

        if text:
            segments.append({
                "start": fmt_time(zh_start),
                "end":   fmt_time(zh_end),
                "text":  text,
                "lang":  lang,
            })

    return segments

def transcribe_audio(file_path, language="zh"):
    model = load_whisper_model("base")
    if language == "mixed":
        return transcribe_mixed(file_path, model)

    result = model.transcribe(file_path, language=language, verbose=False)
    return [
        {
            "start": fmt_time(seg["start"]),
            "end":   fmt_time(seg["end"]),
            "text":  seg["text"].strip(),
        }
        for seg in result["segments"]
    ]

def analyze_with_ollama(transcript_text, model="qwen2.5:7b"):
    prompt = f"""以下は動画の中国語（台湾の中国語・繁体字圏）の音声認識による文字起こしです。誤認識が含まれている可能性があります。

【文字起こし】
{transcript_text}

以下の3点を日本語で回答してください：

## 1. 修正・添削
誤認識の可能性がある箇所を指摘し、正しい表現を提案してください。各修正について「元の表現」「修正案」「理由」を示してください。

## 2. 台湾中国語の特徴・解説
台湾特有の表現、口語表現、または大陸中国語との違いがあれば解説してください。

## 3. 日本語訳
全体の自然な日本語訳を提供してください。"""

    data = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as res:
        result = json.loads(res.read())
        return result.get("response", "")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default logging

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/health':
            self.send_json({"status": "ok"})
        elif parsed.path == '/video_proxy':
            params = urllib.parse.parse_qs(parsed.query)
            url = params.get('url', [''])[0]
            if not url:
                self.send_json({"error": "url required"}, 400)
                return
            self.proxy_video(url)
        else:
            self.send_json({"error": "not found"}, 404)

    def proxy_video(self, url):
        if "drive.google.com" in url:
            url = get_gdrive_direct_url(url)
        range_header = self.headers.get('Range', '')
        req_headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        if range_header:
            req_headers['Range'] = range_header
        req = urllib.request.Request(url, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = 206 if range_header and resp.headers.get('Content-Range') else 200
                self.send_response(status)
                self.send_header('Content-Type', resp.headers.get('Content-Type', 'video/mp4'))
                cl = resp.headers.get('Content-Length', '')
                if cl:
                    self.send_header('Content-Length', cl)
                cr = resp.headers.get('Content-Range', '')
                if cr:
                    self.send_header('Content-Range', cr)
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except Exception as e:
            print(f"Proxy error: {e}")
            try:
                self.send_response(502)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(str(e).encode())
            except Exception:
                pass

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        
        if self.path == '/transcribe_url':
            body = json.loads(self.rfile.read(content_length))
            url = body.get("url", "")
            language = body.get("language", "zh")
            
            if not url:
                self.send_json({"error": "URL が必要です"}, 400)
                return
            
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name
                print(f"Downloading: {url}")
                download_video(url, tmp_path)
                print(f"Transcribing...")
                segments = transcribe_audio(tmp_path, language)
                os.unlink(tmp_path)
                self.send_json({"segments": segments})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        elif self.path == '/transcribe_file':
            # Multipart form data
            content_type = self.headers.get('Content-Type', '')
            boundary = content_type.split('boundary=')[-1].encode()
            raw = self.rfile.read(content_length)
            
            # Extract file data from multipart
            parts = raw.split(b'--' + boundary)
            file_data = None
            language = "zh"
            
            for part in parts:
                if b'filename=' in part:
                    header_end = part.find(b'\r\n\r\n')
                    if header_end != -1:
                        file_data = part[header_end + 4:].rstrip(b'\r\n')
                if b'name="language"' in part:
                    header_end = part.find(b'\r\n\r\n')
                    if header_end != -1:
                        language = part[header_end + 4:].rstrip(b'\r\n').decode()
            
            if not file_data:
                self.send_json({"error": "ファイルが見つかりません"}, 400)
                return
            
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name
                segments = transcribe_audio(tmp_path, language)
                os.unlink(tmp_path)
                self.send_json({"segments": segments})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        elif self.path == '/analyze':
            body = json.loads(self.rfile.read(content_length))
            text = body.get("text", "")
            model = body.get("model", "qwen2.5:7b")
            try:
                result = analyze_with_ollama(text, model)
                self.send_json({"result": result})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
        else:
            self.send_json({"error": "not found"}, 404)

if __name__ == "__main__":
    PORT = 8765
    print(f"🎙 台湾中国語 文字起こしサーバー起動中...")
    print(f"📡 http://localhost:{PORT}")
    print(f"Whisper モデルを事前ロード中...")
    load_whisper_model("base")
    print(f"✅ 準備完了！ブラウザでindex.htmlを開いてください。")
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open(f"file://{index_path}")
    threading.Thread(target=open_browser, daemon=True).start()
    server = ThreadingHTTPServer(('localhost', PORT), Handler)
    server.serve_forever()
