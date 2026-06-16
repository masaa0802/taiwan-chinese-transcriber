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
    from faster_whisper import WhisperModel
except ImportError:
    print("Installing faster-whisper...")
    subprocess.run([sys.executable, "-m", "pip", "install", "faster-whisper", "--break-system-packages", "-q"])
    from faster_whisper import WhisperModel

OLLAMA_URL = "http://localhost:11434"
model_cache = {}

def load_whisper_model(size="medium"):
    if size not in model_cache:
        print(f"Loading Whisper model: {size} (faster-whisper / int8)")
        model_cache[size] = WhisperModel(size, device="cpu", compute_type="int8")
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

def kana_ratio(text):
    if not text: return 0.0
    kana = sum(1 for c in text if 'ぁ' <= c <= 'ゟ' or '゠' <= c <= 'ヿ')
    return kana / len(text)

def hanzi_ratio(text):
    if not text: return 0.0
    hanzi = sum(1 for c in text if '一' <= c <= '鿿')
    return hanzi / len(text)

def detect_text_language(text):
    """文字種からセグメントの言語を推定する"""
    kana   = sum(1 for c in text if 'ぁ' <= c <= 'ゟ' or '゠' <= c <= 'ヿ')
    hanzi  = sum(1 for c in text if '一' <= c <= '鿿')
    hangul = sum(1 for c in text if '가' <= c <= '힣')
    latin  = sum(1 for c in text if c.isalpha() and ord(c) < 256)
    if kana   > 0: return "ja"
    if hangul > 0: return "ko"
    if hanzi  > 0: return "zh"
    if latin  > 0: return "en"
    return "other"

TRANSCRIBE_OPTS = dict(
    beam_size=5,
    vad_filter=True,
    vad_parameters={
        "min_silence_duration_ms": 500,
        "threshold": 0.3,          # デフォルト0.5→小音量の音声も検出
        "speech_pad_ms": 400,      # 音声区間の前後にパディング
    },
    condition_on_previous_text=False,
    no_speech_threshold=0.3,       # デフォルト0.6→小音量セグメントを弾かない
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.5,       # デフォルト-1.0→不確かなセグメントも保持
)

def transcribe_mixed(file_path, whisper_model):
    """中国語・日本語それぞれで文字起こしし、セグメントごとに最適な方を選択する"""
    model = load_whisper_model(whisper_model)
    print("混合モード: 中国語で文字起こし中...")
    zh_gen, _ = model.transcribe(file_path, language="zh", **TRANSCRIBE_OPTS)
    zh_segs = list(zh_gen)
    print("混合モード: 日本語で文字起こし中...")
    ja_gen, _ = model.transcribe(file_path, language="ja", **TRANSCRIBE_OPTS)
    ja_segs = list(ja_gen)

    segments = []
    matched_ja = set()

    for zh_seg in zh_segs:
        zh_start = zh_seg.start
        zh_end   = zh_seg.end
        zh_text  = zh_seg.text.strip()

        best_ja_idx  = -1
        best_ja_text = ""
        best_overlap = 0
        for i, ja_seg in enumerate(ja_segs):
            overlap = max(0, min(zh_end, ja_seg.end) - max(zh_start, ja_seg.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ja_text = ja_seg.text.strip()
                best_ja_idx  = i

        if best_ja_idx >= 0 and best_overlap > 0:
            matched_ja.add(best_ja_idx)

        ja_score = kana_ratio(best_ja_text)
        zh_score = hanzi_ratio(zh_text)
        # かな比率が30%以上 かつ 漢字比率より高い場合のみ日本語と判定
        if ja_score >= 0.3 and ja_score > zh_score:
            auto_lang, auto_text = "ja", best_ja_text
        else:
            auto_lang, auto_text = "zh", zh_text

        if zh_text or best_ja_text:
            segments.append({
                "_start":  zh_start,
                "start":   fmt_time(zh_start),
                "end":     fmt_time(zh_end),
                "text":    auto_text,
                "text_zh": zh_text,
                "text_ja": best_ja_text,
                "lang":    auto_lang,
            })

    for i, ja_seg in enumerate(ja_segs):
        if i in matched_ja:
            continue
        ja_text = ja_seg.text.strip()
        if not ja_text:
            continue
        segments.append({
            "_start":  ja_seg.start,
            "start":   fmt_time(ja_seg.start),
            "end":     fmt_time(ja_seg.end),
            "text":    ja_text,
            "text_zh": "",
            "text_ja": ja_text,
            "lang":    "ja",
        })

    segments.sort(key=lambda s: s["_start"])
    for s in segments:
        del s["_start"]
    return segments

def transcribe_audio(file_path, language="zh", whisper_model="medium"):
    if language == "auto":
        # 中国語・日本語を両方パスして文字種で言語を判別する
        return transcribe_mixed(file_path, whisper_model)

    model = load_whisper_model(whisper_model)
    segments, _ = model.transcribe(file_path, language=language, **TRANSCRIBE_OPTS)
    result = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        result.append({
            "start": fmt_time(seg.start),
            "end":   fmt_time(seg.end),
            "text":  text,
        })
    return result

def analyze_with_ollama(transcript_text, model="qwen2.5:7b"):
    prompt = f"""以下は動画の中国語（台湾繁体字）の音声認識による文字起こしです。音声認識の誤りが含まれている可能性があります。

【文字起こし】
{transcript_text}

以下の手順で日本語で回答してください。

## 1. 修正・添削

文字起こし全文を最初から最後まで通して読み、以下の観点で問題箇所をすべて洗い出してください。

**確認すべき観点（優先順位順）：**
1. **単語・フレーズ自体が不自然**：その言葉が中国語（台湾語）として存在するか、自然な表現かを確認する。不自然な場合は「音が近い別の単語の誤認識」として正しい候補を提案する。
2. **文脈から見て意味がおかしい**：前後の文脈と照らして意味が通らない場合、音が似た正しい表現を推定して提案する。
3. **文法・語順・助詞の誤り**：文法的に正しくない場合のみ、構造の修正を提案する。

**重要：** 単語や表現が不自然・不存在なのに「文法が不完全」とだけ説明するのは不十分です。「その表現自体がおかしい」ことを明確に指摘し、音声認識が何を誤認したかを具体的に推定してください。

発見した誤りを以下の表形式でまとめてください：

| # | 元の表現 | 修正案 | 指摘内容（単語の誤認識 / 意味の不整合 / 文法ミス） |
|---|---------|--------|------|

修正がない場合は「修正箇所なし」と記載してください。

次に、上記の修正をすべて反映した**修正済み全文**を出力してください。

## 2. 台湾中国語の特徴・解説
台湾特有の表現、口語表現、または大陸中国語との違いがあれば解説してください。

## 3. 日本語訳
修正済み全文の自然な日本語訳を提供してください。"""

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
            url           = body.get("url", "")
            language      = body.get("language", "zh")
            whisper_model = body.get("whisper_model", "small")

            if not url:
                self.send_json({"error": "URL が必要です"}, 400)
                return

            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = tmp.name
                print(f"Downloading: {url}")
                download_video(url, tmp_path)
                print(f"Transcribing with whisper:{whisper_model} lang:{language}")
                segments = transcribe_audio(tmp_path, language, whisper_model)
                os.unlink(tmp_path)
                self.send_json({"segments": segments})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        elif self.path == '/transcribe_file':
            content_type = self.headers.get('Content-Type', '')
            boundary = content_type.split('boundary=')[-1].encode()
            raw = self.rfile.read(content_length)

            parts = raw.split(b'--' + boundary)
            file_data     = None
            language      = "zh"
            whisper_model = "small"

            for part in parts:
                if b'filename=' in part:
                    header_end = part.find(b'\r\n\r\n')
                    if header_end != -1:
                        file_data = part[header_end + 4:].rstrip(b'\r\n')
                if b'name="language"' in part and b'filename=' not in part:
                    header_end = part.find(b'\r\n\r\n')
                    if header_end != -1:
                        language = part[header_end + 4:].rstrip(b'\r\n').decode()
                if b'name="whisper_model"' in part:
                    header_end = part.find(b'\r\n\r\n')
                    if header_end != -1:
                        whisper_model = part[header_end + 4:].rstrip(b'\r\n').decode()

            if not file_data:
                self.send_json({"error": "ファイルが見つかりません"}, 400)
                return

            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name
                print(f"Transcribing with whisper:{whisper_model} lang:{language}")
                segments = transcribe_audio(tmp_path, language, whisper_model)
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
    print(f"Whisper モデルを事前ロード中... (medium / faster-whisper)")
    load_whisper_model("medium")
    print(f"✅ 準備完了！ブラウザでindex.htmlを開いてください。")
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    def open_browser():
        import time
        time.sleep(1)
        webbrowser.open(f"file://{index_path}")
    threading.Thread(target=open_browser, daemon=True).start()
    server = ThreadingHTTPServer(('localhost', PORT), Handler)
    server.serve_forever()
