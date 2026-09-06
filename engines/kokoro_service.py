#!/usr/bin/env python3
"""Small localhost Kokoro service.

Runs Kokoro warm in memory and returns WAV bytes for each synth request.
No provider keys, no external API. Intended to bind only to 127.0.0.1.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import soundfile as sf
from kokoro import KPipeline


class KokoroEngine:
    def __init__(self, lang: str) -> None:
        self.lang = lang
        self.pipeline = KPipeline(lang_code=lang)
        self.lock = threading.Lock()

    def synthesize(self, text: str, voice: str, sample_rate: int) -> bytes:
        text = " ".join(text.split())
        if not text:
            raise ValueError("No text was provided")

        with self.lock:
            chunks = []
            for _graphemes, _phonemes, audio in self.pipeline(text, voice=voice):
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                chunks.append(audio)

        if not chunks:
            raise RuntimeError("Kokoro returned no audio")

        waveform = np.concatenate(chunks)
        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV")
        return buffer.getvalue()


def build_handler(engine: KokoroEngine, default_voice: str, sample_rate: int):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HermesKokoro/1.0"

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            self._json({"ok": True, "voice": default_voice, "lang": engine.lang, "sample_rate": sample_rate})

        def do_POST(self) -> None:
            if self.path != "/synthesize":
                self.send_error(404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                text = str(payload.get("text") or "")
                voice = str(payload.get("voice") or default_voice)
                lang = str(payload.get("lang") or engine.lang)
                if lang != engine.lang:
                    raise ValueError(f"Service was started for lang={engine.lang!r}, got {lang!r}")
                wav = engine.synthesize(text=text, voice=voice, sample_rate=sample_rate)
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, status=500)
                return

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes Kokoro localhost TTS service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--lang", default="a")
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()

    engine = KokoroEngine(lang=args.lang)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        build_handler(engine, default_voice=args.voice, sample_rate=args.sample_rate),
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
