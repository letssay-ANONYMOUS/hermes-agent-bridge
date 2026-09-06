#!/usr/bin/env python3
"""Generate local speech with Kokoro.

This script intentionally runs in Kokoro's own virtualenv so the voice-room
server does not need to import torch/Kokoro into its long-running process.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _as_numpy(audio):
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return audio


def main() -> int:
    parser = argparse.ArgumentParser(description="Kokoro text-to-speech runner")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--lang", default="a")
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()

    text = " ".join(args.text.split())
    if not text:
        raise SystemExit("No text was provided")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=args.lang)
    chunks = []
    for _graphemes, _phonemes, audio in pipeline(text, voice=args.voice):
        chunks.append(_as_numpy(audio))

    if not chunks:
        raise SystemExit("Kokoro returned no audio")

    waveform = np.concatenate(chunks)
    sf.write(str(output), waveform, args.sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
