"""Convert every file in raw/ to 16 kHz mono WAV in converted/ and print a sanity report."""
import hashlib
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent
RAW, OUT = ROOT / "raw", ROOT / "converted"
SR = 16000
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

OUT.mkdir(exist_ok=True)
rows = []
for f in sorted(RAW.iterdir()):
    out = OUT / (f.stem + ".wav")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(f),
                    "-vn", "-ac", "1", "-ar", str(SR), str(out)], check=True)
    y, sr = sf.read(out)
    rows.append((out.name, len(y) / sr, float(np.abs(y).max()),
                 hashlib.md5(y.tobytes()).hexdigest()[:8]))

print(f"{'file':38s}{'sec':>7s}{'peak':>8s}  hash")
for n, d, p, h in rows:
    print(f"{n:38s}{d:7.2f}{p:8.3f}  {h}")
hashes = [r[3] for r in rows]
print("\nDuplicate audio detected!" if len(set(hashes)) < len(hashes) else "\nAll recordings are different.")
