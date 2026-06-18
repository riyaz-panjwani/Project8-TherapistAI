"""Download GloVe 6B 100d vectors from Stanford NLP.

Usage:
    python training/scripts/download_glove.py

Saves to: training/data/glove.6B.100d.txt  (~330 MB)
"""
import urllib.request
import zipfile
from pathlib import Path

URL   = "https://nlp.stanford.edu/data/glove.6B.zip"
DEST  = Path("training/data")
FNAME = "glove.6B.100d.txt"


def download():
    DEST.mkdir(parents=True, exist_ok=True)
    out_file = DEST / FNAME
    if out_file.exists():
        print(f"Already exists: {out_file}")
        return

    zip_path = DEST / "glove.6B.zip"
    if not zip_path.exists():
        print(f"Downloading GloVe 6B (~820 MB zip)…")
        def _progress(b, bsize, total):
            pct = min(b * bsize / total * 100, 100)
            print(f"\r  {pct:.1f}%", end="", flush=True)
        urllib.request.urlretrieve(URL, zip_path, _progress)
        print()

    print(f"Extracting {FNAME}…")
    with zipfile.ZipFile(zip_path) as z:
        z.extract(FNAME, DEST)
    print(f"Done → {out_file}")


if __name__ == "__main__":
    download()
