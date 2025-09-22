# download_covers.py
import csv
import os
import time
import requests
from io import BytesIO
from PIL import Image
from urllib.parse import quote_plus
import re

CSV_IN = "books.csv"
CSV_OUT = "books_with_covers.csv"
COVERS_DIR = os.path.join("static", "covers")
PLACEHOLDER = os.path.join(COVERS_DIR, "placeholder.jpg")
IMG_SIZE = (300, 430)
SLEEP_BETWEEN = 0.25

os.makedirs(COVERS_DIR, exist_ok=True)

def slugify(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    s = re.sub(r"^_|_$", "", s)
    return s or "book"

def save_image(pil_img, path):
    pil_img = pil_img.convert("RGB")
    pil_img.thumbnail(IMG_SIZE, Image.LANCZOS)
    pil_img.save(path, format="JPEG", quality=85, optimize=True)

def fetch_cover_by_isbn(isbn):
    if not isbn:
        return None
    url = f"https://covers.openlibrary.org/b/isbn/{quote_plus(isbn)}-L.jpg"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.headers.get("content-type","").startswith("image"):
            return Image.open(BytesIO(r.content))
    except Exception:
        return None
    return None

def fetch_cover_by_title_search(title):
    if not title:
        return None
    try:
        r = requests.get("https://openlibrary.org/search.json", params={"title": title}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        docs = data.get("docs") or []
        for d in docs:
            isbns = d.get("isbn") or []
            # try first few isbns
            for isbn in isbns[:3]:
                img = fetch_cover_by_isbn(isbn)
                if img:
                    return img
    except Exception:
        return None
    return None

def ensure_placeholder():
    if os.path.exists(PLACEHOLDER):
        return
    img = Image.new("RGB", IMG_SIZE, (30, 40, 50))
    save_image(img, PLACEHOLDER)

def main():
    ensure_placeholder()
    rows_out = []
    with open(CSV_IN, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            title = (row.get('title') or "").strip()
            author = (row.get('author') or "").strip()
            # genres in your CSV are comma-separated; keep them
            genres = (row.get('genres') or "").strip()
            # create a safe filename: {id}_{slug}.jpg
            slug = slugify(title)[:40]
            fname = f"{idx}_{slug}.jpg"
            out_path = os.path.join(COVERS_DIR, fname)

            if os.path.exists(out_path):
                print(f"[{idx}] exists -> {fname}")
                cover_url = f"/static/covers/{fname}"
            else:
                # Try to fetch by title search (no ISBN column in your CSV)
                print(f"[{idx}] searching cover for: {title}")
                img = fetch_cover_by_title_search(title)
                time.sleep(SLEEP_BETWEEN)
                if img:
                    try:
                        save_image(img, out_path)
                        cover_url = f"/static/covers/{fname}"
                        print(f"[{idx}] saved cover -> {fname}")
                    except Exception as e:
                        print(f"[{idx}] failed save: {e}; using placeholder")
                        cover_url = "/static/covers/placeholder.jpg"
                else:
                    print(f"[{idx}] no cover found; using placeholder")
                    cover_url = "/static/covers/placeholder.jpg"

            # compose output row with cover path and id
            out_row = {
                "id": str(idx),
                "title": title,
                "author": author,
                "description": row.get('description') or "",
                # keep same delimiter as input (comma-separated); you can change to pipe if you prefer
                "genres": genres,
                "cover": cover_url
            }
            rows_out.append(out_row)

    # write augmented CSV
    fieldnames = ["id", "title", "author", "description", "genres", "cover"]
    with open(CSV_OUT, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print("Done. Wrote", CSV_OUT, "and saved images to", COVERS_DIR)

if __name__ == "__main__":
    main()
