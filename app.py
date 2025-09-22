from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles   # <-- added
from sentence_transformers import SentenceTransformer
import numpy as np
import uvicorn
import pandas as pd
import os
import re

app = FastAPI(title="Quick Book Recommender")

# ✅ Serve static files (for covers)
app.mount("/static", StaticFiles(directory="static"), name="static")

def load_books_from_csv(csv_path="books.csv"):
    """
    Load books from CSV file. Expected columns:
    - title, author, description, genres
    Covers will be mapped to static/covers/{id}.jpg
    """
    try:
        df = pd.read_csv(csv_path)
        books = []
        for idx, row in df.iterrows():
            genres_str = str(row.get('genres', '')).strip()
            if genres_str and genres_str.lower() != 'nan':
                genres = [g.strip() for g in genres_str.split(',') if g.strip()]
            else:
                genres = []

            book_id = idx + 1
            cover_path = f"/static/covers/{book_id}.jpg"  # ✅ auto-link cover by id

            book = {
                "id": book_id,
                "title": str(row.get('title', 'Unknown Title')).strip(),
                "author": str(row.get('author', 'Unknown Author')).strip(),
                "description": str(row.get('description', 'No description available')).strip(),
                "genres": genres,
                "cover": cover_path   # ✅ include cover path
            }
            books.append(book)

        print(f"Loaded {len(books)} books from {csv_path}")
        return books
    except FileNotFoundError:
        print(f"CSV file {csv_path} not found. Using default book list.")
        return get_default_books()
    except Exception as e:
        print(f"Error loading CSV: {e}. Using default book list.")
        return get_default_books()


def get_default_books():
    """Fallback to the original hardcoded book list"""
    return [
        {"id": 1, "title": "Dune", "author": "Frank Herbert",
         "description": "Epic science fiction about politics, religion, and desert planet Arrakis.", "genres": ["Science Fiction","Adventure"]},
        {"id": 2, "title": "Neuromancer", "author": "William Gibson",
         "description": "Cyberpunk classic; a washed-up hacker is hired for one last job.", "genres": ["Science Fiction","Cyberpunk"]},
        {"id": 3, "title": "The Hobbit", "author": "J.R.R. Tolkien",
         "description": "A reluctant hobbit goes on an adventure with dwarves to reclaim treasure.", "genres": ["Fantasy","Adventure"]},
        {"id": 4, "title": "1984", "author": "George Orwell",
         "description": "Dystopian novel about surveillance, totalitarianism and truth control.", "genres": ["Dystopia","Political Fiction"]},
        {"id": 5, "title": "Pride and Prejudice", "author": "Jane Austen",
         "description": "A witty social commentary and romance centered on Elizabeth Bennet.", "genres": ["Romance","Classic"]},
        {"id": 6, "title": "The Martian", "author": "Andy Weir",
         "description": "A stranded astronaut uses engineering and humor to survive on Mars.", "genres": ["Science Fiction","Survival"]},
        {"id": 7, "title": "Foundation", "author": "Isaac Asimov",
         "description": "A mathematician predicts the fall of an empire and starts the Foundation project.", "genres": ["Science Fiction","Epic"]},
        {"id": 8, "title": "The Name of the Wind", "author": "Patrick Rothfuss",
         "description": "A gifted young man grows into a legendary figure; lyrical fantasy.", "genres": ["Fantasy","Epic"]},
        {"id": 9, "title": "Sapiens", "author": "Yuval Noah Harari",
         "description": "A brief history of humankind exploring cognitive, agricultural, and scientific revolutions.", "genres": ["Nonfiction","History"]},
        {"id": 10, "title": "The Left Hand of Darkness", "author": "Ursula K. Le Guin",
         "description": "An envoy visits a planet with ambisexual inhabitants; explores society and politics.", "genres": ["Science Fiction","Social Commentary"]},
        {"id": 11, "title": "Frankenstein", "author": "Mary Shelley",
         "description": "A scientist creates life and faces the consequences; early science fiction and gothic novel.", "genres": ["Classic","Gothic"]},
        {"id": 12, "title": "The Catcher in the Rye", "author": "J.D. Salinger",
         "description": "A coming-of-age story about teenage alienation and angst.", "genres": ["Classic","Coming-of-age"]},
        # Add more default books to expand your dataset
        {"id": 13, "title": "The Great Gatsby", "author": "F. Scott Fitzgerald",
         "description": "A critique of the American Dream set in the Jazz Age.", "genres": ["Classic","Romance"]},
        {"id": 14, "title": "To Kill a Mockingbird", "author": "Harper Lee",
         "description": "A story of racial injustice and childhood innocence in the American South.", "genres": ["Classic","Social Commentary"]},
        {"id": 15, "title": "The Hunger Games", "author": "Suzanne Collins",
         "description": "A dystopian tale of survival and rebellion in a totalitarian society.", "genres": ["Dystopia","Adventure","Young Adult"]},
        {"id": 16, "title": "Harry Potter and the Sorcerer's Stone", "author": "J.K. Rowling",
         "description": "A young wizard discovers his magical heritage and attends Hogwarts.", "genres": ["Fantasy","Adventure","Young Adult"]},
        {"id": 17, "title": "The Girl with the Dragon Tattoo", "author": "Stieg Larsson",
         "description": "A journalist and hacker investigate a wealthy family's dark secrets.", "genres": ["Mystery","Thriller"]},
        {"id": 18, "title": "Gone Girl", "author": "Gillian Flynn",
         "description": "A psychological thriller about a marriage gone wrong.", "genres": ["Thriller","Mystery","Psychological"]},
        {"id": 19, "title": "The Fault in Our Stars", "author": "John Green",
         "description": "A love story between two teenagers with cancer.", "genres": ["Romance","Young Adult","Contemporary Fiction"]},
        {"id": 20, "title": "Educated", "author": "Tara Westover",
         "description": "A memoir about education, family, and the struggle for self-invention.", "genres": ["Nonfiction","Memoir","Biography"]},
        {"id": 21, "title": "The Silent Patient", "author": "Alex Michaelides",
         "description": "A psychotherapist's obsession with treating a woman who refuses to speak.", "genres": ["Thriller","Mystery","Psychological"]},
        {"id": 22, "title": "Where the Crawdads Sing", "author": "Delia Owens",
         "description": "A coming-of-age story set in the marshlands of North Carolina.", "genres": ["Mystery","Coming-of-age","Romance"]},
        {"id": 23, "title": "The Seven Husbands of Evelyn Hugo", "author": "Taylor Jenkins Reid",
         "description": "A reclusive Hollywood icon tells her life story to an unknown journalist.", "genres": ["Romance","Historical Fiction","LGBTQ+"]},
        {"id": 24, "title": "Atomic Habits", "author": "James Clear",
         "description": "A guide to building good habits and breaking bad ones.", "genres": ["Nonfiction","Self-Help","Psychology"]},
        {"id": 25, "title": "The Midnight Library", "author": "Matt Haig",
         "description": "A woman explores alternate versions of her life in a magical library.", "genres": ["Fantasy","Contemporary Fiction","Philosophy"]}
    ]

# Load books (will try CSV first, fallback to default)
BOOKS = load_books_from_csv()

MODEL_NAME = "all-MiniLM-L6-v2"   # small & fast
model = SentenceTransformer(MODEL_NAME)

def build_corpus_embeddings(books):
    texts = [b["title"] + " " + b["author"] + " " + b["description"] for b in books]
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    embs = embs / norms
    return embs

# Update the corpus embeddings and lookup dictionaries to use loaded books
CORPUS_EMBS = build_corpus_embeddings(BOOKS)
_TITLE_LOOKUP = {b['title'].lower(): i for i, b in enumerate(BOOKS)}
ALL_GENRES = sorted({g for b in BOOKS for g in b.get('genres', [])})
NORMALIZED_GENRES = [g.lower() for g in ALL_GENRES]

# synonym map (extend as needed)
GENRE_SYNONYMS = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "sf": "science fiction",
    "sciencefiction": "science fiction",
    "romantic": "romance",
    "romance": "romance",
    "dystopian": "dystopia",
    "non-fiction": "nonfiction",
    "nonfiction": "nonfiction",
    # add more synonyms if you like
}

def _normalize_genre_token(tok):
    t = tok.lower().strip()
    t_simple = re.sub(r'[^a-z0-9]', '', t)  # remove punctuation like "sci-fi" -> scifi
    return GENRE_SYNONYMS.get(t, GENRE_SYNONYMS.get(t_simple, t))

def _find_title_mentioned(query):
    q = (query or "").lower()
    for title, idx in _TITLE_LOOKUP.items():
        if title in q:
            return idx
    return None

def _book_has_genre(book, detected_genres):
    if not detected_genres:
        return False
    book_genres_norm = [g.lower() for g in book.get('genres', [])]
    return any(dg in book_genres_norm for dg in detected_genres)

def _detect_genres_from_query_improved(query):
    """
    Improved genre detection with better matching
    """
    q = query.lower().strip()
    detected = set()
    
    # Create a comprehensive genre mapping
    genre_mapping = {
        # Direct mappings
        "science fiction": "science fiction",
        "sci-fi": "science fiction", 
        "scifi": "science fiction",
        "sf": "science fiction",
        "fantasy": "fantasy",
        "romance": "romance",
        "romantic": "romance",
        "dystopia": "dystopia",
        "dystopian": "dystopia",
        "nonfiction": "nonfiction",
        "non-fiction": "nonfiction",
        "non fiction": "nonfiction",
        "classic": "classic",
        "classics": "classic",
        "adventure": "adventure",
        "cyberpunk": "cyberpunk",
        "gothic": "gothic",
        "history": "history",
        "historical": "history",
        "epic": "epic",
        "survival": "survival",
        "political": "political fiction",
        "politics": "political fiction",
        "coming-of-age": "coming-of-age",
        "coming of age": "coming-of-age",
        "social commentary": "social commentary",
        "thriller": "thriller",
        "mystery": "mystery",
        "psychological": "psychological",
        "young adult": "young adult",
        "ya": "young adult",
        "contemporary": "contemporary fiction",
        "memoir": "memoir",
        "biography": "biography",
        "self-help": "self-help",
        "philosophy": "philosophy"
    }
    
    # Check for direct matches first (whole query)
    if q in genre_mapping:
        detected.add(genre_mapping[q])
        return list(detected)
    
    # Check for substring matches
    for genre_term, canonical_genre in genre_mapping.items():
        if genre_term in q:
            detected.add(canonical_genre)
    
    # If no matches found, try word-by-word matching
    if not detected:
        words = q.split()
        for word in words:
            word = word.strip('.,!?;:"()[]{}')  # Remove punctuation
            if word in genre_mapping:
                detected.add(genre_mapping[word])
    
    return list(detected)

def recommend_by_text(query_text: str, k: int = 5):
    """
    Improved recommendation logic:
    - Direct genre matching gets highest priority
    - If user mentions a book title -> find similar books
    - Otherwise: use embedding similarity
    """
    q = (query_text or "").strip()
    if not q:
        return []

    # 1) Check if user mentioned a specific book title
    title_idx = _find_title_mentioned(q)
    if title_idx is not None:
        # Use the book's embedding as the query vector
        q_emb = CORPUS_EMBS[title_idx]
        sims = (CORPUS_EMBS @ q_emb)
        sims[title_idx] = -1.0  # Don't return the same book
        top_idx = np.argsort(-sims)[:k]
        # Return without similarity scores
        return [BOOKS[int(i)] for i in top_idx]

    # 2) Detect genres from query (improved detection)
    detected_genres = _detect_genres_from_query_improved(q)

    # 3) If genres detected, prioritize genre-based recommendations
    if detected_genres:
        # Find all books that match any of the detected genres
        genre_matches = []
        for i, book in enumerate(BOOKS):
            book_genres_lower = [g.lower() for g in book.get('genres', [])]
            if any(dg in book_genres_lower for dg in detected_genres):
                genre_matches.append((i, book))
        
        # If we have enough genre matches, return them (shuffled for variety)
        if len(genre_matches) >= k:
            # Sort by title for consistent ordering, then take first k
            genre_matches.sort(key=lambda x: x[1]['title'])
            return [book for _, book in genre_matches[:k]]
        elif genre_matches:
            # If we have some genre matches but not enough, fill the rest with embedding similarity
            genre_books = [book for _, book in genre_matches]
            remaining_needed = k - len(genre_books)
            
            if remaining_needed > 0:
                # Get embedding similarity for remaining slots
                q_emb = model.encode([q], convert_to_numpy=True)[0]
                q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
                sims = (CORPUS_EMBS @ q_emb)
                
                # Exclude already selected genre books
                used_indices = {i for i, _ in genre_matches}
                available_sims = [(i, sims[i]) for i in range(len(BOOKS)) if i not in used_indices]
                available_sims.sort(key=lambda x: -x[1])
                
                # Add the top remaining books
                for i, _ in available_sims[:remaining_needed]:
                    genre_books.append(BOOKS[i])
            
            return genre_books[:k]

    # 4) No genre detected and no title mentioned -> use embedding similarity
    q_emb = model.encode([q], convert_to_numpy=True)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    sims = (CORPUS_EMBS @ q_emb)
    top_idx = np.argsort(-sims)[:k]
    return [BOOKS[int(i)] for i in top_idx]

@app.get("/", response_class=HTMLResponse)
async def homepage():
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Book-Rec — Recommendations</title>
  <style>
    :root{
      --bg:#0f1724; --card:#0b1220; --muted:#9aa4b2; --accent:#7c3aed;
      --glass: rgba(255,255,255,0.03);
      --glass-2: rgba(255,255,255,0.04);
    }
    *{box-sizing:border-box}
    html,body{height:100%; margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; background:linear-gradient(180deg,#071029 0%, #0a1220 100%); color:#e6eef6}
    .container{max-width:1100px; margin:28px auto; padding:24px;}
    header{display:flex;align-items:center;gap:16px;margin-bottom:18px}
    .logo{width:56px;height:56px;border-radius:10px;background:linear-gradient(135deg,var(--accent),#4f46e5);display:flex;align-items:center;justify-content:center;font-weight:700;color:white;font-size:20px;box-shadow:0 6px 18px rgba(34,40,49,0.6)}
    h1{margin:0;font-size:1.4rem}
    p.lead{margin:6px 0 0; color:var(--muted); font-size:0.95rem}

    .card{background:linear-gradient(180deg,var(--glass), var(--glass-2)); border:1px solid rgba(255,255,255,0.03); padding:18px;border-radius:12px; box-shadow: 0 8px 30px rgba(2,6,23,0.6)}
    .search-row{display:flex;gap:12px;margin-top:18px;align-items:center}
    .search-input{flex:1;display:flex;align-items:center;gap:8px;background:transparent;border:1px solid rgba(255,255,255,0.06);padding:10px 12px;border-radius:10px}
    .search-input input{flex:1;background:transparent;border:0;color:inherit;outline:none;font-size:1rem}
    .btn{background:var(--accent);border:0;color:white;padding:10px 14px;border-radius:10px;font-weight:600;cursor:pointer;box-shadow:0 6px 18px rgba(124,58,237,0.18)}
    .btn:active{transform:translateY(1px)}
    .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
    .chip{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.02);padding:6px 10px;border-radius:999px;cursor:pointer;color:var(--muted);font-size:0.9rem}
    .chip.active{background:linear-gradient(90deg,var(--accent),#5b21b6);color:white;border:0;box-shadow:0 8px 18px rgba(79,70,229,0.12)}

    .results{margin-top:18px; display:grid; grid-template-columns: repeat(auto-fill,minmax(260px,1fr)); gap:14px}
    .result-card{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px;padding:14px;border:1px solid rgba(255,255,255,0.03); display:flex; gap:12px; align-items:flex-start}
    .cover {
  width:70px;
  height:96px;
  border-radius:6px;
  background:linear-gradient(180deg,#1f2937,#111827);
  display:flex;
  align-items:center;
  justify-content:center;
  color:var(--muted);
  font-weight:700;
  object-fit:cover;  /* <-- ensures images scale nicely */
}

    .meta{flex:1}
    .title{font-weight:700;margin:0;font-size:1rem}
    .author{margin-top:6px;color:var(--muted);font-size:0.9rem}
    .desc{margin-top:8px;color:#cbd5e1;font-size:0.9rem;line-height:1.25}
    .meta-foot{display:flex;justify-content:space-between;align-items:center;margin-top:10px}
    .genres{color:var(--muted);font-size:0.85rem}
    .score{background:rgba(255,255,255,0.04);padding:6px 8px;border-radius:8px;font-weight:700;color:#e6eef6}

    .empty{padding:32px;text-align:center;color:var(--muted)}
    .spinner{width:36px;height:36px;border-radius:50%;border:4px solid rgba(255,255,255,0.08);border-top-color:var(--accent);animation:spin 1s linear infinite;margin-left:8px}
    @keyframes spin{to{transform:rotate(360deg)}}

    footer{margin-top:18px;color:var(--muted);font-size:0.9rem;text-align:center}
    @media (max-width:640px){
      .header-row{flex-direction:column;align-items:flex-start}
      .search-row{flex-direction:column;align-items:stretch}
      .search-input{width:100%}
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">BR</div>
      <div>
        <h1>Book-Rec</h1>
        <p class="lead">Semantic book recommendations — type a mood, genre, or book you liked.</p>
      </div>
    </header>

    <div class="card">
      <div class="header-row">
        <div class="search-row">
          <div class="search-input" id="searchBox">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style="opacity:0.8">
              <path d="M21 21l-4.35-4.35" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <circle cx="11" cy="11" r="6" stroke="currentColor" stroke-width="2"/>
            </svg>
            <input id="q" placeholder="e.g. hard sci-fi with humor, or 'I liked Dune'"/>
            <div id="spinner" style="display:none" class="spinner" aria-hidden="true"></div>
          </div>
          <button class="btn" id="goBtn">Recommend</button>
        </div>

        <div class="chips" style="margin-left:12px">
          <div class="chip" data-q="Science Fiction">Science Fiction</div>
          <div class="chip" data-q="Fantasy">Fantasy</div>
          <div class="chip" data-q="Romance">Romance</div>
          <div class="chip" data-q="Dystopia">Dystopia</div>
          <div class="chip" data-q="Nonfiction">Nonfiction</div>
        </div>
      </div>

      <div id="resultsWrap">
        <div class="empty" id="emptyState">Try a query above to see recommendations.</div>
        <div class="results" id="results" style="display:none"></div>
      </div>
    </div>

    <footer>Built for a college demo · Swap dataset or plug a vector DB to scale</footer>
  </div>

<script>
  const qInput = document.getElementById('q');
  const goBtn = document.getElementById('goBtn');
  const resultsEl = document.getElementById('results');
  const emptyEl = document.getElementById('emptyState');
  const spinner = document.getElementById('spinner');
  const chips = Array.from(document.querySelectorAll('.chip'));

  chips.forEach(c=>{
    c.addEventListener('click', ()=>{
      // toggle active
      if(c.classList.contains('active')) {
        c.classList.remove('active');
        qInput.value = '';
      } else {
        chips.forEach(x=>x.classList.remove('active'));
        c.classList.add('active');
        qInput.value = c.dataset.q;
      }
      runRecommend();
    });
  });

  goBtn.addEventListener('click', runRecommend);
  qInput.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ runRecommend(); } });

  async function runRecommend(){
    const text = qInput.value.trim();
    if(!text) {
      emptyEl.textContent = "Please type a query or tap a genre chip.";
      resultsEl.style.display = 'none';
      emptyEl.style.display = 'block';
      return;
    }
    spinner.style.display = 'inline-block';
    goBtn.disabled = true;

    try {
      const res = await fetch('/recommend', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({text, k:8})
      });
      if(!res.ok) throw new Error('Server error');
      const data = await res.json();
      renderResults(data.results || []);
    } catch(err){
      console.error(err);
      emptyEl.textContent = "Something went wrong — check the server logs.";
      resultsEl.style.display = 'none';
      emptyEl.style.display = 'block';
    } finally {
      spinner.style.display = 'none';
      goBtn.disabled = false;
    }
  }

  function renderResults(items){
    resultsEl.innerHTML = '';
    if(!items || items.length===0){
      emptyEl.textContent = "No matches found. Try a broader query.";
      resultsEl.style.display = 'none';
      emptyEl.style.display = 'block';
      return;
    }
    emptyEl.style.display = 'none';
    resultsEl.style.display = 'grid';
    items.forEach(it=>{
      const card = document.createElement('div'); card.className='result-card';
      let cover;
if (it.cover) {
  cover = document.createElement('img');
  cover.className = 'cover';
  cover.src = it.cover;
  cover.alt = it.title;
  cover.style.width = '70px';
  cover.style.height = '96px';
  cover.style.borderRadius = '6px';
  cover.style.objectFit = 'cover';
} else {
  cover = document.createElement('div');
  cover.className = 'cover';
  const initials = (it.title||'').split(' ').slice(0,2).map(s=>s[0]).join('').toUpperCase();
  cover.textContent = initials;
}
      const meta = document.createElement('div'); meta.className='meta';
      const title = document.createElement('div'); title.className='title'; title.textContent = it.title;
      const author = document.createElement('div'); author.className='author'; author.textContent = it.author;
      const desc = document.createElement('div'); desc.className='desc'; desc.textContent = it.description;
      const foot = document.createElement('div'); foot.className='meta-foot';
      const genres = document.createElement('div'); genres.className='genres'; genres.textContent = (it.genres || []).join(', ');
      const score = document.createElement('div');
      score.className = 'score';
      if (typeof it.score === 'number') {
        const pct = Math.round((it.score) * 100); // e.g. 0.513 -> 51%
        score.textContent = pct + '%';
      } else {
        score.textContent = '';
      }
      foot.appendChild(genres); foot.appendChild(score);
      meta.appendChild(title); meta.appendChild(author); meta.appendChild(desc); meta.appendChild(foot);
      card.appendChild(cover); card.appendChild(meta);
      resultsEl.appendChild(card);
    });
  }
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)

@app.post("/recommend")
async def recommend(request: Request):
    body = await request.json()
    text = body.get("text", "")
    k = int(body.get("k", 5))
    if not text:
        return JSONResponse({"error": "provide 'text' in JSON"}, status_code=400)
    results = recommend_by_text(text, k=k)
    return {"results": results}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
