# How to Prepare Training Data for a Frontier LLM

**A complete guide to building the data pipeline behind models like GPT-4, Claude, DeepSeek-V3, and Llama 3 — from crawling raw internet data to producing a training-ready token stream.**

---

## Table of Contents

1. [Overview: What Goes Into an LLM](#1-overview-what-goes-into-an-llm)
2. [Phase 1: Web Crawling](#2-phase-1-web-crawling)
3. [Phase 2: Text Extraction](#3-phase-2-text-extraction)
4. [Phase 3: Quality Filtering](#4-phase-3-quality-filtering)
5. [Phase 4: Deduplication](#5-phase-4-deduplication)
6. [Phase 5: Text Cleaning & Normalization](#6-phase-5-text-cleaning--normalization)
7. [Phase 6: Domain-Specific Data](#7-phase-6-domain-specific-data)
8. [Phase 7: Custom Data — GitHub, Google, PDFs, APIs](#8-phase-7-custom-data--github-google-pdfs-apis)
9. [Phase 8: Synthetic Data Generation](#9-phase-8-synthetic-data-generation)
10. [Phase 9: Tokenization](#10-phase-9-tokenization)
11. [Phase 10: Data Mixing & Packaging](#11-phase-10-data-mixing--packaging)
12. [Phase 11: Curriculum Learning & Data Scheduling](#12-phase-11-curriculum-learning--data-scheduling)
13. [Phase 12: Training at Scale](#13-phase-12-training-at-scale)
14. [Phase 13: Post-Training — SFT, DPO, RLHF](#14-phase-13-post-training--sft-dpo-rlhf)
15. [Phase 14: End-to-End with superGPT](#15-phase-14-end-to-end-with-supergpt)
16. [Appendix: Open Datasets](#16-appendix-open-datasets)
17. [References](#references)

---

## 1. Overview: What Goes Into an LLM

A frontier LLM requires **trillions of tokens** of diverse, high-quality data. Here's what the leading labs actually use:

### Data Mixture Ratios (from public technical reports)

| Model | Total Tokens | General Text | Code | Math & Reasoning | Multilingual |
|-------|-------------|-------------|------|------------------|-------------|
| **Llama 3** | 15T | 50% | 17% | 25% | 8% |
| **DeepSeek-V3** | 14.8T | ~55% | ~20% | ~15% | ~10% |
| **GPT-4** | ~13T (est.) | ~50% | ~20% | ~20% | ~10% |

### The Pipeline at a Glance

```
┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  RAW CRAWL   │───▶│  EXTRACTION   │───▶│  FILTERING   │
│  (petabytes) │    │  HTML → Text  │    │  Quality/PII │
└──────────────┘    └───────────────┘    └──────────────┘
                                               │
                    ┌───────────────┐    ┌──────▼───────┐
                    │   TOKENIZE    │◀───│   DEDUP      │
                    │  BPE/SentPc   │    │  MinHash/LSH │
                    └───────┬───────┘    └──────────────┘
                            │
                    ┌───────▼───────┐    ┌──────────────┐
                    │  MIX & PACK   │───▶│   TRAINING   │
                    │  Ratios/Shards│    │  FSDP/DeepSp │
                    └───────────────┘    └──────────────┘
```

---

## 2. Phase 1: Web Crawling

### Option A: Use Common Crawl (Recommended)

[Common Crawl](https://commoncrawl.org/) is a free, open repository of web crawl data containing **petabytes** of web pages. This is what FineWeb, RedPajama, The Pile, and most open LLM datasets are built from.

```bash
# Common Crawl data is on AWS S3 (free, no auth required)
# Each monthly crawl = ~3 billion pages

# List available crawls
aws s3 ls s3://commoncrawl/ --no-sign-request

# Download a WARC segment
aws s3 cp s3://commoncrawl/crawl-data/CC-MAIN-2024-10/segments/... \
    ./data/raw/ --no-sign-request
```

**File formats:**
- **WARC** (Web ARChive): Raw HTTP responses + HTML — **use this one**
- **WET**: Pre-extracted plain text (lower quality)
- **WAT**: Metadata only (URLs, timestamps)

> [!TIP]
> Always process WARC files yourself with Trafilatura rather than using WET files. Your text quality will be dramatically better.

### Option B: Custom Crawling

```python
"""Custom web crawler for specialized domains."""
import scrapy

class LLMDataSpider(scrapy.Spider):
    name = "llm_data"
    custom_settings = {
        'ROBOTSTXT_OBEY': True,       # Always respect robots.txt
        'DOWNLOAD_DELAY': 1.0,        # 1 req/sec (be polite)
        'CONCURRENT_REQUESTS': 8,
        'DEPTH_LIMIT': 3,
    }
    
    def parse(self, response):
        text = ' '.join(response.css('p::text, h1::text, h2::text').getall())
        if len(text) > 200:
            yield {'url': response.url, 'text': text}
        for href in response.css('a::attr(href)').getall():
            yield response.follow(href, self.parse)
```

**Crawling tools comparison:**

| Tool | Best For | Scale |
|------|---------|-------|
| [Common Crawl](https://commoncrawl.org/) | Full web corpus, free | Petabyte |
| [Scrapy](https://scrapy.org/) | Custom domain crawls | Medium |
| [Trafilatura](https://trafilatura.readthedocs.io/) | Text extraction from HTML | Any |
| [Firecrawl](https://firecrawl.dev/) | JS-rendered pages | Small-Med |

> [!CAUTION]
> **Always check:** robots.txt, Terms of Service, copyright/licensing, GDPR/CCPA for personal data. Use permissively-licensed data when possible.

---

## 3. Phase 2: Text Extraction

Raw HTML is full of menus, ads, JavaScript. You need only the **main content**.

```python
"""Extract clean text from HTML using Trafilatura (industry standard)."""
import trafilatura

def extract_from_warc(warc_path):
    """Process a Common Crawl WARC file."""
    import warcio
    with open(warc_path, 'rb') as f:
        for record in warcio.ArchiveIterator(f):
            if record.rec_type == 'response':
                html = record.content_stream().read().decode('utf-8', errors='ignore')
                text = trafilatura.extract(html, favor_precision=True)
                if text and len(text) > 200:
                    yield {
                        'url': record.rec_headers.get('WARC-Target-URI'),
                        'text': text,
                    }
```

For **petabyte scale**, use Apache Spark on AWS EMR:

```python
# Spark job: distribute across 1000+ executors
warc_paths = spark.read.text("s3://commoncrawl/.../warc.paths.gz")
extracted = warc_paths.rdd.flatMap(extract_from_warc)
extracted.saveAsTextFile("s3://my-bucket/extracted/")
```

---

## 4. Phase 3: Quality Filtering

This is the **most impactful step**. Quality > quantity.

### Layer 1: Heuristic Filters (Fast, Rule-Based)

```python
"""Based on FineWeb, RedPajama, and DeepMind Gopher heuristics."""
import re
from collections import Counter

def heuristic_filter(text: str) -> tuple[bool, str]:
    """Returns (pass, reason). Document must pass ALL checks."""
    words = text.split()
    n_words = len(words)
    
    if n_words < 50:                    return False, "too_short"
    if n_words > 100_000:               return False, "too_long"
    
    # Average word length (catches garbled text)
    avg_wl = sum(len(w) for w in words) / n_words
    if avg_wl < 3 or avg_wl > 10:       return False, "unusual_words"
    
    # Alpha ratio (catches code dumps, symbol spam)
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(len(text), 1) < 0.5: return False, "low_alpha"
    
    # Line repetition
    lines = text.split('\n')
    if len(set(lines)) / max(len(lines), 1) < 0.5:
        return False, "too_repetitive"
    
    # Top bigram frequency (from Gopher paper)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bc = Counter(bigrams)
    if bc and bc.most_common(1)[0][1] / len(bigrams) > 0.2:
        return False, "repeated_ngrams"
    
    # URL density (catches link farms)
    urls = re.findall(r'https?://\S+', text)
    if len(urls) / max(n_words, 1) > 0.1:
        return False, "too_many_urls"
    
    return True, "passed"
```

### Layer 2: Language Detection

```python
"""Filter by language using fastText (176 languages)."""
import fasttext
model = fasttext.load_model('lid.176.bin')

def detect_language(text):
    pred = model.predict(text.replace('\n', ' ')[:1000], k=1)
    return pred[0][0].replace('__label__', ''), pred[1][0]

def is_english(text, min_conf=0.65):
    lang, conf = detect_language(text)
    return lang == 'en' and conf >= min_conf
```

### Layer 3: Model-Based Quality Scoring (The Secret Weapon)

This is what separates FineWeb-Edu from FineWeb — and why it trains dramatically better models:

```python
"""
The FineWeb-Edu approach:
1. Use Llama-3-70B to score 500K samples on educational quality (0-5)
2. Train a fast fastText classifier on those labels
3. Run classifier on the full 15T token corpus
4. Keep only documents scoring ≥ 3

This single step improved downstream benchmarks by 10-20%.
"""
QUALITY_PROMPT = """Rate the educational quality of this text (0-5):
0 = Spam/ads    1 = Social media    2 = Average web content
3 = Informative  4 = Educational    5 = Textbook-quality

Text: {text}
Score:"""

# Score 500K samples with a big LLM → train fastText → filter everything
# pip install fasttext
# fasttext supervised -input labels.txt -output quality_model -epoch 25
```

### Layer 4: PII Removal

```python
"""Remove personal data (email, phone, SSN, credit cards)."""
import re

PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn':   r'\b\d{3}-\d{2}-\d{4}\b',
    'cc':    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
}

def redact_pii(text):
    for name, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f'[{name.upper()}_REDACTED]', text)
    return text
```

---

## 5. Phase 4: Deduplication

Web crawls contain **30-50% duplicates**. Training on duplicates wastes compute and causes memorization.

### Exact Dedup (Line/Document Level)

```python
import hashlib

def exact_dedup(documents):
    seen = set()
    for doc in documents:
        h = hashlib.sha256(doc['text'].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            yield doc
```

### Fuzzy Dedup: MinHash + LSH (Industry Standard)

Used by FineWeb, RedPajama, The Pile, Dolma. Finds documents that are ~80%+ similar.

```python
"""
MinHash LSH — the industry standard for near-duplicate detection.
1. Convert documents to n-gram sets (shingles)
2. Compute MinHash signatures (compact fingerprints)
3. Use LSH banding to find candidate pairs
4. Remove near-duplicates (Jaccard similarity > 0.8)
"""
from datasketch import MinHash, MinHashLSH

def create_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(len(words) - 4):
        shingle = ' '.join(words[i:i+5])
        m.update(shingle.encode('utf-8'))
    return m

def fuzzy_dedup(documents, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    
    for doc in documents:
        mh = create_minhash(doc['text'])
        minhashes[doc['id']] = mh
        try:
            lsh.insert(doc['id'], mh)
        except ValueError:
            pass  # Already exists
    
    # Find duplicates
    duplicates = set()
    for doc_id, mh in minhashes.items():
        if doc_id in duplicates: continue
        for candidate in lsh.query(mh):
            if candidate != doc_id:
                duplicates.add(candidate)
    
    return [d for d in documents if d['id'] not in duplicates]
```

For **trillion-token scale**, use suffix array dedup (finds duplicate **substrings** across documents):

```bash
# pip install text-dedup
python -m text_dedup.suffix_array --input corpus.jsonl --output deduped.jsonl --k 50
```

---

## 6. Phase 5: Text Cleaning & Normalization

Even after filtering and dedup, raw text has encoding issues, invisible characters, and formatting problems that hurt training.

### Unicode Normalization

```python
"""
Unicode normalization — critical for consistent tokenization.
Without this, 'é' (precomposed) and 'é' (e + combining accent) are different tokens.
"""
import unicodedata
import re

def normalize_unicode(text: str) -> str:
    """Normalize to NFC form and clean problematic characters."""
    # NFC: precompose characters (é as single char, not e + accent)
    text = unicodedata.normalize('NFC', text)
    
    # Remove zero-width characters (invisible but break tokenization)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)
    
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('…', '...')
    
    # Remove control characters (except newline, tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text
```

### HTML & Markdown Cleaning

```python
"""
Deep clean text extracted from web pages.
Even after Trafilatura, some HTML artifacts remain.
"""
import re

def deep_clean_text(text: str) -> str:
    """Remove residual HTML, fix encoding, normalize whitespace."""
    
    # Remove any residual HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove HTML entities
    import html
    text = html.unescape(text)
    
    # Fix common encoding artifacts
    replacements = {
        'â€™': "'", 'â€œ': '"', 'â€\x9d': '"',
        'â€"': '—', 'â€"': '–', 'Â ': ' ',
        'Ã©': 'é', 'Ã¡': 'á', 'Ã±': 'ñ',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    
    # Collapse excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)          # Multiple spaces → one
    text = re.sub(r'\n{3,}', '\n\n', text)        # 3+ newlines → 2
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)   # Blank lines
    
    # Remove lines that are just URLs
    lines = text.split('\n')
    lines = [l for l in lines if not re.match(r'^\s*https?://\S+\s*$', l)]
    
    # Remove lines that are just navigation (short repeated patterns)
    lines = [l for l in lines if len(l.strip()) > 3 or l.strip() == '']
    
    return '\n'.join(lines).strip()
```

### Full Cleaning Pipeline

```python
"""
Complete text cleaning pipeline — run this on every document
AFTER extraction and BEFORE tokenization.
"""

def clean_document(text: str) -> str | None:
    """Full cleaning pipeline. Returns None if document should be dropped."""
    
    # Step 1: Unicode normalize
    text = normalize_unicode(text)
    
    # Step 2: Deep clean (HTML artifacts, encoding issues)
    text = deep_clean_text(text)
    
    # Step 3: Remove excessive repetition within document
    lines = text.split('\n')
    seen_lines = set()
    deduped_lines = []
    for line in lines:
        stripped = line.strip().lower()
        if stripped and stripped in seen_lines:
            continue  # Skip duplicate lines within same document
        seen_lines.add(stripped)
        deduped_lines.append(line)
    text = '\n'.join(deduped_lines)
    
    # Step 4: Final length check
    words = text.split()
    if len(words) < 50:
        return None  # Too short after cleaning
    
    return text
```

---

## 7. Phase 6: Domain-Specific Data

### 6.1 Code (17-20% of training mix)

Code dramatically improves reasoning — even for non-code tasks.

**Primary source: [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)** — 6.4 TB, 358 languages, permissively licensed.

```python
from datasets import load_dataset

ds = load_dataset("bigcode/the-stack-v2", split="train", streaming=True)

def filter_code(example):
    code = example['content']
    if len(code) < 100 or len(code) > 100_000: return False
    # Skip auto-generated
    if any(m in code[:500] for m in ['auto-generated', 'DO NOT EDIT']): return False
    # Skip minified
    if len(code) / max(code.count('\n'), 1) > 200: return False
    return True

# Language weights (by impact on reasoning)
LANG_WEIGHTS = {
    'python': 0.25, 'javascript': 0.12, 'typescript': 0.10,
    'java': 0.10, 'c++': 0.08, 'c': 0.06, 'go': 0.05,
    'rust': 0.05, 'shell': 0.03, 'sql': 0.03,
}
```

### 6.2 Math (15-25% of training mix)

Mathematical data is critical for reasoning transfer.

| Dataset | Size | Content | Use |
|---------|------|---------|-----|
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | 14.7B tokens | Math from web (LaTeX, proofs) | Pre-training |
| [MathPile](https://huggingface.co/datasets/GAIR/MathPile) | 9.5B tokens | Textbooks, arXiv, StackExchange | Pre-training |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 8.5K problems | Grade school word problems | SFT |
| [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | 12.5K problems | Competition math (AMC, Olympiad) | SFT |
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 860K problems | CoT solutions, all levels | SFT |

```python
# Key: preserve LaTeX notation during processing!
def process_math(text):
    # DON'T strip $...$ or \[...\] — these teach math reasoning
    text = re.sub(r'\n{3,}', '\n\n', text)
    has_math = any(m in text for m in ['$', '\\frac', '\\sum', 'theorem', 'proof'])
    return text if has_math else None
```

### 6.3 Academic Papers

- **arXiv**: 2M+ papers with full LaTeX source — [Kaggle mirror](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- **S2ORC**: 81M papers across all fields — [GitHub](https://github.com/allenai/s2orc)
- **PubMed**: 35M+ biomedical abstracts

### 6.4 Books & Long-Form

- **Project Gutenberg**: 70K+ public domain books — [gutenberg.org](https://www.gutenberg.org/)
- Critical for teaching long-range coherence

### 6.5 Conversation & Instruction Data (Post-Training)

| Dataset | Size | Quality |
|---------|------|---------|
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | ~50M tok | High |
| [WizardLM Evol-Instruct](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) | ~100M tok | High |
| [UltraChat](https://huggingface.co/datasets/stingning/ultrachat) | ~500M tok | Medium |
| [Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | ~50M tok | High |

---

## 8. Phase 7: Custom Data — GitHub, Google, PDFs, APIs

Most practitioners don't start from Common Crawl. Here's how to build a custom dataset from real-world sources.

### 8.1 Crawling GitHub Repositories

```python
"""
Crawl GitHub repos for code training data.
Uses the GitHub API + git clone for full content.
"""
import os
import subprocess
import requests
import json

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Set your PAT
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

def search_github_repos(language: str, min_stars: int = 100, max_repos: int = 500):
    """Find popular repos by language."""
    repos = []
    page = 1
    while len(repos) < max_repos:
        url = (f"https://api.github.com/search/repositories"
               f"?q=language:{language}+stars:>{min_stars}"
               f"&sort=stars&per_page=100&page={page}")
        resp = requests.get(url, headers=HEADERS)
        data = resp.json()
        if 'items' not in data:
            break
        for item in data['items']:
            # Only permissively-licensed repos
            license_key = (item.get('license') or {}).get('key', '')
            if license_key in ['mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause']:
                repos.append({
                    'name': item['full_name'],
                    'url': item['clone_url'],
                    'stars': item['stargazers_count'],
                    'license': license_key,
                    'language': language,
                })
        page += 1
        if page > 5:
            break
    return repos[:max_repos]

def clone_and_extract(repo_url, output_dir, extensions=None):
    """Clone a repo and extract source files."""
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    clone_dir = f"/tmp/repos/{repo_name}"
    
    # Shallow clone (saves bandwidth)
    subprocess.run(['git', 'clone', '--depth', '1', repo_url, clone_dir],
                   capture_output=True, timeout=60)
    
    if extensions is None:
        extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs',
                      '.rb', '.php', '.swift', '.kt', '.scala', '.sh', '.sql'}
    
    documents = []
    for root, _, files in os.walk(clone_dir):
        # Skip hidden dirs, tests, vendor, node_modules
        if any(skip in root for skip in ['/.', '/test', '/vendor', '/node_modules',
                                          '/__pycache__', '/dist', '/build']):
            continue
        for fname in files:
            ext = os.path.splitext(fname)[1]
            if ext in extensions:
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', errors='ignore') as f:
                        content = f.read()
                    if 100 < len(content) < 100_000:  # Skip tiny/huge files
                        documents.append({
                            'text': content,
                            'source': f"github:{repo_url}:{fname}",
                            'language': ext[1:],
                        })
                except:
                    pass
    
    # Cleanup
    subprocess.run(['rm', '-rf', clone_dir], capture_output=True)
    return documents

# Example: Crawl top Python repos
repos = search_github_repos('python', min_stars=500, max_repos=200)
all_code = []
for repo in repos:
    docs = clone_and_extract(repo['url'], 'data/code/')
    all_code.extend(docs)
    print(f"  {repo['name']}: {len(docs)} files")

print(f"Total: {len(all_code)} code files from {len(repos)} repos")
```

### 8.2 Crawling Google Search Results

```python
"""
Crawl content from Google search results on specific topics.
Useful for building domain-specific datasets (medical, legal, etc.)
"""
import requests
from bs4 import BeautifulSoup
import trafilatura
import time

def google_search(query: str, num_results: int = 50) -> list[str]:
    """Get URLs from Google search (use SerpAPI for production)."""
    # Option 1: Use SerpAPI (recommended, has free tier)
    # pip install google-search-results
    from serpapi import GoogleSearch
    
    urls = []
    for start in range(0, num_results, 10):
        params = {
            "q": query,
            "start": start,
            "num": 10,
            "api_key": os.getenv("SERPAPI_KEY"),
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        for result in results.get("organic_results", []):
            urls.append(result["link"])
        time.sleep(1)  # Rate limit
    
    return urls[:num_results]

def crawl_search_results(queries: list[str]) -> list[dict]:
    """Crawl and extract text from Google search results."""
    documents = []
    
    for query in queries:
        urls = google_search(query, num_results=30)
        
        for url in urls:
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(downloaded, favor_precision=True)
                    if text and len(text) > 200:
                        documents.append({
                            'text': text,
                            'source': url,
                            'query': query,
                        })
            except Exception as e:
                continue
            time.sleep(0.5)  # Be polite
    
    return documents

# Example: Build a machine learning textbook corpus
ml_queries = [
    "transformer architecture explained tutorial",
    "backpropagation gradient descent mathematics",
    "attention mechanism self-attention multi-head",
    "convolutional neural networks image recognition",
    "reinforcement learning policy gradient methods",
    "natural language processing tokenization embeddings",
    "generative adversarial networks training stability",
    "diffusion models denoising score matching",
]

ml_docs = crawl_search_results(ml_queries)
print(f"Collected {len(ml_docs)} documents on ML topics")
```

### 8.3 Extracting Text from PDFs (Books, Papers, Manuals)

```python
"""
Extract training data from PDF files.
Great for textbooks, research papers, technical manuals.
"""

def extract_pdf_text(pdf_path: str) -> str:
    """Extract clean text from a PDF file."""
    # Option 1: PyMuPDF (fastest, best quality)
    import fitz  # pip install PyMuPDF
    
    doc = fitz.open(pdf_path)
    text_parts = []
    
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            text_parts.append(text)
    
    full_text = '\n'.join(text_parts)
    
    # Clean up PDF artifacts
    import re
    full_text = re.sub(r'-\n(\w)', r'\1', full_text)     # Fix hyphenation
    full_text = re.sub(r'(\w)\n(\w)', r'\1 \2', full_text)  # Fix line breaks mid-sentence
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    
    return full_text

def process_pdf_directory(pdf_dir: str) -> list[dict]:
    """Process all PDFs in a directory."""
    import glob
    documents = []
    
    for pdf_path in glob.glob(os.path.join(pdf_dir, '**/*.pdf'), recursive=True):
        try:
            text = extract_pdf_text(pdf_path)
            if len(text) > 500:  # Skip very short PDFs
                documents.append({
                    'text': text,
                    'source': pdf_path,
                    'type': 'pdf',
                })
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    
    return documents

# Example: Process a directory of ML textbooks
docs = process_pdf_directory('/data/textbooks/')
print(f"Extracted text from {len(docs)} PDFs")
```

### 8.4 Wikipedia (Full Dump)

```python
"""
Process a Wikipedia dump — one of the cleanest, most structured data sources.
Every major LLM uses Wikipedia.
"""
from datasets import load_dataset

# HuggingFace hosts pre-processed Wikipedia
wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

documents = []
for i, article in enumerate(wiki):
    text = article['text']
    if len(text) > 500:  # Skip stubs
        documents.append({
            'text': text,
            'title': article['title'],
            'source': 'wikipedia',
        })
    if i >= 100_000:  # Limit for example
        break

print(f"Loaded {len(documents)} Wikipedia articles")
```

### 8.5 StackOverflow & StackExchange

```python
"""
StackOverflow Q&A — excellent for teaching models to answer technical questions.
Available as public data dumps.
"""
from datasets import load_dataset

# Load StackOverflow questions/answers
so = load_dataset("koutch/stackoverflow_python", split="train", streaming=True)

def format_qa_pair(example):
    """Format as question → answer for training."""
    return {
        'text': (
            f"Question: {example['question_body']}\n\n"
            f"Answer: {example['answer_body']}\n\n"
            f"Score: {example['answer_score']}"
        ),
        'source': 'stackoverflow',
    }

# Alternative: Download the full XML dump from archive.org
# https://archive.org/details/stackexchange
```

### 8.6 API-Based Data Collection

```python
"""
Collect data from APIs — news, research, documentation.
"""
import requests

# Example: arXiv API for recent papers
def fetch_arxiv_papers(query: str, max_results: int = 100):
    """Fetch papers from arXiv API."""
    import xml.etree.ElementTree as ET
    
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
    resp = requests.get(url)
    root = ET.fromstring(resp.text)
    
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        abstract = entry.find('atom:summary', ns).text.strip()
        papers.append({
            'text': f"Title: {title}\n\nAbstract: {abstract}",
            'source': 'arxiv',
        })
    return papers

# Example: News API
def fetch_news(query: str, api_key: str):
    """Fetch news articles from NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&pageSize=100"
    resp = requests.get(url)
    articles = []
    for item in resp.json().get('articles', []):
        if item.get('content'):
            articles.append({
                'text': f"{item['title']}\n\n{item['content']}",
                'source': item['url'],
            })
    return articles
```

### 8.7 Combining All Custom Sources

```python
"""
Merge data from all custom sources into a unified JSONL corpus.
"""
import json

def merge_sources(source_lists: dict[str, list], output_path: str):
    """Merge multiple data sources into a single JSONL file."""
    total = 0
    
    with open(output_path, 'w') as f:
        for source_name, documents in source_lists.items():
            for doc in documents:
                doc['source_type'] = source_name
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                total += 1
    
    print(f"Merged {total} documents from {len(source_lists)} sources")
    print(f"Saved to: {output_path}")

# Example usage:
merge_sources({
    'github_code': all_code,           # From 8.1
    'web_articles': ml_docs,           # From 8.2
    'pdf_textbooks': pdf_docs,         # From 8.3
    'wikipedia': wiki_docs,            # From 8.4
    'stackoverflow': so_docs,          # From 8.5
    'arxiv_papers': arxiv_docs,        # From 8.6
}, output_path='data/custom_corpus.jsonl')
```

---

## 9. Phase 8: Synthetic Data Generation

Synthetic data is now **essential** for modern LLMs. DeepSeek-V3 used synthetic data from DeepSeek-R1 for reasoning. GPT-4 was trained partly on GPT-4-generated data.

### 9.1 Self-Instruct: Generate Instructions from Seed Tasks

```python
"""
Self-Instruct: start with ~100 seed tasks, use an LLM to generate thousands more.
Original method from Stanford (2023). Cost-effective for SFT data.
"""
from openai import OpenAI
import json, random

client = OpenAI()

SEED_TASKS = [
    {"instruction": "Write a Python function to sort a list of dictionaries by a key.",
     "output": "def sort_dicts(lst, key):\n    return sorted(lst, key=lambda x: x[key])"},
    {"instruction": "Explain what a neural network is in simple terms.",
     "output": "A neural network is a computer system inspired by the human brain..."},
    {"instruction": "Solve: What is 15% of 240?",
     "output": "15% of 240 = 0.15 × 240 = 36"},
    # Add 100+ diverse seed tasks
]

def generate_instructions(seed_tasks: list, n_generate: int = 5000):
    """Generate new instructions using self-instruct."""
    all_tasks = list(seed_tasks)
    
    for i in range(n_generate):
        # Sample 3 random examples as context
        examples = random.sample(seed_tasks, min(3, len(seed_tasks)))
        examples_text = "\n".join(
            f"Instruction: {t['instruction']}\nOutput: {t['output']}"
            for t in examples
        )
        
        prompt = f"""Here are some example tasks:

{examples_text}

Generate a new, different instruction and its output. 
Be creative and diverse. Cover coding, math, writing, analysis, and reasoning.

New Instruction:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap and fast
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.9,
        )
        
        text = response.choices[0].message.content
        # Parse instruction and output from response
        if "Output:" in text:
            parts = text.split("Output:", 1)
            all_tasks.append({
                "instruction": parts[0].strip(),
                "output": parts[1].strip(),
            })
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_generate} instructions")
    
    return all_tasks
```

### 9.2 Evol-Instruct: Evolve Simple Instructions into Complex Ones

```python
"""
Evol-Instruct (WizardLM): Take simple instructions and evolve them
through multiple rounds to increase complexity and diversity.

Evolution operators:
1. Add constraints ("...do this without using loops")
2. Deepen ("explain the underlying mathematics")  
3. Concretize ("give a specific example with real data")
4. Increase reasoning ("solve step-by-step showing all work")
5. Broaden ("extend to handle edge cases")
"""

EVOLUTION_PROMPTS = {
    "add_constraints": """Rewrite this instruction to add 2-3 specific constraints 
that make it more challenging:
Original: {instruction}
Evolved:""",

    "deepen": """Rewrite this instruction to require deeper knowledge and understanding:
Original: {instruction}
Evolved:""",

    "concretize": """Rewrite this instruction to be more specific with concrete 
examples, real data, or specific scenarios:
Original: {instruction}
Evolved:""",

    "increase_reasoning": """Rewrite this instruction to require multi-step 
reasoning, showing all intermediate steps:
Original: {instruction}
Evolved:""",
}

def evolve_instruction(instruction: str, n_rounds: int = 3) -> list[str]:
    """Evolve a simple instruction through multiple complexity rounds."""
    evolved = [instruction]
    
    for round_num in range(n_rounds):
        operator = random.choice(list(EVOLUTION_PROMPTS.keys()))
        prompt = EVOLUTION_PROMPTS[operator].format(instruction=evolved[-1])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        
        new_instruction = response.choices[0].message.content.strip()
        evolved.append(new_instruction)
    
    return evolved
```

### 9.3 Magpie: Fully Automated Data Synthesis

```python
"""
Magpie (2024): Generate instruction data WITHOUT any seed tasks.
Just prompt an instruction-tuned model with the chat template prefix.
The model naturally generates a user query, then you generate a response.

This is the cheapest, most scalable synthetic data method.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def magpie_generate(model_name="meta-llama/Meta-Llama-3-8B-Instruct", n=1000):
    """Generate instruction-response pairs using Magpie method."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # The "trick": just give the model the chat template prefix
    # and it will generate a realistic user query
    system_prefix = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    
    pairs = []
    for i in range(n):
        # Step 1: Generate a user instruction
        inputs = tokenizer(system_prefix, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200, temperature=0.9,
                                     do_sample=True, top_p=0.95)
        instruction = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Step 2: Generate the assistant response
        full_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1000, temperature=0.7)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        pairs.append({"instruction": instruction, "response": response})
        
        if (i + 1) % 100 == 0:
            print(f"  Magpie: {i+1}/{n} pairs generated")
    
    return pairs
```

### 9.4 Chain-of-Thought Distillation

```python
"""
Generate CoT (Chain-of-Thought) reasoning data from a strong model.
This is how DeepSeek-V3 got reasoning from DeepSeek-R1.
"""

def generate_cot_data(problems: list[str], model="gpt-4o"):
    """Generate step-by-step solutions using a strong model."""
    cot_data = []
    
    for problem in problems:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": "Solve the following problem step by step. "
                           "Show ALL intermediate reasoning. "
                           "Use <think>...</think> tags for reasoning, "
                           "then give the final answer."
            }, {
                "role": "user",
                "content": problem
            }],
            temperature=0.3,
        )
        
        solution = response.choices[0].message.content
        cot_data.append({
            "problem": problem,
            "solution": solution,
            "type": "chain_of_thought",
        })
    
    return cot_data

# Combine with rejection sampling: generate N solutions, keep only correct ones
def rejection_sample(problem, correct_answer, model="gpt-4o-mini", n=8):
    """Generate N solutions, keep only ones that reach the correct answer."""
    correct_solutions = []
    
    for _ in range(n):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Solve: {problem}"}],
            temperature=0.9,  # High temp for diversity
        )
        solution = response.choices[0].message.content
        if str(correct_answer) in solution:
            correct_solutions.append(solution)
    
    return correct_solutions  # Multiple diverse correct solutions
```

---

## 10. Phase 9: Tokenization

### Choosing a Tokenizer

| Tokenizer | Vocab Size | Used By | Notes |
|-----------|-----------|---------|-------|
| **tiktoken** (cl100k) | 100K | GPT-4, GPT-3.5 | Fast, byte-level BPE |
| **SentencePiece** | 32K-128K | LLaMA, Mistral | Good multilingual support |
| **Qwen tokenizer** | 151K | Qwen2.5, DeepSeek-V3 | Best for code + multilingual |

### Training a Custom Tokenizer

```python
"""Train a BPE tokenizer optimized for YOUR data mix."""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=100_000,
    min_frequency=100,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    show_progress=True,
)

# Train on a representative sample of your data mix
tokenizer.train(files=[
    "data/web_sample.txt",     # 50%
    "data/code_sample.txt",    # 20%
    "data/math_sample.txt",    # 20%
    "data/multilingual.txt",   # 10%
], trainer=trainer)

tokenizer.save("my_tokenizer.json")
```

### Convert Text → Token IDs → Binary

```python
"""Convert filtered text corpus into binary training files."""
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
vocab_size = tokenizer.vocab_size  # 151,643

# Choose dtype based on vocab size
dtype = np.uint32 if vocab_size > 65535 else np.uint16

all_tokens = []
for doc in read_jsonl("filtered_corpus.jsonl"):
    tokens = tokenizer.encode(doc['text'])
    tokens.append(tokenizer.eos_token_id)  # Document separator
    all_tokens.extend(tokens)

# Save as memory-mapped binary
arr = np.array(all_tokens, dtype=dtype)
arr.tofile("data/train.bin")

print(f"Saved {len(arr):,} tokens ({arr.nbytes / 1e9:.1f} GB)")
```

> [!IMPORTANT]
> **Always use `uint32` if your vocab size > 65,535!** Using `uint16` for large vocabs (like Qwen's 151K) will silently corrupt your data by truncating token IDs. This is a common bug — we found and fixed it in superGPT.

---

## 11. Phase 10: Data Mixing & Packaging

### The Art of Data Mixing

The ratio of different data domains **dramatically** affects model capabilities. Here's the recipe:

```python
"""
Data mixing strategy based on Llama 3 and DeepSeek-V3.

Key insights:
1. Start with ~50% general web text (foundation)
2. Code improves reasoning even on non-code tasks (17-20%)
3. Math is the highest-leverage domain for reasoning (15-25%)
4. Overtrain on high-quality data in later phases (annealing)
"""

# Phase 1: Main pre-training (first 80% of compute)
PRETRAIN_MIX = {
    'web_text':     0.50,   # FineWeb-Edu, RedPajama, C4
    'code':         0.17,   # The Stack, GitHub
    'math':         0.15,   # OpenWebMath, MathPile
    'academic':     0.08,   # arXiv, S2ORC, PubMed
    'books':        0.05,   # Gutenberg, BookCorpus
    'multilingual': 0.05,   # CC-100, OPUS
}

# Phase 2: Annealing (final 20% of compute)
# Upsample high-quality data for final polish
ANNEAL_MIX = {
    'web_text':     0.30,   # Only top-quality (FineWeb-Edu score ≥ 4)
    'code':         0.25,   # Upsample — code helps reasoning
    'math':         0.25,   # Heavy upsample on math
    'academic':     0.10,
    'books':        0.05,
    'multilingual': 0.05,
}
```

### Sharding for Distributed Training

```python
"""
Shard data for multi-GPU/multi-node training.
Each shard should be ~256 MB - 1 GB.
"""
import numpy as np
import os

def shard_data(input_path, output_dir, n_shards=256):
    """Split a large .bin file into evenly-sized shards."""
    os.makedirs(output_dir, exist_ok=True)
    
    data = np.memmap(input_path, dtype=np.uint32, mode='r')
    shard_size = len(data) // n_shards
    
    for i in range(n_shards):
        start = i * shard_size
        end = start + shard_size if i < n_shards - 1 else len(data)
        
        shard = np.array(data[start:end], dtype=np.uint32)
        shard.tofile(os.path.join(output_dir, f"shard_{i:05d}.bin"))
    
    print(f"Created {n_shards} shards in {output_dir}")
    print(f"Shard size: ~{shard_size * 4 / 1e6:.0f} MB each")

# Example: 100B tokens → 400 GB → 400 shards of 1 GB each
shard_data("data/train.bin", "data/shards/", n_shards=400)
```

### Data Loading for Training

```python
"""
Efficient data loading for billion-token training runs.
Uses memory-mapped files — never loads the full dataset into RAM.
"""
import numpy as np
import torch

class ShardedDataLoader:
    """Load data from multiple shards, cycling through them."""
    
    def __init__(self, shard_dir, block_size, batch_size, dtype=np.uint32):
        self.shards = sorted(
            [os.path.join(shard_dir, f) for f in os.listdir(shard_dir)
             if f.endswith('.bin')]
        )
        self.block_size = block_size
        self.batch_size = batch_size
        self.dtype = dtype
        self.current_shard = 0
        self._load_shard()
    
    def _load_shard(self):
        self.data = np.memmap(
            self.shards[self.current_shard],
            dtype=self.dtype, mode='r'
        )
    
    def get_batch(self):
        ix = torch.randint(len(self.data) - self.block_size - 1, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64))
            for i in ix
        ])
        
        # Advance shard every N batches
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self._load_shard()
        
        return x, y
```

---

## 12. Phase 11: Curriculum Learning & Data Scheduling

### Why Data Order Matters

Training on data in a strategic order — easy → hard — can improve convergence by 10-20%. This is called **curriculum learning**.

### Scoring Document Difficulty

```python
"""
Score documents by difficulty using multiple metrics.
Train on easy data first, complex data later.
"""
import re
import math

def score_difficulty(text: str) -> float:
    """Score text difficulty 0-1 (0 = easy, 1 = hard)."""
    words = text.split()
    n_words = len(words)
    
    if n_words == 0:
        return 0.5
    
    # 1. Average word length (longer words = harder)
    avg_word_len = sum(len(w) for w in words) / n_words
    word_score = min(avg_word_len / 10, 1.0)
    
    # 2. Sentence length (longer sentences = harder)
    sentences = re.split(r'[.!?]+', text)
    avg_sent_len = n_words / max(len(sentences), 1)
    sent_score = min(avg_sent_len / 40, 1.0)
    
    # 3. Vocabulary richness (more unique words = harder)
    vocab_ratio = len(set(w.lower() for w in words)) / n_words
    vocab_score = vocab_ratio
    
    # 4. Technical content (LaTeX, code, formulas = harder)
    tech_markers = sum(1 for m in ['$', '\\frac', 'def ', 'class ', 'import ',
                                     'theorem', 'proof', 'algorithm']
                       if m in text)
    tech_score = min(tech_markers / 5, 1.0)
    
    # 5. Document length (longer = harder to learn from)
    len_score = min(math.log(n_words + 1) / math.log(10000), 1.0)
    
    # Weighted combination
    return (0.2 * word_score + 0.2 * sent_score + 0.2 * vocab_score +
            0.25 * tech_score + 0.15 * len_score)
```

### Staged Training Implementation

```python
"""
Implement curriculum learning with staged data scheduling.
"""
import numpy as np

def create_curriculum_shards(documents, n_stages=4, output_dir='data/curriculum/'):
    """Sort documents by difficulty and create staged shards."""
    import os
    
    # Score all documents
    scored = [(score_difficulty(doc['text']), doc) for doc in documents]
    scored.sort(key=lambda x: x[0])  # Easy first
    
    # Split into stages
    stage_size = len(scored) // n_stages
    
    for stage in range(n_stages):
        start = stage * stage_size
        end = start + stage_size if stage < n_stages - 1 else len(scored)
        
        stage_docs = [doc for _, doc in scored[start:end]]
        
        stage_dir = os.path.join(output_dir, f'stage_{stage}')
        os.makedirs(stage_dir, exist_ok=True)
        
        # Tokenize and save this stage
        # ... (use your tokenization pipeline)
        
        avg_diff = np.mean([s for s, _ in scored[start:end]])
        print(f"Stage {stage}: {len(stage_docs)} docs, avg difficulty: {avg_diff:.3f}")

# Training schedule:
# Stage 0 (iter 0-2500):     Easy web text, simple sentences
# Stage 1 (iter 2500-5000):  Medium articles, basic code
# Stage 2 (iter 5000-7500):  Hard academic papers, complex code
# Stage 3 (iter 7500-10000): Hardest math proofs, research papers
```

### Annealing: The Final Push

```python
"""
In the last 20% of training, switch to a curated high-quality mix.
This is what Llama 3 calls the 'annealing' phase.

Key changes during annealing:
1. Upsample highest-quality data (FineWeb-Edu score >= 4)
2. Upsample code and math (helps reasoning)  
3. Reduce learning rate to near-zero
4. May increase context length (e.g., 4K → 128K)
"""

# Llama 3 annealing schedule:
# - Last 40M tokens on highest quality mix
# - Linear LR decay from 3.5e-5 → 0
# - Code weight: 17% → 25%
# - Math weight: 25% → 35%
# - Result: +2-3% on reasoning benchmarks vs no annealing
```

---

## 13. Phase 12: Training at Scale

### Hardware Requirements

| Model Size | GPUs Needed | VRAM per GPU | Training Time | Estimated Cost |
|-----------|------------|-------------|--------------|---------------|
| 125M | 1× A100/H100 | 40 GB | 1-3 days | $100-500 |
| 1B | 4× A100 | 40 GB each | 1-2 weeks | $2K-5K |
| 7B | 8× A100 | 80 GB each | 2-4 weeks | $20K-50K |
| 70B | 64× A100 | 80 GB each | 1-2 months | $300K-1M |
| 405B (Llama 3) | 16K H100 | 80 GB each | 54 days | $30M+ |
| 671B (DeepSeek-V3) | 2048 H800 | 80 GB each | 2 months | $5.5M |

### Parallelism Strategies

```python
"""
At scale, you need multiple forms of parallelism:

1. Data Parallelism (DP): Same model on each GPU, different data
   - Simplest. Works up to ~8 GPUs.
   - PyTorch DDP or FSDP

2. Tensor Parallelism (TP): Split layers across GPUs
   - Splits attention heads and FFN across GPUs within a node
   - Requires fast interconnect (NVLink)

3. Pipeline Parallelism (PP): Split layers across nodes
   - Layer 0-15 on Node 1, Layer 16-31 on Node 2, etc.
   - Works across slower interconnects

4. Expert Parallelism (EP): For MoE models
   - Each GPU holds a subset of experts
   - Tokens are all-to-all dispatched to the right GPU
"""

# For most users: FSDP is the easiest path to multi-GPU
# torchrun --nproc_per_node=8 -m supergpt.training.train \
#     --preset medium --distributed --compile

# superGPT handles FSDP setup automatically:
from supergpt.training.train import main
# Just add --distributed flag
```

### Training with superGPT

```bash
# Single GPU (up to ~1B params on 80GB VRAM)
python -m supergpt.training.train \
    --preset small \
    --data-dir data \
    --max-iters 50000 \
    --batch-size 32 \
    --lr 3e-4 \
    --compile \
    --device cuda

# Multi-GPU with FSDP
torchrun --nproc_per_node=8 \
    -m supergpt.training.train \
    --preset medium \
    --data-dir data \
    --max-iters 100000 \
    --distributed \
    --compile

# With FP8 (2× memory savings on H100)
python -m supergpt.training.train \
    --preset large \
    --data-dir data \
    --fp8 \
    --compile

# With logging
python -m supergpt.training.train \
    --preset small \
    --wandb \
    --tensorboard
```

### Key Training Hyperparameters

| Parameter | Small (≤1B) | Medium (1-10B) | Large (≥10B) |
|-----------|------------|---------------|-------------|
| Learning Rate | 3e-4 | 1.5e-4 | 6e-5 |
| Batch Size (tokens) | 512K | 2M | 4M+ |
| Warmup Steps | 2,000 | 2,000 | 2,000 |
| LR Schedule | Cosine | Cosine | Cosine |
| Weight Decay | 0.1 | 0.1 | 0.1 |
| Gradient Clipping | 1.0 | 1.0 | 1.0 |
| Context Length | 2K → 8K | 4K → 32K | 4K → 128K |

---

## 14. Phase 13: Post-Training — SFT, DPO, RLHF

After pre-training, the model can predict text but can't follow instructions. Post-training turns it into an assistant.

### Stage 1: Supervised Fine-Tuning (SFT)

Train on 500K-1.5M instruction-following examples. The model learns the system/user/assistant turn pattern.

```bash
# superGPT SFT with LoRA (memory-efficient)
python -m supergpt.training.finetune \
    --checkpoint checkpoints/best.pt \
    --data-dir data/sft/ \
    --lora-rank 64 \
    --max-iters 5000 \
    --lr 2e-5
```

### Stage 2: RLHF / DPO Alignment

Align the model to human preferences. DPO (Direct Preference Optimization) is simpler than classic RLHF — no reward model needed.

```bash
# superGPT DPO alignment
python -m supergpt.training.align_dpo \
    --checkpoint checkpoints/sft_best.pt \
    --data-dir data/preferences/ \
    --beta 0.1 \
    --max-iters 2000
```

### Stage 3: Knowledge Distillation (Optional)

Transfer reasoning from a large teacher to a smaller student. DeepSeek-V3 distilled from DeepSeek-R1 for enhanced reasoning.

```bash
# superGPT distillation
python -m supergpt.training.distill \
    --teacher checkpoints/large_model.pt \
    --student-preset small \
    --data-dir data/ \
    --alpha 0.7 \
    --temperature 2.0
```

---

## 11. Appendix: Open Datasets

### Pre-Training Datasets

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **FineWeb-Edu** | 1.3T tokens | Ultra-high-quality filtered web text | [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| **FineWeb** | 15T tokens | Full Common Crawl processed with quality filters | [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| **RedPajama-V2** | 30T tokens | Full web corpus with 40+ quality annotations | [Together AI](https://together.ai/blog/redpajama-data-v2) |
| **The Pile** | 825 GB | Curated mix of 22 diverse sources | [HuggingFace](https://huggingface.co/datasets/EleutherAI/pile) |
| **Dolma** | 3T tokens | Open corpus for OLMo (Allen AI) | [HuggingFace](https://huggingface.co/datasets/allenai/dolma) |
| **C4** | 750 GB | Cleaned Common Crawl (original T5 data) | [HuggingFace](https://huggingface.co/datasets/allenai/c4) |

### Code Datasets

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **The Stack v2** | 6.4 TB | 358 programming languages, permissive licenses | [HuggingFace](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| **StarCoder Data** | 783 GB | Curated code for StarCoder models | [HuggingFace](https://huggingface.co/datasets/bigcode/starcoderdata) |

### Math Datasets

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **OpenWebMath** | 14.7B tokens | Math content filtered from Common Crawl | [HuggingFace](https://huggingface.co/datasets/open-web-math/open-web-math) |
| **MathPile** | 9.5B tokens | Textbooks, arXiv, StackExchange | [HuggingFace](https://huggingface.co/datasets/GAIR/MathPile) |
| **GSM8K** | 8.5K | Grade school math with solutions | [HuggingFace](https://huggingface.co/datasets/openai/gsm8k) |
| **MATH** | 12.5K | Competition math with step-by-step solutions | [HuggingFace](https://huggingface.co/datasets/hendrycks/competition_math) |
| **NuminaMath-CoT** | 860K | Chain-of-thought math solutions | [HuggingFace](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |

### Instruction / SFT Datasets

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **OpenAssistant** | 161K msgs | Human-written multi-turn conversations | [HuggingFace](https://huggingface.co/datasets/OpenAssistant/oasst2) |
| **UltraChat** | 1.5M convos | Multi-turn synthetic conversations | [HuggingFace](https://huggingface.co/datasets/stingning/ultrachat) |
| **WizardLM** | 196K | Evolved instructions for complex tasks | [HuggingFace](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) |
| **Orca-Math** | 200K | Math word problems for reasoning | [HuggingFace](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) |

### Tools & Libraries

| Tool | Purpose | Link |
|------|---------|------|
| **text-dedup** | Deduplication at scale | [GitHub](https://github.com/ChenghaoMou/text-dedup) |
| **datasketch** | MinHash / LSH implementation | [GitHub](https://github.com/ekzhu/datasketch) |
| **Trafilatura** | Web text extraction | [GitHub](https://github.com/adbar/trafilatura) |
| **fastText** | Language detection + quality classification | [GitHub](https://github.com/facebookresearch/fastText) |
| **tokenizers** | Fast BPE/WordPiece/Unigram training | [GitHub](https://github.com/huggingface/tokenizers) |
| **datatrove** | HuggingFace's data processing toolkit | [GitHub](https://github.com/huggingface/datatrove) |

---

## References

- [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) — Meta, 2024
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek AI, 2024
- [FineWeb: 15T Tokens of High-Quality Web Data](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) — HuggingFace, 2024
- [The Stack: 3TB of Source Code](https://arxiv.org/abs/2211.15533) — BigCode Project, 2022
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — Muennighoff et al., 2023
- [OpenWebMath: Open Dataset of High-Quality Mathematical Web Text](https://arxiv.org/abs/2310.06786) — Paster et al., 2023
- [RedPajama-V2: An Open Dataset with 30T Tokens](https://together.ai/blog/redpajama-data-v2) — Together AI, 2023
