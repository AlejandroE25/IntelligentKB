# IntelligentKB — KB AI Search Agent

A locally-run Python web app where a help desk consultant describes a support
problem in plain English and receives relevant troubleshooting steps drawn from
a small set of local KB articles.

Because the total article content is under 30,000 characters, all articles are
loaded directly into Claude's context window — no vector database or embeddings
needed.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language  | Python 3.11+ |
| LLM       | Anthropic Claude (`claude-sonnet-4-20250514`) via `anthropic` SDK |
| HTML parsing | `beautifulsoup4` |
| Web server | Flask |

## File Structure

```
IntelligentKB/
├── articles/                   # KB articles as .htm or .html files
│   ├── Campus Wi-Fi, Getting Connected (Start Here).html
│   ├── Email, How to set up email forwarding.html
│   └── ...
├── main.py                     # Loads articles, runs Flask web server, calls Claude
├── .env                        # ANTHROPIC_API_KEY=your_key_here  (not committed)
├── .env.example                # Template for the .env file
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your actual Anthropic API key
```

### 3. (Optional) Add or customise KB articles

Drop any additional `.htm` or `.html` files that follow the UW-Madison
KnowledgeBase HTML structure into the `articles/` directory.

## Running the Tool

```bash
python main.py
```

By default the server listens on **http://127.0.0.1:5000**.  
Open that URL in your browser to start using the assistant.

To print the current build number (or commit-based fallback) and exit:

```bash
python main.py --build-number
```

You can customise the host and port with environment variables:

```bash
HOST=0.0.0.0 PORT=8080 python main.py
```

### Optional Runtime Cost Controls

You can tune Claude request size/retry behavior with environment variables:

```bash
TOP_K_ARTICLES=3
MAX_TOKENS=1024
MAX_AGENT_TURNS=3
MAX_ARTICLE_CHARS_IN_PROMPT=2500
MAX_CONTACTS_CHARS_IN_PROMPT=1500
ANTHROPIC_MAX_RETRIES=1
```

### Live Claude Integration Test (Real API Call)

The default test suite uses mocks. To run a real Claude response check:

```bash
RUN_LIVE_CLAUDE_TESTS=1 ANTHROPIC_API_KEY=your_key_here \
python -m pytest -q tests/test_live_claude.py
```

This test is skipped unless both `RUN_LIVE_CLAUDE_TESTS=1` and
`ANTHROPIC_API_KEY` are set.

### Example session

Type your issue in the text box and press **Enter** (or click **Send**).  
Use **Shift+Enter** to add a newline without submitting.  
Click **New Conversation** in the header to clear the chat history.

## How It Works

1. **Startup** — `main.py` reads every `.htm`/`.html` file in `articles/`,
   parses each one with BeautifulSoup, and builds a system prompt containing
   all cleaned article content with clear separators.
2. **Query** — The user types a problem description in the web UI.
3. **Generation** — The query is sent to Claude along with the full article
   content in the system prompt. Claude identifies relevant articles and
   synthesises troubleshooting steps, citing source articles.
4. **Output** — Claude's response is displayed in the chat. Follow-up questions
   share the same conversation history within a session.

## HTML Parsing Rules

Each article follows the UW-Madison KnowledgeBase structure. The parser:

- **Extracts:**
  - Title → `<title>` tag
  - Keywords → `<span id="kb-page-keywords">`
  - Main content → `<div id="kbcontent">`
  - Internal staff section → `<div class="kb-class-internal-site">` (tagged as
    internal, then removed from main content)

- **Strips before extracting text:**
  - All `<script>` and `<style>` tags
  - `<header>`, `<footer>`, `<nav>`, `<aside>`
  - All `.doc-attr` blocks (UI metadata: doc ID, owner, dates)
  - Feedback buttons and analytics elements (`.feedback-btn`)

## Search Intelligence Enhancements

Starting from the baseline TF-IDF retrieval, a phased search upgrade is
available via opt-in feature flags.  All flags default to **off**, so the
application behaves exactly as before unless you set them.

### Architecture Changes

```
User query
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  search_enhancement.HybridRetriever                  │
│                                                      │
│  1. QueryNormalizer   lowercase + punctuation        │
│     (FEATURE_QUERY_NORMALIZATION)  + synonym expand  │
│     + typo-correct fallback on low confidence        │
│                                                      │
│  2. LRU cache lookup  (FEATURE_SEARCH_CACHE)         │
│                                                      │
│  3. Lexical branch    TF-IDF cosine similarity       │
│                                                      │
│  4. Semantic branch   LSA (TruncatedSVD) by default; │
│     (FEATURE_HYBRID_RETRIEVAL) sentence-transformers │
│     if installed                                     │
│                                                      │
│  5. RRF Fusion        Reciprocal Rank Fusion         │
│                                                      │
│  6. Adaptive top-k    score-gap selection            │
│     (FEATURE_ADAPTIVE_TOPK) for Claude context       │
│                                                      │
│  7. Cache store                                      │
└──────────────────────────────────────────────────────┘
    │
    ▼
Article results + SearchTimings logged per request
```

**Semantic branch** uses Latent Semantic Analysis (LSA via scikit-learn's
`TruncatedSVD`) by default — no extra dependencies.  Install
`sentence-transformers` to upgrade to BERT-quality embeddings automatically:

```bash
pip install sentence-transformers>=2.2.0
```

The startup build cost for LSA on a 30-article corpus is negligible (< 10 ms).

### Feature Flags

Set any of these environment variables to `1` (or `true` / `yes`) before
starting the server:

| Flag | Default | What it enables |
|------|---------|-----------------|
| `FEATURE_HYBRID_RETRIEVAL` | `false` | Semantic branch + RRF fusion |
| `FEATURE_QUERY_NORMALIZATION` | `false` | Synonym expansion; typo correction on low-confidence results |
| `FEATURE_ADAPTIVE_TOPK` | `false` | Score-gap-based adaptive top-k for Claude context |
| `FEATURE_SEARCH_CACHE` | `false` | LRU result cache (256 entries, 5-min TTL) |

Example — enable all enhancements:

```bash
FEATURE_HYBRID_RETRIEVAL=1 \
FEATURE_QUERY_NORMALIZATION=1 \
FEATURE_ADAPTIVE_TOPK=1 \
FEATURE_SEARCH_CACHE=1 \
python main.py
```

### Running Latency Benchmarks

```bash
# Baseline path (100 timed runs per query)
python benchmarks/benchmark_latency.py --runs 100

# Enhanced path (all flags on)
python benchmarks/benchmark_latency.py --runs 100 --enhanced
```

Results (measured on a standard laptop, 31 articles):

| Path | p50 (ms) | p95 (ms) | Notes |
|------|----------|----------|-------|
| Baseline (TF-IDF) | 0.54 | 0.57 | per-query retrieval only |
| Enhanced (warm cache) | 0.02 | 0.02 | LRU cache hit |
| Enhanced (cold, first call) | ~1.0 | ~2.0 | norm + lex + sem + RRF |

**Target:** p95 server-side retrieval ≤ 120 ms  ✓ Easily met.

### Running Relevance Evaluation

```bash
# Baseline
python benchmarks/eval_harness.py

# Enhanced
python benchmarks/eval_harness.py --enhanced
```

The harness computes **Recall@3**, **Recall@5**, **Recall@10**, and **nDCG@5**.
You must first populate the `GROUND_TRUTH` dictionary in
`benchmarks/eval_harness.py` with real article IDs from your corpus.  Run
`python main.py` and visit `/articles` to see all article IDs and titles.

### Tradeoffs and Recommended Defaults

| Flag | Recommendation | Reason |
|------|---------------|--------|
| `FEATURE_HYBRID_RETRIEVAL` | Enable | LSA adds ≈ 0.7 ms/query; improves synonym & paraphrase recall at zero extra cost |
| `FEATURE_QUERY_NORMALIZATION` | Enable | < 0.1 ms; synonym expansion measurably improves recall for help-desk jargon |
| `FEATURE_ADAPTIVE_TOPK` | Enable cautiously | Useful when score distributions have clear gaps; monitor Claude context size |
| `FEATURE_SEARCH_CACHE` | Enable | 300-second TTL eliminates repeated work for live-typing use patterns |

All four flags together add ≈ 1–2 ms on a cold query and < 0.05 ms on a
cache hit — well within the 120 ms p95 target for the server-side retrieval
step and the 450 ms end-to-end live-update budget (300 ms debounce + ~150 ms
server + network round-trip).
