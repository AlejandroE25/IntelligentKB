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

You can customise the host and port with environment variables:

```bash
HOST=0.0.0.0 PORT=8080 python main.py
```

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
