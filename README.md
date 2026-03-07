# IntelligentKB — KB AI Search Agent

A locally-run Python CLI tool where a help desk consultant describes a support
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
| CLI       | Plain `input()` loop, no framework |

## File Structure

```
IntelligentKB/
├── articles/                   # 5–10 KB articles as .htm or .html files
│   ├── email_forwarding.htm
│   ├── vpn_setup.htm
│   ├── password_reset.htm
│   ├── wifi_troubleshooting.htm
│   ├── two_factor_auth.htm
│   ├── network_troubleshooting.htm
│   └── office365_setup.htm
├── main.py                     # Loads and parses articles, runs CLI loop, calls Claude
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
# Edit .env and replace "your_api_key_here" with your actual Anthropic API key
```

### 3. (Optional) Add or customise KB articles

Drop any additional `.htm` or `.html` files that follow the UW-Madison
KnowledgeBase HTML structure into the `articles/` directory.

## Running the Tool

```bash
python main.py
```

### Example session

```
Loaded 7 KB article(s): email_forwarding.htm, network_troubleshooting.htm, ...
KB AI Search Agent — type your support issue and press Enter.
Type 'quit' or 'exit' to stop.

Enter your support issue: A user can't connect to the VPN from off campus.

Based on the KB articles:

1. Confirm the user has Cisco AnyConnect installed. (Source: vpn_setup.htm)
2. Verify they are using their NetID credentials (username only, without
   @university.edu) and that the password has not expired. (Source: vpn_setup.htm)
3. If the connection times out, have them temporarily disable any third-party
   firewall or antivirus. (Source: network_troubleshooting.htm)
4. If a Duo push is not received, the user can use a passcode from the Duo Mobile
   app instead. (Source: two_factor_auth.htm)

If the issue persists, escalate to the networking team.

Enter your support issue: exit
Goodbye!
```

## How It Works

1. **Startup** — `main.py` reads every `.htm`/`.html` file in `articles/`,
   parses each one with BeautifulSoup, and builds a system prompt containing
   all cleaned article content with clear separators.
2. **Query** — The user types a problem description into the terminal.
3. **Generation** — The query is sent to Claude along with the full article
   content in the system prompt. Claude identifies relevant articles and
   synthesises troubleshooting steps, citing source articles.
4. **Output** — Claude's response is printed. The loop continues for follow-up
   questions with full conversation history maintained across turns.

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
