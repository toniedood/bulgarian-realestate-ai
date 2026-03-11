# Bulgarian Real Estate AI

An AI-powered real estate auction system built with Gemini 2.0 Flash, RAG, multi-agent reasoning, and LangGraph — built as a pre-entry technical assessment for an AI Developer role.

---

## What This Is

The system simulates a Bulgarian property auction where three AI buyer agents read real estate listings, evaluate them based on their own budgets and preferences, and compete in a live bidding process. A FastAPI web interface lets you search listings in natural language and watch auctions run in real time.

---

## Architecture

```
60 Bulgarian property listings (Markdown)
              │
              ▼
   rmihaylov/roberta-base-nli-stsb-bg     ← Bulgarian-native sentence embedding model
              │   (2 chunks per listing: structured header + narrative)
              ▼
         ChromaDB                         ← vector store (cosine similarity)
              │
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
 RAG Pipeline        Auction System
 (rag/)              (agents/)
    │                    │
    │  ask(question)      │  LangGraph state machine
    │       │             │       │
    ▼       ▼             ▼       ▼
 Search  Gemini        BuyerAgents  Orchestrator
 chunks  answers       evaluate     manages
 in DB   in BG         & bid        rounds
    │                    │
    └─────────┬──────────┘
              ▼
         FastAPI + HTML
         (web/)
         localhost:8000
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Gemini 2.0 Flash via Vertex AI | Answer generation, agent reasoning |
| Embeddings | `rmihaylov/roberta-base-nli-stsb-bg` | Bulgarian-native sentence vectors |
| Vector DB | ChromaDB | Semantic search over listings |
| Agent Framework | LangGraph | Stateful multi-step auction orchestration |
| Backend | FastAPI + Uvicorn | REST API + serves frontend |
| Language | Python 3.11.9 | Runtime (3.13 incompatible with torch) |
| Cloud | Google Cloud / Vertex AI | Gemini API with $300 credit tier |

---

## Project Structure

```
bulgarian-realestate-ai/
│
├── config/
│   └── settings.yaml           # All config: model names, auction params, RAG settings
│
├── data/
│   ├── generate_bulgarian_listings.py   # Generates 60 listings via Gemini (M1)
│   └── listings/               # 60 Bulgarian property listings (imot_001.md … imot_060.md)
│
├── rag/
│   ├── ingest.py               # Embeds all listings → stores 120 chunks in ChromaDB (M2)
│   ├── search.py               # Embeds a query → retrieves top-N matching chunks
│   └── pipeline.py             # search() + Gemini → natural language answer
│
├── agents/
│   ├── buyer_agent.py          # BuyerAgent class: evaluate property, decide bid (S1)
│   ├── orchestrator.py         # Runs auction loop without LangGraph (S2)
│   └── auction_graph.py        # Full LangGraph auction: nodes, edges, state (S3)
│
├── web/
│   ├── app.py                  # FastAPI server: /search, /auction, /listings (M3)
│   └── index.html              # Single-page frontend: search + auction + browse tabs
│
├── vector_db/                  # ChromaDB persistent storage (created by ingest.py)
├── venv/                       # Python virtual environment
├── test_gemini.py              # Sanity check: Gemini responds in Bulgarian
└── progress.txt                # Session notes (Bulgarian)
```

---

## Setup & Run

### Prerequisites
- Python 3.11.x (3.13 is incompatible with PyTorch)
- Google Cloud account with Vertex AI enabled
- `gcloud` CLI authenticated (`gcloud auth application-default login`)

### 1. Clone and create virtual environment

```bash
git clone https://github.com/toniedood/bulgarian-realestate-ai.git
cd bulgarian-realestate-ai
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install \
  chromadb \
  torch \
  "transformers==4.44.2" \
  google-cloud-aiplatform \
  google-genai \
  vertexai \
  langgraph \
  fastapi \
  uvicorn \
  pyyaml \
  python-dotenv
```

### 3. Configure Google Cloud

Update `config/settings.yaml` with your project ID:

```yaml
google_cloud:
  project_id: your-gcp-project-id
  location: global
  model: gemini-2.0-flash-001
```

### 4. Generate listings (M1 — already done, skip if cloning)

```bash
python data/generate_bulgarian_listings.py
```

### 5. Build the vector database (M2)

```bash
python rag/ingest.py
```

This reads all 60 listings, embeds them, and stores 120 chunks in ChromaDB.

### 6. Start the web interface (M3)

```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Features

### Property Search (RAG)
Type a question in Bulgarian or English. The system embeds your query, retrieves the most semantically similar listing chunks from ChromaDB, and sends them to Gemini to generate a grounded answer. The listings used as context are shown below the answer.

### AI Auction Simulation (LangGraph)
Three buyer agents with distinct personalities compete for 5 randomly selected properties:

| Agent | Budget | Strategy | Preference |
|-------|--------|----------|-----------|
| Мария | 400,000 EUR | Conservative (+3% bids, score ≥ 6) | Small apartments, Sofia |
| Георги | 800,000 EUR | Aggressive (+8% bids, score ≥ 4) | Luxury sea-view, Varna/Burgas |
| Елена | 550,000 EUR | Balanced (+5% bids, score ≥ 7) | 3+ bedroom family homes |

Each agent uses Gemini to evaluate a listing from their own perspective, returning a structured JSON response with an interest score, maximum willingness to pay, and reasoning in Bulgarian.

### Browse All Listings
View all 60 generated properties with titles and asking prices.

---

## Design Decisions

**Why `rmihaylov/roberta-base-nli-stsb-bg` instead of a multilingual model?**
This model is trained specifically on Bulgarian using NLI + STS-B objectives — exactly the tasks that make embeddings good for semantic search. A multilingual model like `paraphrase-multilingual-MiniLM-L12-v2` spreads its capacity across 50+ languages; this one focuses entirely on Bulgarian, producing sharper semantic matches for Bulgarian queries.

**Why two chunks per listing instead of one?**
Each listing has two semantically distinct zones: a structured header (facts: price, size, floor) and a narrative description (location, features, atmosphere). Splitting them allows the retriever to match on either factual criteria ("3 bedrooms under 500k") or descriptive language ("quiet neighbourhood near the sea") independently. Merging them into one chunk would dilute both signals.

**Why Vertex AI instead of the direct Gemini API key?**
The direct API (ai.google.dev) has strict free-tier quotas that hit rate limits almost immediately during development. Vertex AI uses Google Cloud billing, which provided $300 in credits — enough to run the full pipeline repeatedly without throttling.

**Why LangGraph over a plain loop?**
The orchestrator works as a simple for-loop, but LangGraph makes the auction state explicit and inspectable at every node. Each decision point (should this round continue? are there more properties?) is a named conditional edge rather than buried logic. This pattern scales to human-in-the-loop, streaming output, and checkpointing — which a plain loop cannot do.
