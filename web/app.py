import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import random
from pathlib import Path

import yaml
from agents.buyer_agent import create_agents
from agents.orchestrator import parse_asking_price, parse_title
from rag.pipeline import ask as rag_ask
from rag.search import search as rag_search

# ─── LOAD CONFIG ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

LISTINGS_FOLDER      = config["rag"]["listings_folder"]
NUMBER_OF_PROPERTIES = config["auction"]["number_of_properties"]
STARTING_BID_PCT     = config["auction"]["starting_bid_percentage"]
MAX_ROUNDS           = config["auction"]["max_rounds"]

app = FastAPI(title="Bulgarian Real Estate AI")

# ─── SERVE FRONTEND ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    question: str

@app.post("/search")
def search(req: SearchRequest):
    """RAG search — returns Gemini answer + the matching listings used."""
    answer = rag_ask(req.question)
    hits   = rag_search(req.question)

    sources = [
        {"listing_id": h["listing_id"], "chunk_type": h["chunk_type"], "preview": h["text"][:200]}
        for h in hits
    ]
    return JSONResponse({"answer": answer, "sources": sources})


@app.post("/auction")
def run_auction():
    """
    Run a full auction with 5 random properties and 3 buyer agents.
    Returns a structured result for the frontend to render.
    """
    agents      = create_agents()
    all_files   = sorted(Path(LISTINGS_FOLDER).glob("*.md"))
    selected    = random.sample(all_files, min(NUMBER_OF_PROPERTIES, len(all_files)))

    auction_log = []   # per-property results
    agent_map   = {a.name: a for a in agents}

    for filepath in selected:
        text      = filepath.read_text(encoding="utf-8")
        prop_id   = filepath.stem
        title     = parse_title(text)
        asking    = parse_asking_price(text) or 0
        start     = round(asking * STARTING_BID_PCT, -2)

        # Agents evaluate
        evaluations   = {}
        eval_summaries = []
        for agent in agents:
            ev = agent.evaluate_property(text)
            evaluations[agent.name] = ev
            eval_summaries.append({
                "agent":      agent.name,
                "strategy":   agent.strategy,
                "interested": ev["interested"],
                "score":      ev["interest_score"],
                "max_pay":    ev["max_willing_to_pay"],
                "reasoning":  ev["reasoning"]
            })

        # Filter active agents
        active = [
            a for a in agents
            if evaluations[a.name]["interested"]
            and evaluations[a.name]["max_willing_to_pay"] >= start
        ]

        # Bidding rounds
        current_price  = start
        current_leader = None
        rounds         = []

        for round_num in range(1, MAX_ROUNDS + 1):
            bids        = {}
            still_active = []
            round_log   = {"round": round_num, "price_before": current_price, "bids": []}

            for agent in active[:]:
                bid = agent.make_bid(current_price, evaluations[agent.name])
                if bid and bid > current_price:
                    bids[agent.name] = bid
                    still_active.append(agent)
                    round_log["bids"].append({"agent": agent.name, "bid": bid, "passed": False})
                else:
                    round_log["bids"].append({"agent": agent.name, "bid": None, "passed": True})

            if not bids:
                rounds.append(round_log)
                break

            winner_name    = max(bids, key=bids.__getitem__)
            current_price  = bids[winner_name]
            current_leader = winner_name
            active         = still_active
            round_log["leader"] = winner_name
            round_log["price_after"] = current_price
            rounds.append(round_log)

            if len(active) <= 1:
                break

        # Mark winning agent
        if current_leader:
            agent_map[current_leader].won_property = prop_id

        auction_log.append({
            "property_id":   prop_id,
            "title":         title,
            "asking_price":  asking,
            "starting_price": start,
            "final_price":   current_price if current_leader else None,
            "winner":        current_leader,
            "rounds_played": len(rounds),
            "evaluations":   eval_summaries,
            "rounds":        rounds
        })

    return JSONResponse({
        "agents": [
            {"name": a.name, "budget": a.budget, "strategy": a.strategy, "won": a.won_property}
            for a in agents
        ],
        "properties": auction_log
    })


@app.get("/listings")
def list_properties():
    """Return a summary of all 60 listings for browsing."""
    files   = sorted(Path(LISTINGS_FOLDER).glob("*.md"))
    listing_list = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        listing_list.append({
            "id":    f.stem,
            "title": parse_title(text),
            "price": parse_asking_price(text)
        })
    return JSONResponse({"listings": listing_list})
