import random
import yaml
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, END

from agents.buyer_agent import BuyerAgent, create_agents
from agents.orchestrator import parse_asking_price, parse_title

# ─── LOAD CONFIG ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

LISTINGS_FOLDER      = config["rag"]["listings_folder"]
NUMBER_OF_PROPERTIES = config["auction"]["number_of_properties"]
MAX_ROUNDS           = config["auction"]["max_rounds"]
STARTING_BID_PCT     = config["auction"]["starting_bid_percentage"]

SEP  = "─" * 60
SEP2 = "=" * 60


# ─── STATE ───────────────────────────────────────────────────────────────────
class AuctionState(TypedDict):
    """
    The single source of truth that flows through every node in the graph.
    Each node reads from it and returns only the fields it changes.
    """
    # Setup
    property_paths:   list[str]         # file paths of selected listings
    property_index:   int               # which property we're currently auctioning
    agents:           list[BuyerAgent]  # the 3 buyer agents

    # Current property
    current_id:       str
    current_text:     str
    asking_price:     float
    current_price:    float

    # Current auction round
    evaluations:       dict             # agent_name -> evaluation dict
    active_agents:     list[str]        # names of agents still bidding
    current_leader:    str | None       # name of current highest bidder
    round_number:      int
    bids_made:         bool             # were any bids made in last round?

    # Accumulated results
    results:           list[dict]


# ─── NODES ───────────────────────────────────────────────────────────────────

def initialize(state: AuctionState) -> dict:
    """Pick 5 random listings and set up the 3 buyer agents."""
    all_files = sorted(Path(LISTINGS_FOLDER).glob("*.md"))
    selected  = [str(p) for p in random.sample(all_files, min(NUMBER_OF_PROPERTIES, len(all_files)))]
    agents    = create_agents()

    print(f"\n{SEP2}")
    print(f"  БЪЛГАРСКИ ТЪРГ НА НЕДВИЖИМИ ИМОТИ  [LangGraph]")
    print(f"  Имоти: {len(selected)} | Купувачи: {len(agents)}")
    print(SEP2)
    for a in agents:
        print(f"  {a.name} | Бюджет: {a.budget:,.0f} EUR | Стратегия: {a.strategy}")

    return {
        "property_paths": selected,
        "property_index": 0,
        "agents":         agents,
        "results":        []
    }


def start_property(state: AuctionState) -> dict:
    """Load the current property and set the starting price."""
    path    = state["property_paths"][state["property_index"]]
    text    = Path(path).read_text(encoding="utf-8")
    prop_id = Path(path).stem
    price   = parse_asking_price(text) or 0
    start   = round(price * STARTING_BID_PCT, -2)

    print(f"\n{SEP}")
    print(f"  ИМОТ {state['property_index'] + 1}/{len(state['property_paths'])}: {parse_title(text)}")
    print(f"  Обява: {prop_id} | Питаща: {price:,.0f} EUR | Начална тръжна: {start:,.0f} EUR")
    print(SEP)

    return {
        "current_id":    prop_id,
        "current_text":  text,
        "asking_price":  price,
        "current_price": start,
        "evaluations":   {},
        "active_agents": [],
        "current_leader": None,
        "round_number":  0,
        "bids_made":     False
    }


def agents_evaluate(state: AuctionState) -> dict:
    """All agents evaluate the current property and decide if they're interested."""
    print(f"\n  Фаза 1 — Оценка на имота:\n")

    evaluations   = {}
    active_agents = []

    for agent in state["agents"]:
        ev     = agent.evaluate_property(state["current_text"])
        status = "заинтересован" if ev["interested"] else "не е заинтересован"
        evaluations[agent.name] = ev

        print(f"    {agent.name} ({agent.strategy}): {status} | score {ev['interest_score']}/10")
        print(f"      Макс. сума: {ev['max_willing_to_pay']:,.0f} EUR")
        print(f"      Мнение: {ev['reasoning']}\n")

        if ev["interested"] and ev["max_willing_to_pay"] >= state["current_price"]:
            active_agents.append(agent.name)

    if not active_agents:
        print(f"  Няма заинтересовани купувачи. Имотът не е продаден.\n")

    return {
        "evaluations":   evaluations,
        "active_agents": active_agents
    }


def run_round(state: AuctionState) -> dict:
    """Run one bidding round — each active agent bids or drops out."""
    round_num     = state["round_number"] + 1
    current_price = state["current_price"]
    current_leader = state["current_leader"]

    print(f"\n  Фаза 2 — Рунд {round_num} | Текуща цена: {current_price:,.0f} EUR")

    agent_map   = {a.name: a for a in state["agents"]}
    bids        = {}
    still_active = []

    for name in state["active_agents"]:
        agent = agent_map[name]
        bid   = agent.make_bid(current_price, state["evaluations"][name])
        if bid and bid > current_price:
            bids[name] = bid
            still_active.append(name)
            print(f"    {name}: оферира {bid:,.0f} EUR")
        else:
            print(f"    {name}: пасува и отпада")

    bids_made = bool(bids)

    if bids:
        winner_name    = max(bids, key=bids.__getitem__)
        current_price  = bids[winner_name]
        current_leader = winner_name
        print(f"    --> Лидер: {winner_name} с {current_price:,.0f} EUR")

    return {
        "round_number":   round_num,
        "current_price":  current_price,
        "current_leader": current_leader,
        "active_agents":  still_active,
        "bids_made":      bids_made
    }


def end_property(state: AuctionState) -> dict:
    """Record the result of this property's auction and advance to the next."""
    winner = state["current_leader"]
    price  = state["current_price"]

    print(f"\n{SEP}")
    if winner:
        diff     = price - state["asking_price"]
        diff_pct = (diff / state["asking_price"]) * 100
        sign     = "+" if diff >= 0 else ""
        print(f"  ПОБЕДИТЕЛ: {winner} купува '{parse_title(state['current_text'])}'")
        print(f"  ФИНАЛНА ЦЕНА: {price:,.0f} EUR  ({sign}{diff_pct:.1f}% от питащата)")

        # Mark the winning agent
        for agent in state["agents"]:
            if agent.name == winner:
                agent.won_property = state["current_id"]
    else:
        print(f"  РЕЗУЛТАТ: Имотът не беше продаден.")
    print(SEP)

    result = {
        "property_id":  state["current_id"],
        "title":        parse_title(state["current_text"]),
        "asking_price": state["asking_price"],
        "final_price":  price if winner else None,
        "winner":       winner,
        "rounds":       state["round_number"]
    }

    return {
        "results":        state["results"] + [result],
        "property_index": state["property_index"] + 1
    }


def summarize(state: AuctionState) -> dict:
    """Print the final summary of all auctions."""
    print(f"\n\n{SEP2}")
    print(f"  ФИНАЛНИ РЕЗУЛТАТИ")
    print(SEP2)

    for r in state["results"]:
        if r["winner"]:
            diff     = r["final_price"] - r["asking_price"]
            diff_pct = (diff / r["asking_price"]) * 100
            sign     = "+" if diff >= 0 else ""
            print(f"  {r['property_id']} | Победител: {r['winner']:<8} | "
                  f"{r['final_price']:>10,.0f} EUR  ({sign}{diff_pct:.1f}%)")
        else:
            print(f"  {r['property_id']} | Не е продаден")

    print(f"\n  Купувачи:")
    for agent in state["agents"]:
        if agent.won_property:
            print(f"    {agent.name}: спечели {agent.won_property}")
        else:
            print(f"    {agent.name}: не спечели имот")

    print(SEP2)
    return {}


# ─── CONDITIONAL EDGES ───────────────────────────────────────────────────────

def route_after_evaluate(state: AuctionState) -> str:
    """After evaluation: if no active agents, skip straight to end_property."""
    if not state["active_agents"]:
        return "end_property"
    return "run_round"


def route_after_round(state: AuctionState) -> str:
    """After a bidding round: continue or end this property's auction."""
    if not state["bids_made"]:
        return "end_property"        # no one bid → auction over
    if len(state["active_agents"]) <= 1:
        return "end_property"        # only one or zero bidders left → done
    if state["round_number"] >= MAX_ROUNDS:
        return "end_property"        # hit round limit
    return "run_round"               # keep going


def route_after_property(state: AuctionState) -> str:
    """After finishing a property: more to go, or wrap up?"""
    if state["property_index"] >= len(state["property_paths"]):
        return "summarize"
    return "start_property"


# ─── BUILD THE GRAPH ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AuctionState)

    # Add nodes
    graph.add_node("initialize",      initialize)
    graph.add_node("start_property",  start_property)
    graph.add_node("agents_evaluate", agents_evaluate)
    graph.add_node("run_round",       run_round)
    graph.add_node("end_property",    end_property)
    graph.add_node("summarize",       summarize)

    # Linear edges
    graph.add_edge("initialize",      "start_property")
    graph.add_edge("start_property",  "agents_evaluate")

    # Conditional edges (the decision points)
    graph.add_conditional_edges("agents_evaluate", route_after_evaluate, {
        "run_round":    "run_round",
        "end_property": "end_property"
    })
    graph.add_conditional_edges("run_round", route_after_round, {
        "run_round":    "run_round",
        "end_property": "end_property"
    })
    graph.add_conditional_edges("end_property", route_after_property, {
        "start_property": "start_property",
        "summarize":      "summarize"
    })

    graph.add_edge("summarize", END)

    # Entry point
    graph.set_entry_point("initialize")

    return graph.compile()


# ─── RUN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    auction = build_graph()
    auction.invoke({})
