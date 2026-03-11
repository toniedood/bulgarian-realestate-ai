import re
import random
import yaml
from pathlib import Path
from agents.buyer_agent import BuyerAgent, create_agents

# ─── LOAD CONFIG ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

LISTINGS_FOLDER       = config["rag"]["listings_folder"]
NUMBER_OF_PROPERTIES  = config["auction"]["number_of_properties"]
MAX_ROUNDS            = config["auction"]["max_rounds"]
STARTING_BID_PCT      = config["auction"]["starting_bid_percentage"]

SEP = "─" * 60


def parse_asking_price(listing_text: str) -> float | None:
    """
    Extract the asking price from a listing's markdown header.
    Looks for: **Цена:** 1,147,000 EUR
    Returns the price as a float, or None if not found.
    """
    match = re.search(r"\*\*Цена:\*\*\s*([\d,\.]+)", listing_text)
    if not match:
        return None
    price_str = match.group(1).replace(",", "").replace(".", "")
    return float(price_str)


def parse_title(listing_text: str) -> str:
    """Extract the property title (second ## heading) from a listing."""
    match = re.search(r"^##\s+(.+)$", listing_text, re.MULTILINE)
    return match.group(1).strip() if match else "Неизвестен имот"


def run_auction(property_id: str, listing_text: str, agents: list[BuyerAgent]) -> dict:
    """
    Run a single property auction.

    Flow:
      1. All agents evaluate the property
      2. Drop agents who are not interested or can't afford starting price
      3. Run bidding rounds until one winner remains or everyone drops out
      4. Return the result

    Returns a dict with: property_id, title, asking_price, final_price, winner, rounds
    """
    title        = parse_title(listing_text)
    asking_price = parse_asking_price(listing_text)

    if asking_price is None:
        return {"property_id": property_id, "title": title, "error": "Не намерих цена в обявата."}

    starting_price = round(asking_price * STARTING_BID_PCT, -2)

    print(f"\n{SEP}")
    print(f"  ИМОТ: {title}")
    print(f"  Обява: {property_id} | Цена: {asking_price:,.0f} EUR | Начална тръжна: {starting_price:,.0f} EUR")
    print(SEP)

    # ── Step 1: All agents evaluate ───────────────────────────────────────────
    print("\n  Фаза 1 — Оценка на имота от купувачите:\n")
    evaluations = {}
    for agent in agents:
        ev = agent.evaluate_property(listing_text)
        evaluations[agent.name] = ev
        status = "заинтересован" if ev["interested"] else "не е заинтересован"
        print(f"    {agent.name} ({agent.strategy}): {status} | score {ev['interest_score']}/10")
        print(f"      Макс. сума: {ev['max_willing_to_pay']:,.0f} EUR")
        print(f"      Мнение: {ev['reasoning']}\n")

    # ── Step 2: Filter agents who can participate ─────────────────────────────
    active = [
        a for a in agents
        if evaluations[a.name]["interested"]
        and evaluations[a.name]["max_willing_to_pay"] >= starting_price
    ]

    if not active:
        print("  Резултат: Няма заинтересовани купувачи. Имотът не е продаден.\n")
        return {
            "property_id":  property_id,
            "title":        title,
            "asking_price": asking_price,
            "final_price":  None,
            "winner":       None,
            "rounds":       0
        }

    # ── Step 3: Bidding rounds ─────────────────────────────────────────────────
    print(f"\n  Фаза 2 — Наддаване (начална цена: {starting_price:,.0f} EUR):\n")

    current_price  = starting_price
    current_leader = None
    rounds_played  = 0

    for round_num in range(1, MAX_ROUNDS + 1):
        rounds_played = round_num
        print(f"    Рунд {round_num} | Текуща цена: {current_price:,.0f} EUR")

        round_bids = {}
        passed     = []

        for agent in active[:]:   # iterate a copy so we can modify active
            bid = agent.make_bid(current_price, evaluations[agent.name])
            if bid and bid > current_price:
                round_bids[agent.name] = bid
                print(f"      {agent.name}: оферира {bid:,.0f} EUR")
            else:
                passed.append(agent.name)
                active.remove(agent)
                print(f"      {agent.name}: пасува и отпада")

        if not round_bids:
            print(f"\n    Няма оферти в рунд {round_num}. Търгът приключва.")
            break

        # Highest bid this round wins
        winner_name = max(round_bids, key=round_bids.__getitem__)
        current_price  = round_bids[winner_name]
        current_leader = next(a for a in agents if a.name == winner_name)
        print(f"      --> Лидер: {winner_name} с {current_price:,.0f} EUR")

        if len(active) == 1:
            print(f"\n    Остана само един купувач. Търгът приключва.")
            break

        if len(active) == 0:
            break

    # ── Step 4: Result ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    if current_leader:
        current_leader.won_property = property_id
        print(f"  ПОБЕДИТЕЛ: {current_leader.name} купува '{title}'")
        print(f"  ФИНАЛНА ЦЕНА: {current_price:,.0f} EUR  (питаща: {asking_price:,.0f} EUR)")
    else:
        print(f"  РЕЗУЛТАТ: Имотът не беше продаден.")
    print(SEP)

    return {
        "property_id":  property_id,
        "title":        title,
        "asking_price": asking_price,
        "final_price":  current_price if current_leader else None,
        "winner":       current_leader.name if current_leader else None,
        "rounds":       rounds_played
    }


def run_full_auction():
    """
    Main entry point.
    Picks 5 random listings, runs an auction for each, prints final summary.
    """
    agents = create_agents()
    all_files = sorted(Path(LISTINGS_FOLDER).glob("*.md"))
    selected  = random.sample(all_files, min(NUMBER_OF_PROPERTIES, len(all_files)))

    print(f"\n{'=' * 60}")
    print(f"  БЪЛГАРСКИ ТЪРГ НА НЕДВИЖИМИ ИМОТИ")
    print(f"  Имоти: {len(selected)} | Купувачи: {len(agents)}")
    print(f"{'=' * 60}")
    for agent in agents:
        print(f"  {agent.name} | Бюджет: {agent.budget:,.0f} EUR | Стратегия: {agent.strategy}")

    results = []
    for filepath in selected:
        listing_text = filepath.read_text(encoding="utf-8")
        result = run_auction(filepath.stem, listing_text, agents)
        results.append(result)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 60}")
    print(f"  ФИНАЛНИ РЕЗУЛТАТИ")
    print(f"{'=' * 60}")

    for r in results:
        if r.get("winner"):
            diff     = r["final_price"] - r["asking_price"]
            diff_pct = (diff / r["asking_price"]) * 100
            sign     = "+" if diff >= 0 else ""
            print(f"  {r['property_id']} | Победител: {r['winner']:<8} | "
                  f"{r['final_price']:>10,.0f} EUR  ({sign}{diff_pct:.1f}% от питащата)")
        else:
            print(f"  {r['property_id']} | Не е продаден")

    print(f"\n  Купувачи с спечелени имоти:")
    for agent in agents:
        if agent.won_property:
            print(f"    {agent.name}: {agent.won_property}")
        else:
            print(f"    {agent.name}: не спечели имот")

    print(f"{'=' * 60}\n")
    return results


if __name__ == "__main__":
    run_full_auction()
