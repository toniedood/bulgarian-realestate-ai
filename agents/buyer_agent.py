import json
import yaml
import vertexai
from vertexai.generative_models import GenerativeModel

# ─── LOAD CONFIG ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

vertexai.init(
    project=config["google_cloud"]["project_id"],
    location=config["google_cloud"]["location"]
)

gemini = GenerativeModel(config["google_cloud"]["model"])


class BuyerAgent:
    """
    An AI-powered buyer agent that evaluates properties and decides how to bid.

    Each agent has:
      - name / budget / preferences  -> WHO they are and WHAT they want
      - strategy                     -> HOW they behave in a bidding war
    """

    def __init__(self, name: str, budget: float, preferences: str, strategy: str):
        self.name         = name
        self.budget       = budget
        self.preferences  = preferences
        self.strategy     = strategy    # "conservative" | "aggressive" | "balanced"
        self.won_property = None        # filled in by the orchestrator when they win

    # ─── EVALUATE ────────────────────────────────────────────────────────────
    def evaluate_property(self, listing_text: str) -> dict:
        """
        Show the agent a property listing.
        Gemini responds AS that agent: do I want this? how much would I pay?

        Returns a dict:
          {
            "interested": bool,
            "interest_score": int (1-10),
            "max_willing_to_pay": float (EUR),
            "reasoning": str (in Bulgarian)
          }
        """
        prompt = f"""Ti si kupuvach na imot na targ s imeto {self.name}.
Tvoyat byudzhet e {self.budget:,.0f} EUR.
Tvoite predpochitaniya: {self.preferences}
Tvoyta strategiya: {self.strategy}

Procheti sledvashata obyava i reshi dali te interesva imota.

--- OBYAVA ---
{listing_text}
--- KRAY ---

Otgovori SAMO s validen JSON v slednia format, bez nikakvi obyasneniya izvyn nego:
{{
  "interested": true ili false,
  "interest_score": chislo ot 1 do 10,
  "max_willing_to_pay": chislo v EUR (ili 0 ako ne te interesva),
  "reasoning": "kratko obyasnenie na bylgarski"
}}"""

        prompt = f"""Ти си купувач на имот на търг с името {self.name}.
Твоят бюджет е {self.budget:,.0f} EUR.
Твоите предпочитания: {self.preferences}
Твоята стратегия: {self.strategy}

Прочети следната обява и реши дали те интересува имотът.

--- ОБЯВА ---
{listing_text}
--- КРАЙ ---

Отговори САМО с валиден JSON в следния формат, без никакви обяснения извън него:
{{
  "interested": true или false,
  "interest_score": число от 1 до 10,
  "max_willing_to_pay": число в EUR (или 0 ако не те интересува),
  "reasoning": "кратко обяснение на български"
}}"""

        response = gemini.generate_content(prompt)
        text     = response.text.strip()

        # Strip markdown code fences if Gemini wraps in ```json ... ```
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Fallback: agent passes if Gemini returns unexpected output
            return {
                "interested": False,
                "interest_score": 0,
                "max_willing_to_pay": 0,
                "reasoning": "Грешка при четене на отговора."
            }

    # ─── BID ─────────────────────────────────────────────────────────────────
    def make_bid(self, current_price: float, evaluation: dict) -> float | None:
        """
        Given the current auction price and the agent's evaluation,
        decide whether to bid and for how much.

        Returns a bid amount (float) or None to pass.

        Strategy logic:
          conservative -> small increments (+3%), backs off if score < 6
          aggressive   -> large increments (+8%), bids even on score 4+
          balanced     -> medium increments (+5%), only bids on score 7+
        """
        if not evaluation["interested"]:
            return None

        score   = evaluation["interest_score"]
        max_pay = min(evaluation["max_willing_to_pay"], self.budget)

        if current_price >= max_pay:
            return None

        if self.strategy == "conservative":
            if score < 6:
                return None
            bid = current_price * 1.03

        elif self.strategy == "aggressive":
            if score < 4:
                return None
            bid = current_price * 1.08

        else:  # balanced
            if score < 7:
                return None
            bid = current_price * 1.05

        bid = min(bid, max_pay)

        if bid <= current_price:
            return None

        return round(bid, -2)   # nearest 100 EUR

    def __repr__(self) -> str:
        return f"BuyerAgent({self.name}, budget={self.budget:,.0f} EUR, strategy={self.strategy})"


# ─── PREDEFINED AGENTS ───────────────────────────────────────────────────────
def create_agents() -> list[BuyerAgent]:
    """Return the three buyer agents for the auction."""
    return [
        BuyerAgent(
            name        = "Maria",
            budget      = 400_000,
            preferences = "Malki apartamenti v Sofia, do 2 spalny, tihi kvartali",
            strategy    = "conservative"
        ),
        BuyerAgent(
            name        = "Georgi",
            budget      = 800_000,
            preferences = "Prostorni imoti s morska gledka, Varna ili Burgas, luksozni",
            strategy    = "aggressive"
        ),
        BuyerAgent(
            name        = "Elena",
            budget      = 550_000,
            preferences = "Semeyni apartamenti s pone 3 spalny, dobri kvartali, vseki grad",
            strategy    = "balanced"
        ),
    ]
