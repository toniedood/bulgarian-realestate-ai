import yaml
import vertexai
from vertexai.generative_models import GenerativeModel
from rag.search import search

# ─── LOAD CONFIG ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

vertexai.init(
    project=config["google_cloud"]["project_id"],
    location=config["google_cloud"]["location"]
)

gemini = GenerativeModel(config["google_cloud"]["model"])


def ask(question: str) -> str:
    """
    Full RAG pipeline:
      1. Search ChromaDB for relevant listing chunks
      2. Format them as context
      3. Ask Gemini to answer using that context
    """
    hits = search(question)

    if not hits:
        return "Не намерих подходящи имоти за вашето запитване."

    # Build context block from retrieved chunks
    context_parts = []
    for i, hit in enumerate(hits, 1):
        text = hit.get("full_text", hit["text"])
        context_parts.append(f"[Имот {i} — {hit['listing_id']}]\n{text}")

    context = "\n\n".join(context_parts)

    prompt = f"""Ти си асистент за недвижими имоти в България.
Използвай само информацията от обявите по-долу, за да отговориш на въпроса.
Ако информацията не е достатъчна, кажи го честно.

--- ОБЯВИ ---
{context}
--- КРАЙ НА ОБЯВИТЕ ---

Въпрос: {question}

Отговор:"""

    response = gemini.generate_content(prompt)

    return response.text


# ─── INTERACTIVE MODE ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🏠 Bulgarian Real Estate AI — RAG Pipeline")
    print("   Задайте въпрос на български или английски. Напишете 'изход' за край.\n")

    while True:
        question = input("Въпрос: ").strip()
        if question.lower() in ("изход", "exit", "quit"):
            print("Довиждане!")
            break
        if not question:
            continue

        print("\n⏳ Търся...\n")
        answer = ask(question)
        print(f"💬 Отговор:\n{answer}\n")
        print("─" * 60 + "\n")
