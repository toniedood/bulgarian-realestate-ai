import torch
import chromadb
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# ─── LOAD CONFIG ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

VECTOR_DB_PATH  = config["rag"]["vector_db_path"]
COLLECTION_NAME = config["rag"]["collection_name"]
N_RESULTS       = config["rag"]["n_results"]
LISTINGS_FOLDER = config["rag"]["listings_folder"]
EMBEDDING_MODEL = "rmihaylov/roberta-base-nli-stsb-bg"
MAX_TOKENS      = 512

# ─── LOAD EMBEDDING MODEL ────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model     = AutoModel.from_pretrained(EMBEDDING_MODEL)
model.eval()


def embed(text: str) -> list[float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask   = inputs["attention_mask"]
    mask_expanded    = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings   = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask         = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return (sum_embeddings / sum_mask).squeeze().numpy().tolist()


KNOWN_CITIES = [
    "София", "Пловдив", "Варна", "Бургас", "Русе",
    "Стара Загора", "Плевен", "Велико Търново"
]

KNOWN_NEIGHBORHOODS = [
    "Лозенец", "Витоша", "Младост", "Иван Вазов", "Борово", "Овча купел",
    "Кършияка", "Тракия", "Захарна фабрика", "Христо Смирненски",
    "Чайка", "Левски", "Владиславово", "Бриз", "Аспарухово",
    "Лазур", "Славейков", "Меден рудник", "Сарафово",
    "Центъра", "Дружба", "Родина", "Чародейка", "Възраждане",
    "Самара", "Зора", "Железник", "Кайлъка",
    "Картала", "Бойчеви колиби", "Мл. Кооператор",
]


def detect_location_filter(query: str) -> dict:
    """
    Check if the query mentions a specific city or neighborhood by name.
    Returns a ChromaDB where clause if a match is found, otherwise {}.
    This handles proper nouns that semantic embeddings struggle with.
    """
    query_lower = query.lower()
    conditions  = []

    for city in KNOWN_CITIES:
        if city.lower() in query_lower:
            conditions.append({"city": city})
            break

    for hood in KNOWN_NEIGHBORHOODS:
        if hood.lower() in query_lower:
            conditions.append({"neighborhood": hood})
            break

    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search(query: str, n_results: int = N_RESULTS) -> list[dict]:
    """
    Search across ALL chunk types (price, city, neighborhood, narrative).

    1. Detect if the query mentions a specific city/neighborhood by name
       → if yes, pre-filter the DB to only those listings (fixes proper noun problem)
    2. Run semantic search on the (filtered) results
    3. Deduplicate by listing_id, keeping the best-scoring chunk per listing
    4. Fetch full listing text so Gemini has complete context
    """
    client     = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embed(query)
    where_filter    = detect_location_filter(query)

    # Retrieve more candidates than needed — we'll deduplicate down to n_results
    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(n_results * 4, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    if where_filter:
        query_kwargs["where"] = where_filter

    raw = collection.query(**query_kwargs)

    # Deduplicate: keep only the best (lowest distance) chunk per listing
    seen     = {}
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0]
    ):
        lid = meta["listing_id"]
        if lid not in seen or dist < seen[lid]["distance"]:
            seen[lid] = {
                "listing_id": lid,
                "chunk_type": meta["chunk_type"],
                "text":       doc,
                "distance":   round(dist, 4),
                "file":       meta.get("file", ""),
                "city":       meta.get("city", ""),
                "price":      meta.get("price", 0),
                "bedrooms":   meta.get("bedrooms", "?"),
            }

    # Sort by distance and take top n_results
    ranked = sorted(seen.values(), key=lambda x: x["distance"])[:n_results]

    # Fetch full listing text for each result so Gemini has complete context
    for hit in ranked:
        path = Path(hit["file"])
        if path.exists():
            hit["full_text"] = path.read_text(encoding="utf-8")
        else:
            hit["full_text"] = hit["text"]

    return ranked


# ─── QUICK TEST ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "апартамент с морска гледка",
        "евтин имот в София",
        "3 спални в тих квартал",
    ]
    for q in queries:
        print(f"\n🔍 {q}")
        for i, hit in enumerate(search(q), 1):
            print(f"  [{i}] {hit['listing_id']} via '{hit['chunk_type']}' "
                  f"— dist: {hit['distance']} — {hit['city']} — {hit['price']:,.0f} EUR")
