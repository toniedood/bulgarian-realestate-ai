import re
import yaml
import torch
import chromadb
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# ─── LOAD CONFIG ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

LISTINGS_FOLDER = config["rag"]["listings_folder"]
VECTOR_DB_PATH  = config["rag"]["vector_db_path"]
COLLECTION_NAME = config["rag"]["collection_name"]
CHUNK_MIN_LEN   = config["rag"]["chunk_min_length"]
EMBEDDING_MODEL = "rmihaylov/roberta-base-nli-stsb-bg"
MAX_TOKENS      = 512

# ─── LOAD EMBEDDING MODEL ────────────────────────────────────────────────────
print("⏳ Loading embedding model...")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model     = AutoModel.from_pretrained(EMBEDDING_MODEL)
model.eval()
print("✅ Model loaded.")


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def clean_markdown(text: str) -> str:
    """Strip markdown symbols so the model embeds clean text."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',     r'\1', text)
    text = re.sub(r'^#{1,6}\s+',   '',    text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}',       '\n\n', text)
    return text.strip()


def parse_metadata(content: str) -> dict:
    """
    Extract structured fields from a listing.
    Stored on ALL chunks so every chunk is filterable.
    """
    meta = {}

    # Title: ## Апартамент за продажба — Аспарухово, Варна
    title_match = re.search(r'^##\s+(.+)$', content, re.MULTILINE)
    if title_match:
        title = title_match.group(1)
        type_match = re.match(r'^(\S+)', title)
        if type_match:
            meta["property_type"] = type_match.group(1).lower()
        location_match = re.search(r'—\s+(.+),\s+(.+)$', title)
        if location_match:
            meta["neighborhood"] = location_match.group(1).strip()
            meta["city"]         = location_match.group(2).strip()

    price_match = re.search(r'Цена:\*\*\s*([\d\s,\.]+)\s*EUR', content)
    if price_match:
        try:
            meta["price"] = float(re.sub(r'[\s,]', '', price_match.group(1)))
        except ValueError:
            pass

    area_match = re.search(r'Площ:\*\*\s*([\d,\.]+)', content)
    if area_match:
        try:
            meta["area_sqm"] = float(area_match.group(1).replace(",", "."))
        except ValueError:
            pass

    bed_match = re.search(r'Спални:\*\*\s*(\d+)', content)
    if bed_match:
        meta["bedrooms"] = int(bed_match.group(1))

    year_match = re.search(r'Година на строеж:\*\*\s*(\d{4})', content)
    if year_match:
        meta["year_built"] = int(year_match.group(1))

    return meta


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


# ─── CONNECT TO CHROMADB ─────────────────────────────────────────────────────
print(f"⏳ Connecting to ChromaDB at {VECTOR_DB_PATH}...")
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

existing = [c.name for c in chroma_client.list_collections()]
if COLLECTION_NAME in existing:
    chroma_client.delete_collection(COLLECTION_NAME)
    print(f"🗑️  Deleted old collection '{COLLECTION_NAME}'")

collection = chroma_client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"✅ Collection '{COLLECTION_NAME}' created.\n")

# ─── INGEST ───────────────────────────────────────────────────────────────────
listing_files = sorted(Path(LISTINGS_FOLDER).glob("*.md"))
print(f"📂 Found {len(listing_files)} listings — 4 chunks each...\n")

total_chunks = 0

for filepath in listing_files:
    listing_id = filepath.stem
    content    = filepath.read_text(encoding="utf-8")

    meta = parse_metadata(content)
    meta["listing_id"] = listing_id
    meta["file"]       = str(filepath)

    # Split at --- divider
    parts     = content.split("---", maxsplit=1)
    narrative = parts[1].strip() if len(parts) > 1 else ""

    # ── Chunk 1: PRICE ───────────────────────────────────────────────────────
    # Focused on facts — matches queries about price, size, bedrooms
    price_parts = []
    if "property_type" in meta:
        price_parts.append(meta["property_type"].capitalize())
    if "price" in meta:
        price_parts.append(f"Цена {meta['price']:,.0f} EUR")
    if "area_sqm" in meta:
        price_parts.append(f"Площ {meta['area_sqm']:.0f} кв.м.")
    if "bedrooms" in meta:
        price_parts.append(f"{meta['bedrooms']} спални")
    if "year_built" in meta:
        price_parts.append(f"Построен {meta['year_built']}")
    price_chunk = ". ".join(price_parts) + "." if price_parts else ""

    # ── Chunk 2: CITY ────────────────────────────────────────────────────────
    # Focused on city-level location — matches "имот в Варна", "Sofia apartment"
    city  = meta.get("city", "")
    ptype = meta.get("property_type", "имот").capitalize()
    beds  = meta.get("bedrooms", "")
    price = meta.get("price", 0)
    city_chunk = (
        f"{ptype} за продажба в град {city}. "
        f"{str(beds) + ' спални, ' if beds else ''}"
        f"цена {price:,.0f} EUR."
    ) if city else ""

    # ── Chunk 3: NEIGHBORHOOD ────────────────────────────────────────────────
    # Focused on neighborhood — matches "тих квартал", "Аспарухово", "близо до центъра"
    hood = meta.get("neighborhood", "")
    hood_chunk = (
        f"Квартал {hood}, {city}. "
        f"{ptype} с {str(beds) + ' спални и ' if beds else ''}"
        f"цена {price:,.0f} EUR."
    ) if hood and city else ""

    # ── Chunk 4: NARRATIVE ───────────────────────────────────────────────────
    # Full description — matches semantic queries about features, atmosphere, views
    narrative_clean = clean_markdown(narrative) if narrative else ""

    # Build the final list of chunks to store
    chunks = [
        ("price",        price_chunk),
        ("city",         city_chunk),
        ("neighborhood", hood_chunk),
        ("narrative",    narrative_clean),
    ]

    for chunk_type, chunk_text in chunks:
        if len(chunk_text) < CHUNK_MIN_LEN:
            continue

        # If narrative is too long, truncate to 512 tokens
        if chunk_type == "narrative":
            tokens = tokenizer.encode(chunk_text)
            if len(tokens) > MAX_TOKENS:
                chunk_text = tokenizer.decode(
                    tokens[:MAX_TOKENS],
                    skip_special_tokens=True
                )

        collection.add(
            ids        = [f"{listing_id}_{chunk_type}"],
            embeddings = [embed(chunk_text)],
            documents  = [chunk_text],
            metadatas  = [{**meta, "chunk_type": chunk_type}]
        )
        total_chunks += 1

    print(f"  ✅ {listing_id} | {city:15} | {meta.get('price',0):>12,.0f} EUR | "
          f"{meta.get('bedrooms','?')} спални | {meta.get('neighborhood','')}")

print(f"\n🎉 Done! {total_chunks} chunks stored in ChromaDB.")
