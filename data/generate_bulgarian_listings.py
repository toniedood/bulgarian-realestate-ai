import vertexai
from vertexai.generative_models import GenerativeModel
import yaml
import os
import time

# ─── LOAD CONFIG ─────────────────────────────────────────────────────────────
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

# ─── INITIALIZE VERTEX AI ────────────────────────────────────────────────────
vertexai.init(
    project=config["google_cloud"]["project_id"],
    location=config["google_cloud"]["location"]
)

model = GenerativeModel(config["google_cloud"]["model"])

# ─── PROPERTY TEMPLATES ──────────────────────────────────────────────────────
cities = [
    "София", "Пловдив", "Варна", "Бургас", "Русе",
    "Стара Загора", "Плевен", "Велико Търново"
]

neighbourhoods = {
    "София": ["Лозенец", "Витоша", "Младост", "Иван Вазов", "Борово", "Овча купел"],
    "Пловдив": ["Кършияка", "Тракия", "Центъра", "Христо Смирненски", "Захарна фабрика"],
    "Варна": ["Чайка", "Левски", "Владиславово", "Бриз", "Аспарухово"],
    "Бургас": ["Лазур", "Славейков", "Меден рудник", "Центъра", "Сарафово"],
    "Русе": ["Центъра", "Дружба", "Родина", "Чародейка", "Възраждане"],
    "Стара Загора": ["Центъра", "Зора", "Три чучура", "Самара", "Железник"],
    "Плевен": ["Центъра", "Сторгозия", "Дружба", "Кайлъка", "Мл. Кооператор"],
    "Велико Търново": ["Центъра", "Асенова махала", "Картала", "Колю Фичето", "Бойчеви колиби"],
}

property_types = ["Апартамент", "Къща", "Вила", "Студио", "Мезонет", "Пентхаус"]

# ─── GENERATE ONE LISTING ────────────────────────────────────────────────────
def generate_listing(property_id):
    import random
    random.seed(property_id)

    city = random.choice(cities)
    neighbourhood = random.choice(neighbourhoods[city])
    prop_type = random.choice(property_types)
    bedrooms = random.randint(1, 5)
    bathrooms = random.randint(1, min(bedrooms, 3))
    size = random.randint(40, 450)
    price = round(size * random.randint(800, 4000), -3)
    year_built = random.randint(1960, 2024)
    floor = random.randint(1, 12) if prop_type in ["Апартамент", "Студио", "Пентхаус"] else None

    floor_info = f"Етаж {floor}" if floor else "Самостоятелна сграда"

    prompt = f"""Ти си опитен брокер на недвижими имоти в България. 
Напиши реалистична обява за имот на БЪЛГАРСКИ ЕЗИК.

Детайли за имота:
- Тип: {prop_type}
- Град: {city}
- Квартал: {neighbourhood}
- Цена: {price:,} EUR
- Площ: {size} кв.м.
- Спални: {bedrooms}
- Бани: {bathrooms}
- Година на строеж: {year_built}
- {floor_info}

Обявата трябва да съдържа точно тези секции:

## Описание
(2-3 изречения за имота и квартала)

## Характеристики
(обзавеждане, състояние, особености на имота)

## Локация и транспорт
(близост до транспорт, училища, магазини)

## Правна информация
(вид собственост, документи, данъчна оценка)

Пиши естествено и убедително на български. Не добавяй излишни символи."""

    response = model.generate_content(prompt)
    
    header = f"""# Обява #{property_id:03d}

## {prop_type} за продажба — {neighbourhood}, {city}

**Цена:** {price:,} EUR
**Площ:** {size} кв.м.
**Спални:** {bedrooms}
**Бани:** {bathrooms}
**Година на строеж:** {year_built}
**{floor_info}**

---

"""
    return header + response.text

# ─── GENERATE ALL LISTINGS ───────────────────────────────────────────────────
os.makedirs("data/listings", exist_ok=True)

total = 60
for i in range(1, total + 1):
    print(f"⏳ Generating property {i}/{total}...")
    
    try:
        content = generate_listing(i)
        filename = f"data/listings/imot_{i:03d}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"✅ Saved {filename}")
        
        # Small pause to avoid hitting API rate limits
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error on property {i}: {e}")
        time.sleep(5)

print(f"\n🎉 Done! Generated {total} Bulgarian property listings in data/listings/")