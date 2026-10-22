from app.tools.qdrant_client import qdrant_db
from app.tools.triton_client import triton_client
from qdrant_client.models import Filter, FieldCondition, MatchValue

QUERY = "каким методом api я могу подтвердить заказы"
DOC_ID = "api_instruktsiya_ea8f"

print(f"Запрос: {QUERY}\n")

# 1. Проверяем, есть ли чанк в базе
scroll_res, _ = qdrant_db.client.scroll(
    collection_name=qdrant_db.collection,
    scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=DOC_ID))]),
    limit=100,
    with_payload=True
)
print(f"Всего чанков для API_инструкция: {len(scroll_res)}")
for p in scroll_res:
    text = p.payload.get("text", "")
    if "ConfirmOrderBuyGroup" in text:
        print(f"✅ Чанк найден в базе! ID: {p.id}")
        print(f"Текст: {text[:400]}...\n")
        break
else:
    print("❌ Чанк НЕ найден в базе! Возможно, переиндексация не применилась.")

# 2. Проверяем гибридный поиск
dense = triton_client.embed([QUERY])[0].tolist()
candidates = qdrant_db.hybrid_search(query=QUERY, dense_vector=dense, top_k=32, prefetch_limit=64)

print(f"Кандидатов из Qdrant: {len(candidates)}")
found = False
for i, c in enumerate(candidates, 1):
    text = c["payload"].get("text", "")
    if "ConfirmOrderBuyGroup" in text:
        print(f"✅ НАЙДЕН в кандидате №{i} (RRF: {c['score']:.4f})")
        found = True
        break
    if i <= 3:
        print(f"  Топ-{i} (RRF: {c['score']:.4f}): {text[:100]}...")
        
if not found:
    print("❌ Чанк НЕ НАЙДЕН в топ-32 гибридного поиска!")