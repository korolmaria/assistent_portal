"""Диагностика: chunker на API_инструкция.docx."""
import sys
import time
from pathlib import Path

print("1. Импорт...", flush=True)
from app.ingest.docx_parser import parse_docx
from app.ingest.chunker import chunk_document
print("   ✅ Импорты OK", flush=True)

docx = Path('data/uploads/API_инструкция.docx')
print(f"2. Парсинг {docx.name}...", flush=True)
t0 = time.time()
parsed = parse_docx(docx)
print(f"   ✅ Парсинг: {time.time()-t0:.2f}s", flush=True)
print(f"   Элементы: {parsed['metadata']['elements_count']}", flush=True)

# Смотрим первые 20 элементов
print(f"\n3. Первые 20 элементов:", flush=True)
for el in parsed['pages'][0]['elements'][:20]:
    etype = el['type']
    text = (el.get('text') or ' '.join(el.get('items', [])))[:70]
    print(f"   [{etype:8}] {text}", flush=True)

print(f"\n4. Чанкинг...", flush=True)
t0 = time.time()
try:
    chunks = chunk_document(parsed)
    print(f"   ✅ Чанкинг: {time.time()-t0:.2f}s", flush=True)
    print(f"   Чанков: {len(chunks)}", flush=True)
except Exception as e:
    print(f"   ❌ Ошибка: {e}", flush=True)
    import traceback
    traceback.print_exc()
