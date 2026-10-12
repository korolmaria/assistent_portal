"""Диагностика: какой элемент вызывает зависание."""
import signal
import sys
import time
from pathlib import Path

from app.ingest.docx_parser import parse_docx
from app.ingest.chunker import _split_text, _table_to_chunks, _make_chunk_id

docx = Path('data/uploads/API_инструкция.docx')
parsed = parse_docx(docx)
elements = parsed['pages'][0]['elements']

print(f"Всего элементов: {len(elements)}")
print(f"Типы: header={sum(1 for e in elements if e['type']=='header')}, "
      f"text={sum(1 for e in elements if e['type']=='text')}")
print()

# Проверяем _split_text отдельно на каждом большом тексте
print("Проверяем _split_text на текстах >500 символов...")
big_texts = [
    (i, el.get('text', ''))
    for i, el in enumerate(elements)
    if el['type'] == 'text' and len(el.get('text', '')) > 500
]
print(f"Найдено текстов >500 символов: {len(big_texts)}")
print()

for i, text in big_texts[:5]:  # первые 5
    print(f"[{i}] длина={len(text)}: {text[:80]!r}")
    t0 = time.time()
    
    # Таймаут через signal
    class Timeout(Exception): pass
    def handler(s, f): raise Timeout()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(5)
    
    try:
        chunks = _split_text(text)
        signal.alarm(0)
        print(f"   ✅ split: {len(chunks)} чанков за {time.time()-t0:.3f}s")
    except Timeout:
        print(f"   ❌ ЗАВИС на этом тексте!")
        print(f"   Полный текст (первые 1000): {text[:1000]!r}")
        break
    except Exception as e:
        signal.alarm(0)
        print(f"   ❌ Ошибка: {e}")
"""Диагностика: какой элемент вызывает зависание."""
import signal
import time
from pathlib import Path

from app.ingest.docx_parser import parse_docx
from app.ingest.chunker import _split_text


class Timeout(Exception):
    pass


def _handler(s, f):
    raise Timeout()


docx = Path('data/uploads/API_инструкция.docx')
parsed = parse_docx(docx)
elements = parsed['pages'][0]['elements']

print(f"Всего элементов: {len(elements)}")
print(f"Типы: header={sum(1 for e in elements if e['type']=='header')}, "
      f"text={sum(1 for e in elements if e['type']=='text')}")
print()

print("Проверяем _split_text на текстах >500 символов...")
big_texts = [
    (i, el.get('text', ''))
    for i, el in enumerate(elements)
    if el['type'] == 'text' and len(el.get('text', '')) > 500
]
print(f"Найдено текстов >500 символов: {len(big_texts)}")
print()

for i, text in big_texts[:10]:
    print(f"[{i}] длина={len(text)}: {text[:80]!r}")
    t0 = time.time()

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(5)

    try:
        chunks = _split_text(text)
        signal.alarm(0)
        print(f"   OK split: {len(chunks)} чанков за {time.time()-t0:.3f}s")
    except Timeout:
        print(f"   ЗАВИС! Первые 500 симв: {text[:500]!r}")
        break
    except Exception as e:
        signal.alarm(0)
        print(f"   Ошибка: {e}")
