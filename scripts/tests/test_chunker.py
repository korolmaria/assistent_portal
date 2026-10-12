"""Тест чанкера на реальном DOCX."""
import asyncio
from pathlib import Path

from app.ingest.router import parse_document
from app.ingest.chunker import chunk_document


async def main():
    # DOCX Требования
    docx = list(Path('data/uploads').rglob('Требования*.docx'))[0]
    parsed = await parse_document(docx)
    chunks = chunk_document(parsed)

    print(f"\n{'=' * 70}")
    print(f"=== {parsed['source_file']} ===")
    print(f"Всего чанков: {len(chunks)}")
    print('=' * 70)

    for i, ch in enumerate(chunks, 1):
        payload = ch['payload']
        text = ch['text'][:180].replace('\n', ' ⏎ ')
        print(f"\n[{i}] id={ch['id'][:12]}...")
        print(f"    type={payload['element_type']:8} page={payload['page']} "
              f"char=[{payload['chunk_char_start']}:{payload['chunk_char_end']}]")
        print(f"    section={payload['section']!r}")
        print(f"    section_slug={payload['section_slug']!r}")
        if payload.get('is_table_row'):
            print(f"    TABLE row #{payload.get('table_row_index')} "
                  f"caption={payload.get('caption', '')[:40]!r}")
        print(f"    prev={payload.get('prev_chunk_id', '')[:12]}... "
              f"next={payload.get('next_chunk_id', '')[:12]}...")
        print(f"    TEXT: {text}")


if __name__ == "__main__":
    asyncio.run(main())
