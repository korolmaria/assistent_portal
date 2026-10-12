"""Тест полного ingest-пайплайна на 1 DOCX."""
import asyncio
from pathlib import Path

from app.ingest.router import parse_document
from app.ingest.chunker import chunk_document
from app.ingest.indexer import index_chunks, get_document_stats
from app.tools.qdrant_client import qdrant_db


async def main():
    # 1. Парсинг
    docx = list(Path('data/uploads').rglob('Требования*.docx'))[0]
    print(f"📄 Парсинг: {docx.name}")
    parsed = await parse_document(docx)

    # 2. Чанкинг
    print(f"✂️  Чанкинг...")
    chunks = chunk_document(parsed)
    print(f"   Получено чанков: {len(chunks)}")

    # 3. Индексация
    print(f"🔢 Индексация в Qdrant...")
    total = index_chunks(chunks)

    # 4. Проверка
    info = qdrant_db.info()
    print(f"\n✅ Итого в Qdrant: {info['points_count']} точек, status={info['status']}")

    # 5. Проверяем что нашли наш документ
    doc_id = parsed["document_id"]
    stats = get_document_stats(doc_id)
    print(f"✅ Документ {doc_id}: {stats['chunks_count']} чанков")


if __name__ == "__main__":
    asyncio.run(main())
