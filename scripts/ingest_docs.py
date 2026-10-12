"""
CLI для массового ingest документов в Qdrant.

Usage:
    python -m scripts.ingest_docs --dir data/uploads/
    python -m scripts.ingest_docs --file path.pdf
    python -m scripts.ingest_docs --dir data/uploads/ --recreate
    python -m scripts.ingest_docs --status
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from app.ingest.router import parse_document, get_supported_extensions
from app.ingest.chunker import chunk_document
from app.ingest.indexer import index_chunks, delete_document, get_document_stats
from app.ingest.document_id import make_document_id
from app.tools.qdrant_client import qdrant_db


# ==========================================================================
# Журнал ingest
# ==========================================================================
STATE_FILE = Path("data/.ingest_state.json")


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"documents": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


# ==========================================================================
# Поиск файлов
# ==========================================================================
def find_documents(directory: Path) -> list[Path]:
    """Находит все поддерживаемые файлы рекурсивно."""
    extensions = get_supported_extensions()
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


# ==========================================================================
# Ingest одного файла
# ==========================================================================
async def ingest_file(
    file_path: Path,
    state: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Парсит + чанкит + индексирует один файл. Возвращает статистику."""
    document_id = make_document_id(file_path)
    start = time.time()

    # Проверка в журнале
    existing = state["documents"].get(document_id)
    if existing and existing.get("status") == "done" and not force:
        logger.info(f"[ingest] пропускаю (уже загружен): {file_path.name}")
        return {"skipped": True, **existing}

    # Удаляем старые точки, если переиндексация
    if existing and force:
        logger.info(f"[ingest] удаляю старые точки: {document_id}")
        delete_document(document_id)

    try:
        # 1. Парсинг
        logger.info(f"[ingest] парсинг: {file_path.name}")
        parsed = await parse_document(file_path)

        # 2. Чанкинг
        chunks = chunk_document(parsed)
        logger.info(f"[ingest] {file_path.name}: {len(chunks)} чанков")

        # 3. Индексация
        total = index_chunks(chunks)

        # 4. Запись в журнал
        elapsed = round(time.time() - start, 2)
        entry = {
            "source_file": file_path.name,
            "source_path": str(file_path),
            "document_id": document_id,
            "status": "done",
            "chunks": total,
            "elapsed_sec": elapsed,
        }
        state["documents"][document_id] = entry
        save_state(state)

        logger.info(
            f"[ingest] ✅ {file_path.name}: {total} чанков за {elapsed} сек"
        )
        return entry

    except Exception as e:
        logger.exception(f"[ingest] ошибка на {file_path.name}")
        state["documents"][document_id] = {
            "source_file": file_path.name,
            "status": "error",
            "error": str(e),
        }
        save_state(state)
        return {"error": str(e), "source_file": file_path.name}


# ==========================================================================
# Основной сценарий
# ==========================================================================
async def run_ingest(
    files: list[Path],
    force: bool = False,
) -> None:
    state = load_state()
    qdrant_db.create_collection()
    qdrant_db.create_payload_indexes()

    total_files = len(files)
    print(f"\n{'=' * 70}")
    print(f"📥 Ingest {total_files} файлов")
    print(f"{'=' * 70}\n")

    start = time.time()
    success = 0
    skipped = 0
    errors = 0
    total_chunks = 0

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{total_files}] {file_path.relative_to('data/uploads')}")
        result = await ingest_file(file_path, state, force=force)

        if result.get("skipped"):
            skipped += 1
        elif result.get("error"):
            errors += 1
        else:
            success += 1
            total_chunks += result.get("chunks", 0)

    elapsed = round(time.time() - start, 1)

    print(f"\n{'=' * 70}")
    print(f"✅ Ingest завершён за {elapsed} сек")
    print(f"{'=' * 70}")
    print(f"  Успешно:  {success}")
    print(f"  Пропущено: {skipped}")
    print(f"  Ошибок:   {errors}")
    print(f"  Чанков:   {total_chunks}")
    print()

    # Информация о коллекции
    info = qdrant_db.info()
    print(f"  Qdrant: {info['points_count']} точек, status={info['status']}")


def show_status() -> None:
    """Показывает статус ingest."""
    state = load_state()
    docs = state.get("documents", {})

    print(f"\n{'=' * 70}")
    print(f"📊 Статус ingest")
    print(f"{'=' * 70}")
    print(f"Всего документов в журнале: {len(docs)}\n")

    done = [d for d in docs.values() if d.get("status") == "done"]
    errors = [d for d in docs.values() if d.get("status") == "error"]

    if done:
        print(f"✅ Загружено ({len(done)}):")
        for d in done:
            print(f"  {d['source_file']:60} {d.get('chunks', 0):4} чанков")
        print()
    if errors:
        print(f"❌ Ошибки ({len(errors)}):")
        for d in errors:
            print(f"  {d['source_file']:60} {d.get('error', '')[:50]}")
        print()

    info = qdrant_db.info()
    print(f"Qdrant: {info['points_count']} точек, status={info['status']}")


# ==========================================================================
# CLI
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="Ingest документов в Qdrant")
    parser.add_argument("--dir", type=Path, help="Папка с документами")
    parser.add_argument("--file", type=Path, help="Один файл")
    parser.add_argument("--recreate", action="store_true",
                        help="Удалить старые чанки и загрузить заново")
    parser.add_argument("--status", action="store_true",
                        help="Показать статус")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    files: list[Path] = []
    if args.file:
        if not args.file.exists():
            print(f"❌ Файл не найден: {args.file}")
            sys.exit(1)
        files = [args.file]
    elif args.dir:
        if not args.dir.exists():
            print(f"❌ Папка не найдена: {args.dir}")
            sys.exit(1)
        files = find_documents(args.dir)
    else:
        # По умолчанию — data/uploads
        files = find_documents(Path("data/uploads"))

    if not files:
        print("❌ Не найдено файлов для ingest")
        sys.exit(1)

    asyncio.run(run_ingest(files, force=args.recreate))


if __name__ == "__main__":
    main()
