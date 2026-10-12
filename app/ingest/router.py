"""
Роутер: выбирает парсер по расширению файла.

- .pdf  → pdf_parser.parse_pdf (pdfplumber + VL)
- .docx → docx_parser.parse_docx (python-docx)
- .doc  → конвертация через LibreOffice в .docx → docx_parser
"""
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from app.ingest.docx_parser import parse_docx
from app.ingest.pdf_parser import parse_pdf


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}


class UnsupportedFormatError(Exception):
    """Формат файла не поддерживается."""
    pass


def _convert_doc_to_docx(doc_path: Path) -> Path:
    """
    Конвертирует .doc → .docx через LibreOffice.
    Возвращает путь к временному .docx.
    """
    if not shutil.which("libreoffice"):
        raise RuntimeError(
            "LibreOffice не установлен. Установите: "
            "apt-get install -y libreoffice-writer"
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="doc_convert_"))
    logger.info(f"[router] конвертирую {doc_path.name} → .docx")

    result = subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to", "docx",
            "--outdir", str(tmp_dir),
            str(doc_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice failed: {result.stderr or result.stdout}"
        )

    # Ищем результат
    docx_files = list(tmp_dir.glob("*.docx"))
    if not docx_files:
        raise RuntimeError(f"LibreOffice не создал .docx в {tmp_dir}")

    converted = docx_files[0]
    logger.info(f"[router] конвертировано: {converted}")
    return converted


async def parse_document(file_path: str | Path) -> dict[str, Any]:
    """
    Универсальный парсер: определяет тип по расширению.
    Возвращает ParsedDocument.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Формат '{suffix}' не поддерживается. "
            f"Допустимые: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    logger.info(f"[router] парсинг {file_path.name} (формат: {suffix})")

    if suffix == ".pdf":
        return await parse_pdf(file_path)

    if suffix == ".docx":
        return parse_docx(file_path)

    if suffix == ".doc":
        # Конвертируем во временный .docx
        converted = _convert_doc_to_docx(file_path)
        try:
            result = parse_docx(converted)
            # Меняем source_file на исходный
            result["source_file"] = file_path.name
            result["source_type"] = "doc"
            result["parser"] = "libreoffice+python-docx"
            return result
        finally:
            # Чистим временные файлы
            converted.unlink(missing_ok=True)
            try:
                converted.parent.rmdir()
            except OSError:
                pass


def get_supported_extensions() -> set[str]:
    """Возвращает множество поддерживаемых расширений."""
    return SUPPORTED_EXTENSIONS.copy()
