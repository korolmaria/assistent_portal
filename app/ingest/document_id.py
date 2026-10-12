"""
Генерация читаемых ID из русских имён файлов.

'Требования к остаточному сроку годности товаров при приёмке.docx'
→ 'trebovaniya_k_ostatochnomu_sroku_godnosti_tovarov_pri_priemke_a3f5'
"""
import hashlib
import re
import unicodedata
from pathlib import Path


# Таблица транслитерации (ГОСТ 7.79-2000, система Б)
TRANSLIT = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    'А': 'a', 'Б': 'b', 'В': 'v', 'Г': 'g', 'Д': 'd', 'Е': 'e', 'Ё': 'e',
    'Ж': 'zh', 'З': 'z', 'И': 'i', 'Й': 'y', 'К': 'k', 'Л': 'l', 'М': 'm',
    'Н': 'n', 'О': 'o', 'П': 'p', 'Р': 'r', 'С': 's', 'Т': 't', 'У': 'u',
    'Ф': 'f', 'Х': 'kh', 'Ц': 'ts', 'Ч': 'ch', 'Ш': 'sh', 'Щ': 'shch',
    'Ъ': '', 'Ы': 'y', 'Ь': '', 'Э': 'e', 'Ю': 'yu', 'Я': 'ya',
}


def make_document_id(file_path: Path, max_len: int = 60) -> str:
    """
    Из пути файла делает читаемый ID.
    Уникальность гарантируется коротким хешем.
    """
    name = file_path.stem
    # Нормализация Unicode (NFC — убираем разницу macOS/Linux)
    name = unicodedata.normalize('NFC', name)

    # Транслитерация
    result = []
    for ch in name:
        if ch in TRANSLIT:
            result.append(TRANSLIT[ch])
        elif ch.isalnum() or ch in '_-':
            result.append(ch)
        else:
            result.append('_')

    slug = ''.join(result)
    slug = re.sub(r'_+', '_', slug).strip('_').lower()

    if len(slug) > max_len:
        slug = slug[:max_len].rstrip('_')

    # Короткий хеш от полного пути — 4 символа
    short_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:4]

    return f"{slug}_{short_hash}"


def make_section_slug(text: str, max_len: int = 50) -> str:
    """Из заголовка раздела делает короткий slug (для метаданных)."""
    text = unicodedata.normalize('NFC', text.strip())
    result = []
    for ch in text:
        if ch in TRANSLIT:
            result.append(TRANSLIT[ch])
        elif ch.isalnum() or ch in ' _-':
            result.append(ch)
        else:
            result.append('_')
    slug = ''.join(result)
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug[:max_len]
