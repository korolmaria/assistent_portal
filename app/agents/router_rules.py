"""
Детерминированный пре-фильтр маршрута.
Срабатывает ДО LLM-роутера. Возвращает route или None (пусть решает LLM).
"""
import re
from loguru import logger

# Пытаемся вытащить справочники из SPEC заказов
try:
    from app.tools.tool_registry.orders import SPEC as ORDERS_SPEC
except ImportError:
    ORDERS_SPEC = {}
    logger.warning("[router_rules] не удалось импортировать SPEC заказов — использую fallback")


def _extract_names_from_spec(param_name: str) -> list[str]:
    """Вытаскивает названия из строк вида '2=Хит, 3=Грошык' в описании параметра."""
    try:
        desc = ORDERS_SPEC["function"]["parameters"]["properties"][param_name]["description"]
        matches = re.findall(r"\d+=([А-Яа-яA-Za-z\s\-]+?)(?:,|$)", desc)
        names: list[str] = []
        for m in matches:
            clean = m.strip()
            # Убираем "ская обл", "ский" и т.д., оставляем корень
            clean = re.sub(r"(ская|ский)\s+обл", "", clean).strip()
            clean = re.sub(r"(ская|ский)", "", clean).strip()
            if clean:
                names.append(clean.lower())
        return names
    except Exception:
        return []


BRANDS = _extract_names_from_spec("brand_ids") if ORDERS_SPEC else []
REGIONS = _extract_names_from_spec("region_ids") if ORDERS_SPEC else []

# Fallback, если SPEC не подгрузился
if not BRANDS:
    BRANDS = ["евроопт", "хит", "грошык", "санта", "алми"]
if not REGIONS:
    REGIONS = ["минск", "брест", "гомель", "витебск", "гродно", "могилёв"]

# ============================================================
# РЕГУЛЯРКИ
# ============================================================

# Номер заказа / претензии: 6+ цифр подряд
ID_RE = re.compile(r"\b\d{6,}\b")

# Период
_MONTHS = (
    r"(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|"
    r"сентябр|октябр|ноябр|декабр)\w*"
)
DATE_RE = re.compile(
    rf"(?:за\s+)?(?:{_MONTHS}|"
    rf"\d{{4}}\s*год|"
    rf"последн\w+\s+(?:недел|месяц|дн)\w*|"
    rf"сегодня|вчера|позавчера)",
    re.IGNORECASE,
)

# Бренды
BRAND_RE = re.compile(rf"\b(?:{'|'.join(BRANDS)})\b", re.IGNORECASE) if BRANDS else None

# Склад
WAREHOUSE_RE = re.compile(r"\bсклад\w*\s*№?\s*\d+", re.IGNORECASE)

# Регионы
REGION_RE = re.compile(rf"\b(?:по|в)\s+(?:{'|'.join(REGIONS)})\w*\b", re.IGNORECASE) if REGIONS else None

# Статусы заказов (из SPEC)
ORDER_STATUS_RE = re.compile(
    r"\b(?:на\s+сайте|автоподтверждён\w*|подтверждён\w*|выполнен\w*|отменён\w*)\b",
    re.IGNORECASE,
)

# Претензии
CLAIM_RE = re.compile(
    r"\b(?:претензи\w*|брак\w*|возврат\w*|рекламаци\w*|недопоставк\w*)\b",
    re.IGNORECASE,
)

# Суперлативы (запрос конкретного экземпляра: первая, последняя, самая дорогая)
SUPERLATIVE_RE = re.compile(
    r"\b(?:перв\w+|последн\w+|крайн\w+|сам\w+\s+(?:дорог\w+|дешев\w+|больш\w+|маленьк\w+))\b",
    re.IGNORECASE,
)

# ============================================================
# КЛЮЧЕВЫЕ СЛОВА
# ============================================================

SHOW_KW = (
    "покажи", "выведи", "список", "найди", "сколько",
    "какие", "есть ли", "был ли", "дай", "покажите", "найдите",
    "расскажи", "расскажите", "что там",
)
HOW_KW = (
    "как ", "можно ли", "инструкция", "правило", "метод",
    "описание", "работает", "нужно ли", "что делать",
    "каким образом", "почему", "разрешается ли",
)

# ============================================================
# ЛОГИКА
# ============================================================

def check_filters(query: str) -> tuple[bool, dict[str, bool]]:
    """Публичная функция — используется и в роутере, и в логах датасета."""
    flags = {
        "id":          bool(ID_RE.search(query)),
        "date":        bool(DATE_RE.search(query)),
        "brand":       bool(BRAND_RE.search(query)) if BRAND_RE else False,
        "warehouse":   bool(WAREHOUSE_RE.search(query)),
        "region":      bool(REGION_RE.search(query)) if REGION_RE else False,
        "status":      bool(ORDER_STATUS_RE.search(query)),
        "claim":       bool(CLAIM_RE.search(query)),
        "superlative": bool(SUPERLATIVE_RE.search(query)),
    }
    return any(flags.values()), flags


def hard_route(query: str) -> tuple[str | None, str]:
    """
    Возвращает (route | None, reason).
    route is None → пусть решает LLM.
    """
    q = query.lower()
    _, flags = check_filters(query)
    has_show = any(k in q for k in SHOW_KW)
    has_how  = any(k in q for k in HOW_KW)

    # 1. Суперлативы (первая, последняя, самая дорогая) → всегда tool
    if flags["superlative"]:
        return "tool", f"rule: суперлатив (первая/последняя) {flags}"

    # 2. Претензии — отдельная логика
    if flags["claim"]:
        if has_show and not has_how:
            return "tool", f"rule: показать претензии {flags}"
        if has_how and not has_show:
            return "rag", "rule: как работать с претензиями"
        if has_how and has_show:
            if flags["date"] or flags["id"] or flags["warehouse"]:
                return "rag+tool", f"rule: претензии + фильтр + how {flags}"
            return "tool", "rule: претензии (show приоритетнее how)"
        return "tool", f"rule: претензии без маркеров {flags}"

    # 3. Фильтр + «как/можно ли» → rag+tool
    if flags["id"] or flags["date"] or flags["brand"] or flags["warehouse"] \
            or flags["region"] or flags["status"]:
        if has_how:
            return "rag+tool", f"rule: фильтр+how {flags}"
        if has_show:
            return "tool", f"rule: фильтр+show {flags}"
        if flags["id"]:
            return "tool", "rule: только номер"

    # 4. «как/правило» без фильтра → rag
    if has_how and not any(flags.values()):
        return "rag", "rule: how без фильтра"

    # 5. Непонятно → LLM
    return None, "rule: неоднозначно → LLM"