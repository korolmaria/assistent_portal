"""
Общие справочники и хелперы для всех инструментов Tool-агента.
"""
import calendar
from datetime import datetime


BRAND_IDS: dict[str, int] = {
    "хит": 2, "хит!": 2, "hit": 2,
    "грошык": 3,
    "евроопт": 6,
}

CLAIM_STATUS_LABELS = {
    "40": ("Ожидает ответа", "warning"),
    "50": ("Подтверждена", "success"),
    "51": ("Отклонена", "error"),
}

ORDER_STATUS_IDS: dict[str, int] = {
    "на сайте": 10,
    "автоподтверждён": 48,
    "подтверждён": 50,
    "выполнен": 90,
}

STATUS_LABELS: dict[int, str] = {
    10: "На сайте",
    48: "Автоподтверждён",
    50: "Подтверждён",
    90: "Выполнен",
}

PROCESSED_IDS: dict[str, int] = {
    "не обработан": 0, "необработанный": 0, "новый": 0,
    "обработан": 1, "обработанный": 1,
}

REGION_IDS: dict[str, int] = {
    "брестская область": 70, "брест": 71,
    "витебская область": 72, "витебск": 73,
    "гомельская область": 74, "гомель": 75,
    "гродненская область": 76, "гродно": 77,
    "минск": 78, "минская область": 79,
    "могилевская область": 80, "могилев": 81, "могилёв": 81,
}


# ==========================================================================
# Нормализация запроса → параметры
# ==========================================================================
MONTHS_RU: dict[str, int] = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5,
    "июн": 6, "июл": 7, "август": 8, "сентябр": 9, "октябр": 10,
    "ноябр": 11, "декабр": 12,
}

# Слова, после которых месяц относится к ДАТЕ ПОСТАВКИ
DELIVERY_DATE_MARKERS = ("поставк", "привоз", "привез", "на ", "за ")
# Слова, после которых месяц относится к ДАТЕ СОЗДАНИЯ
CREATED_DATE_MARKERS = ("созда", "оформ", "сформир", "я заказ", "сделал заказ")


def normalize_month_dates(params: dict, query: str) -> dict:
    """
    Если в query есть «за <месяц>» и дат ещё нет — подставляем.
    Определяем, к какой дате отнести месяц, по маркерам в query.
    """
    q = query.lower()

    # Уже есть даты — не трогаем
    if params.get("date_begin") or params.get("tt_time_begin"):
        return params

    month_num: int | None = None
    for stem, m in MONTHS_RU.items():
        if stem in q:
            # «ма» слишком короткое — требуем слово «май»/«мая»
            if stem == "ма" and not any(x in q for x in ("май", "мая", "мае")):
                continue
            month_num = m
            break

    if month_num is None:
        return params

    year = datetime.now().year
    # Если месяц уже прошёл (текущий месяц — октябрь, а просят январь) —
    # это скорее текущий год; если просят декабрь, а сейчас январь — прошлый год.
    now_month = datetime.now().month
    if month_num > now_month + 1:
        year -= 1

    last_day = calendar.monthrange(year, month_num)[1]
    begin = f"{year}-{month_num:02d}-01"
    end = f"{year}-{month_num:02d}-{last_day:02d}"

    # Куда отнести — к поставке или к созданию
    if any(m in q for m in CREATED_DATE_MARKERS) and not any(
        m in q for m in DELIVERY_DATE_MARKERS if m != "на "
    ):
        params["tt_time_begin"] = begin
        params["tt_time_end"] = end
    else:
        params["date_begin"] = begin
        params["date_end"] = end

    return params


# ==========================================================================
# Предохранители
# ==========================================================================
ORDER_ID_CONFLICTS: set[str] = {
    "checkx", "date_begin", "date_end", "tt_time_begin", "tt_time_end",
    "brand_ids", "is_processed", "region_ids", "warehouse_ids",
    "goods_ids", "goods_name",
}

CLAIM_ID_CONFLICTS: set[str] = {
    "date_begin", "date_end", "tt_time_begin", "tt_time_end",
    "brand_ids", "region_ids", "warehouse_ids",
}


def sanitize_params(tool_name: str, params: dict, query: str) -> dict:
    """
    Убирает конфликтующие параметры, если модель их лишне подставила.
    Работает независимо от того, послушала ли модель промпт.
    """
    q = query.lower()

    # 1. Заказы: если есть order_ids — срезаем все фильтры
    if tool_name == "get_supplier_orders" and params.get("order_ids"):
        for f in ORDER_ID_CONFLICTS:
            params.pop(f, None)

    # 2. Претензии: если есть claim_ids — срезаем фильтры
    if tool_name in ("get_claim_details",) and params.get("claim_ids"):
        for f in CLAIM_ID_CONFLICTS:
            params.pop(f, None)

    # 3. is_processed — только если явно сказано
    if "is_processed" in params and not any(
        w in q for w in ("обработан", "необработан", "не обработан")
    ):
        params.pop("is_processed", None)

    # 4. Если есть order_ids — не должно быть goods_* (это другой тул)
    if params.get("order_ids") and tool_name == "get_order_items":
        params.pop("goods_ids", None)
        params.pop("goods_name", None)

    return params


# ==========================================================================
# Статусы
# ==========================================================================
def compute_order_status(
    checkx: int | None,
    checkx_fact: int | None = None,
    is_revised: int | None = None,
    is_auto: int | None = None,
) -> tuple[str, str]:
    """(status_text, status_type) для заказов и позиций."""
    status = str(checkx) if checkx is not None else None
    is_auto_str = str(is_auto) if is_auto is not None else None
    is_revised_str = str(is_revised) if is_revised is not None else None

    if status == "10" and is_revised_str == "2":
        return "Корректировка разрешена", "success_default"
    if (not is_revised_str or is_revised_str == "99") and status == "10":
        return "На сайте", "success_default"
    if (not is_revised_str or is_revised_str == "99") and status == "50":
        text = "Автоподтверждён" if is_auto_str in ("1",) else "Подтверждён"
        return text, "success"
    if status in ("50", "10") and is_revised_str == "1":
        return "Запрошена корректировка", "warning"
    if status in ("50", "10") and is_revised_str == "3":
        return "Корректировка отклонена", "error"
    if status == "90":
        return "Выполнен", "default"

    fallback = STATUS_LABELS.get(checkx or 0, f"Статус {checkx}")
    return fallback, "warning"


def status_label(code: int) -> str:
    return STATUS_LABELS.get(code, f"Статус {code}")


def _to_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _to_int(val) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None
