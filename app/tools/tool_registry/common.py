"""
Общие справочники и хелперы для всех инструментов Tool-агента.
"""


BRAND_IDS: dict[str, int] = {
    "хит": 2, "хит!": 2, "hit": 2,
    "грошык": 3,
    "евроопт": 6,
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