"""
Pydantic-схемы для всех агентов, Supervisor и SSE-стриминга.

Схемы совпадают с форматом фронта (ai.service.ts).
"""
from typing import Any, Literal

from pydantic import BaseModel, Field


# ==========================================================================
# RAG-источники
# ==========================================================================
class Source(BaseModel):
    """Источник для RAG-ответа."""
    doc_id: str
    doc_title: str | None = None
    page: int | None = None
    page_label: str | None = None
    section: str | None = None
    content_type: str = "text"           # text | table | header | image
    snippet: str = ""
    grade_score: float | None = None     # скор реранкера


# ==========================================================================
# Верификация
# ==========================================================================
class Verification(BaseModel):
    faithful: bool = True
    issues: list[str] = Field(default_factory=list)


# ==========================================================================
# Display-блоки (структурированные данные для фронта)
# ==========================================================================
class OrderCardData(BaseModel):
    """Карточка заказа (из SupplierPortal.GetOrderBuyGroup)."""
    order_id: int | str
    contract_group_id: int | str | None = None
    contractor: str | None = None
    contractor_id: int | str | None = None
    manager: str | None = None
    phone: str | None = None
    status: str = ""
    status_code: int | None = None
    checkx: int | None = None
    checkx_code: int | str | None = None
    type: str | None = None
    revised: int | str | None = None
    is_auto: int | str | None = None
    created_at: str | None = None
    datex: str | None = None
    next_order_date: str | None = None
    shipment_date: str | None = None
    sum: float | None = None
    warehouse: int | str | None = None
    warehouse_name: str | None = None
    gln_warehouse: int | str | None = None
    cross_dock: int | str | None = None
    brand: str | None = None
    brand_id: int | str | None = None
    address: str | None = None
    region: str | None = None
    district: str | None = None
    city: str | None = None


class OrdersTableExtra(BaseModel):
    """Extra для таблицы заказов."""
    count: int
    total_sum: float | None = None


class InfoField(BaseModel):
    """Одно поле для info-блока."""
    field: str
    label: str
    value: str | int
    badge_type: str | None = None


class InfoData(BaseModel):
    """Info-блок (одна карточка с полями)."""
    order_id: int | str
    fields: list[InfoField]


class DisplayBlock(BaseModel):
    """
    Структурированный блок для рендера на фронте.
    Тип определяет, как интерпретировать data:
      - card  → OrderCardData
      - info  → InfoData
      - table → list[OrderCardData]
      - text  → None (только extra)
    """
    type: Literal["card", "info", "table", "text", "order_items_table"]
    data: Any = None
    extra: dict[str, Any] | None = None


# ==========================================================================
# Метаданные ответа
# ==========================================================================
class MessageMetadata(BaseModel):
    """Метаданные ответа ассистента."""
    sources: list[Source] = Field(default_factory=list)
    verification: Verification | None = None
    steps: list[str] = Field(default_factory=list)
    processingTime: float | None = None
    answer: str | None = None
    route: Literal["auto", "rag", "tool"] | None = None
    route_reason: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    display: DisplayBlock | None = None


# ==========================================================================
# Запрос от фронта
# ==========================================================================
class ChatRequest(BaseModel):
    """Тело POST /api/v1/ask и /api/v1/ask/stream."""
    message: str = Field(..., min_length=1, max_length=4000)
    route: Literal["auto", "rag", "tool"] = "auto"


# ==========================================================================
# SSE-события (то, что фронт парсит из data: {...})
# ==========================================================================
class SSEToken(BaseModel):
    """Чанк текста ответа."""
    type: Literal["token"] = "token"
    content: str


class SSERoute(BaseModel):
    """Выбранный маршрут."""
    type: Literal["route"] = "route"
    route: str
    reason: str = ""


class SSEStep(BaseModel):
    """Промежуточный шаг агента (для дебага/индикации)."""
    type: Literal["step"] = "step"
    message: str


class SSEMetadata(BaseModel):
    """Финальные метаданные."""
    type: Literal["metadata"] = "metadata"
    sources: list[Source] = Field(default_factory=list)
    verification: Verification | None = None
    steps: list[str] = Field(default_factory=list)
    processingTime: float | None = None
    answer: str | None = None
    route: str | None = None
    route_reason: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    display: DisplayBlock | None = None


class SSEError(BaseModel):
    """Ошибка стрима."""
    type: Literal["error"] = "error"
    error: str


# ==========================================================================
# Health check
# ==========================================================================
class HealthStatus(BaseModel):
    """Статус всех сервисов."""
    status: str = "ok"
    version: str = "0.1.0"
    supervisor: bool = True
    vllm_vl: bool = False
    vllm_text: bool = False
    triton: bool = False
    qdrant: bool = False
    neo4j: bool = False
