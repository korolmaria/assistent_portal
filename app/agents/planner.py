"""
Планер: превращает запрос пользователя в упорядоченный список шагов.

Слои (включаются флагами ниже):
  1. hard_route (router_rules)  — детерминированные правила, 0 токенов
  2. prescan                     — сигнал на аналитика (tool → analyst)
  3. LLM-роутер (guided JSON)    — для неочевидных случаев
  4. fallback                    — если LLM упала / вернула мусор

Инварианты плана (валидируются в Pydantic):
  - len(steps) >= 1
  - analyst не первый
  - нет дублей агентов, кроме rag+tool
"""
import json
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from app.agents.router_rules import check_filters, hard_route, wants_analyst
from app.llm.vllm_client import vllm_client
from app.schemas import ChatHistoryItem


# ==========================================================================
# Флаги слоёв
# ==========================================================================
USE_HARD_ROUTE = True     # слой 1: regex-правила
USE_PRESCAN = True        # слой 2: детектор аналитика (включён)


# ==========================================================================
# Модели
# ==========================================================================
class PlanStep(BaseModel):
    agent: Literal["rag", "tool", "analyst"]
    purpose: str = ""


class Plan(BaseModel):
    steps: list[PlanStep] = Field(..., min_length=1)
    reason: str = ""
    source: Literal["rule", "prescan", "llm", "fallback"] = "llm"

    @field_validator("steps")
    @classmethod
    def _validate_invariants(cls, steps: list[PlanStep]) -> list[PlanStep]:
        if not steps:
            raise ValueError("plan must contain at least one step")
        if steps[0].agent == "analyst":
            raise ValueError("analyst cannot be the first step")
        agents = [s.agent for s in steps]
        if len(agents) != len(set(agents)):
            if not (len(agents) == 2 and set(agents) == {"rag", "tool"}):
                raise ValueError(f"duplicate agents in plan: {agents}")
        return steps


# ==========================================================================
# JSON-схема для guided JSON
# ==========================================================================
PLAN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "route_plan",
        "schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent": {"type": "string", "enum": ["rag", "tool", "analyst"]},
                            "purpose": {"type": "string"},
                        },
                        "required": ["agent", "purpose"],
                        "additionalProperties": False,
                    },
                },
                "reason": {"type": "string"},
            },
            "required": ["steps", "reason"],
            "additionalProperties": False,
        },
    },
}


# ==========================================================================
# Слой 2: prescan
# ==========================================================================
def _prescan(query: str) -> Plan | None:
    if not wants_analyst(query):
        return None

    # НОВОЕ: если есть маркеры документов, не перехватываем (пусть решает LLM)
    q_lower = query.lower()
    if any(kw in q_lower for kw in ("требовани", "инструкци", "правил", "регламент")):
        return None

    has_filters, flags = check_filters(query)

    # ИСПОЛЬЗУЕМ q_lower ВМЕСТО q
    if any(k in q_lower for k in ("сравн", "тренд", "динамик", "во сколько раз", "насколько больше", "насколько меньше")):
        return Plan(
            steps=[
                PlanStep(agent="tool", purpose="получить данные для сравнения"),
                PlanStep(agent="analyst", purpose="сравнить / посчитать динамику"),
            ],
            reason=f"prescan: сравнение + фильтры {flags}",
            source="prescan",
        )
    if any(k in q_lower for k in ("почему", "из-за чего")):
        return Plan(
            steps=[
                PlanStep(agent="tool", purpose="получить данные для анализа причин"),
                PlanStep(agent="analyst", purpose="объяснить причины изменения"),
            ],
            reason=f"prescan: причинный вопрос + фильтры {flags}",
            source="prescan",
        )
    return None


# ==========================================================================
# Слой 1: адаптер legacy route → Plan
# ==========================================================================
_LEGACY_MAP: dict[str, list[PlanStep]] = {
    "rag": [PlanStep(agent="rag", purpose="поиск по документации")],
    "tool": [PlanStep(agent="tool", purpose="запрос данных из API")],
    "rag+tool": [
        PlanStep(agent="rag", purpose="найти правила"),
        PlanStep(agent="tool", purpose="получить данные"),
    ],
}


def _plan_from_legacy_route(route: str, reason: str, source: str = "rule") -> Plan:
    return Plan(steps=_LEGACY_MAP[route], reason=reason, source=source)  # type: ignore[arg-type]


# ==========================================================================
# Слой 3: LLM-роутер
# ==========================================================================
PLANNER_SYSTEM_PROMPT = """Ты — планировщик мультиагентной системы.

Верни УПОРЯДОЧЕННЫЙ список шагов (1-3) для ответа на запрос пользователя.

Доступные агенты:
- rag      — ищет в документации (инструкции, регламенты, методы API, правила)
- tool     — запрашивает данные из API (заказы, претензии, статусы, суммы)
- analyst  — анализирует УЖЕ ПОЛУЧЕННЫЕ данные (сравнение, тренды, причины, агрегаты)

ПРАВИЛА:
1. analyst НИКОГДА не первый — он работает по данным от tool.
2. analyst ставится только для расчёта/сравнения/объяснения.
   НЕ ставь analyst для «покажи», «список», «сколько всего» — это делает tool.
3. Правила + данные → [rag, tool] (или [tool, rag]).
4. Правила + данные + анализ → [rag, tool, analyst] или [tool, analyst].
5. Если непонятно → [rag].

ПРИМЕРЫ:
Q: как оформить возврат?                → [{"agent":"rag","purpose":"инструкция"}]
Q: покажи заказы за сентябрь            → [{"agent":"tool","purpose":"заказы"}]
Q: сколько заказов на складе 700?       → [{"agent":"tool","purpose":"агрегат"}]
Q: сравни заказы сентябрь и август      → [{"agent":"tool","purpose":"данные"},{"agent":"analyst","purpose":"сравнить"}]
Q: заказ 98010205 — можно ли отменить?  → [{"agent":"rag","purpose":"правила"},{"agent":"tool","purpose":"статус"}]
Q: почему упала сумма заказов?          → [{"agent":"tool","purpose":"данные"},{"agent":"analyst","purpose":"причины"}]
Q: какие методы API для создания заказа?→ [{"agent":"rag","purpose":"методы API"}]
Q: Какие требования к паллетам, и сравни заказы за сентябрь и август → 
→ [{"agent":"rag","purpose":"требования к паллетам"},{"agent":"tool","purpose":"заказы"},{"agent":"analyst","purpose":"сравнить"}]

Формат ответа — строго JSON по схеме."""


async def _llm_plan(query: str, history: list[ChatHistoryItem] | None = None) -> Plan:
    history = history or []
    
    # Формируем блок истории, если она есть (для контекста планера)
    history_block = ""
    if history:
        recent = history[-4:]  # Берем последние 4 сообщения, чтобы не раздувать промпт
        lines = []
        for m in recent:
            prefix = "Пользователь" if m.role == "user" else "Ассистент"
            content = m.content[:200] + "…" if len(m.content) > 200 else m.content
            lines.append(f"{prefix}: {content}")
        history_block = "\n".join(lines)

    prompt = f"""=== ИСТОРИЯ ДИАЛОГА (для контекста) ===
{history_block if history_block else "(пусто)"}

=== ТЕКУЩИЙ ЗАПРОС ===
{query}

Верни только JSON."""

    answer = await vllm_client.generate_text(
        prompt=prompt,
        system=PLANNER_SYSTEM_PROMPT,
        max_tokens=400,
        temperature=0.0,
        response_format=PLAN_SCHEMA,
    )
    data = json.loads(answer)
    return Plan(**data, source="llm")


# ==========================================================================
# Публичная точка входа
# ==========================================================================
async def make_plan(query: str, forced_route: str | None = None, history: list[ChatHistoryItem] | None = None) -> Plan:
    """
    Возвращает Plan. Никогда не бросает исключение.

    forced_route — если пользователь явно выбрал route (ChatRequest.route != "auto").
    history — история диалога (используется только в LLM-планере для контекста).
    """
    if forced_route and forced_route in _LEGACY_MAP:
        return _plan_from_legacy_route(forced_route, f"выбрано пользователем: {forced_route}")

    if USE_HARD_ROUTE:
        try:
            forced, rule_reason = hard_route(query)
            if forced:
                plan = _plan_from_legacy_route(forced, rule_reason)
                if wants_analyst(query) and any(s.agent == "tool" for s in plan.steps):
                    plan.steps.append(PlanStep(agent="analyst", purpose="проанализировать данные"))
                    plan.reason += " + analyst (prescan)"
                    plan.source = "prescan"
                logger.info(f"[planner] rule → {[s.agent for s in plan.steps]}, reason={rule_reason}")
                return plan
        except Exception as e:
            logger.warning(f"[planner] hard_route упал: {e}")

    if USE_PRESCAN:
        plan = _prescan(query)
        if plan:
            logger.info(f"[planner] prescan → {[s.agent for s in plan.steps]}")
            return plan

    try:
        # Передаем историю в LLM-планер
        plan = await _llm_plan(query, history=history)
        logger.info(f"[planner] llm → {[s.agent for s in plan.steps]}, reason={plan.reason}")
        return plan
    except Exception as e:
        logger.warning(f"[planner] LLM упал: {e} → fallback")

    plan = _prescan(query)
    if plan:
        plan.source = "fallback"
        return plan
    return Plan(
        steps=[PlanStep(agent="rag", purpose="fallback: не удалось определить маршрут")],
        reason="fallback: LLM недоступна",
        source="fallback",
    )