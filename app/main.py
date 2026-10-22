"""
FastAPI-приложение: SSE-стриминг ответов Supervisor для фронта.

Эндпоинты:
- POST /api/v1/ask/stream — SSE-стриминг
- POST /api/v1/ask — обычный JSON
- GET /api/v1/health — health check

Память диалога: stateless. Фронт присылает history/summary в каждом запросе,
бэк возвращает обновлённое summary в metadata (фронт хранит в sessionStorage).
"""
import json
import time
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from app.agents.supervisor import run_supervisor
from app.config import settings
from app.schemas import ChatRequest, HealthStatus
from app.tools.qdrant_client import qdrant_db
from app.tools.triton_client import triton_client


app = FastAPI(title="Assistant Eportal API", version="0.1.0")

# CORS — разрешаем фронту обращаться
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# Helpers
# ==========================================================================
def _extract_jwt(authorization: str | None) -> str:
    """Извлекает JWT из заголовка Authorization: Bearer xxx."""
    if not authorization:
        logger.warning("[auth] Authorization header ОТСУТСТВУЕТ")
        return ""

    logger.info(f"[auth] Authorization: {authorization[:80]}...")

    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        jwt = parts[1]
        logger.info(
            f"[auth] JWT извлечён: {jwt[:40]}... "
            f"(длина: {len(jwt)} символов)"
        )
        return jwt

    logger.warning(f"[auth] Нестандартный формат: {authorization[:80]}")
    return authorization


def _sse_event(data: dict) -> str:
    """Формат SSE-события."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ==========================================================================
# SSE-стриминг — основной эндпоинт фронта
# ==========================================================================
@app.post("/api/v1/ask/stream")
async def ask_stream(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    """
    SSE-стриминг ответа от Supervisor.
    """
    jwt = _extract_jwt(authorization)
    logger.info(
        f"[/ask/stream] route={request.route}, "
        f"memory={len(request.history)} msgs + "
        f"{'summary' if request.summary else 'no-summary'}, "
        f"message={request.message[:80]!r}"
    )

    async def generate() -> AsyncGenerator[str, None]:
        start_time = time.time()
        try:
            async for event in run_supervisor(
                query=request.message,
                route=request.route,
                jwt=jwt,
                history=request.history,
                summary=request.summary,
                summarized_up_to_id=request.summarized_up_to_id,
            ):
                if event.get("type") == "metadata":
                    event["processingTime"] = round(time.time() - start_time, 2)
                yield _sse_event(event)

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("[/ask/stream] ошибка")
            yield _sse_event({"type": "error", "error": str(e)})
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==========================================================================
# Обычный JSON
# ==========================================================================
@app.post("/api/v1/ask")
async def ask(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    """Обычный (нестриминговый) запрос."""
    jwt = _extract_jwt(authorization)
    start_time = time.time()

    answer = ""
    sources = []
    display = None
    route = request.route
    route_reason = ""
    tools_used = []
    new_summary: str | None = None
    new_summarized_up_to_id: str | None = None

    try:
        async for event in run_supervisor(
            query=request.message,
            route=request.route,
            jwt=jwt,
            history=request.history,
            summary=request.summary,
            summarized_up_to_id=request.summarized_up_to_id,
        ):
            etype = event.get("type")
            if etype == "route":
                route = event.get("route", route)
                route_reason = event.get("reason", "")
            elif etype == "metadata":
                answer = event.get("answer", "")
                sources = event.get("sources", [])
                display = event.get("display")
                tools_used = event.get("tools_used", [])
                new_summary = event.get("summary")
                new_summarized_up_to_id = event.get("summarized_up_to_id")
            elif etype == "error":
                return {
                    "success": False,
                    "error": event.get("error", "unknown"),
                }

        return {
            "success": True,
            "response": answer,
            "sources": sources,
            "route": route,
            "route_reason": route_reason,
            "tools_used": tools_used,
            "display": display,
            "processingTime": round(time.time() - start_time, 2),
            "summary": new_summary,
            "summarized_up_to_id": new_summarized_up_to_id,
        }

    except Exception as e:
        logger.exception("[/ask] ошибка")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================================
# ==========================================================================
# Health check
# ==========================================================================
@app.get("/api/v1/health", response_model=HealthStatus)
async def health():
    """Проверка всех сервисов."""
    triton_status = triton_client.health_check()
    qdrant_ok = qdrant_db.health_check()

    vllm_vl_ok = False
    vllm_text_ok = False
    try:
        r = httpx.get(f"{settings.vllm_vl_url}/models", timeout=3.0)
        vllm_vl_ok = r.status_code == 200
    except Exception:
        pass
    try:
        r = httpx.get(f"{settings.vllm_text_url}/models", timeout=3.0)
        vllm_text_ok = r.status_code == 200
    except Exception:
        pass

    return HealthStatus(
        status="ok",
        version="0.1.0",
        supervisor=True,
        vllm_vl=vllm_vl_ok,
        vllm_text=vllm_text_ok,
        triton=triton_status.get("triton", False),
        qdrant=qdrant_ok,
        neo4j=False,
    )


# ==========================================================================
# Проверка JWT (для отладки)
# ==========================================================================
@app.get("/api/v1/whoami")
async def whoami(authorization: str | None = Header(default=None)):
    """Проверка: приходит ли JWT на бэк."""
    jwt = _extract_jwt(authorization)

    return {
        "authorization_header_present": authorization is not None,
        "authorization_header_preview": (
            authorization[:80] + "..." if authorization else None
        ),
        "jwt_extracted": jwt[:40] + "..." if jwt else None,
        "jwt_length": len(jwt),
        "jwt_first_20": jwt[:20] if jwt else None,
        "jwt_last_20": jwt[-20:] if len(jwt) > 20 else None,
    }


# ==========================================================================
# Информация о базе (для отладки)
# ==========================================================================
@app.get("/api/v1/documents")
async def documents():
    """Список загруженных документов (для отладки)."""
    from pathlib import Path

    state_file = Path("data/.ingest_state.json")
    if not state_file.exists():
        return {"documents": []}

    state = json.loads(state_file.read_text())
    docs = state.get("documents", {})

    return {
        "total": len(docs),
        "documents": [
            {
                "document_id": doc_id,
                "source_file": info.get("source_file", ""),
                "chunks": info.get("chunks", 0),
                "status": info.get("status", ""),
            }
            for doc_id, info in docs.items()
        ],
    }


# ==========================================================================
# Запуск
# ==========================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )