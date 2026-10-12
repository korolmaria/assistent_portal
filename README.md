# Assistant Eportal

Мультиагентная RAG-система для работы с документами поставщиков торговой сети «Евроторг».

## Возможности

- Ingest PDF + DOCX в Qdrant
- Гибридный поиск (dense + sparse + RRF)
- Реранкинг (BGE-reranker-v2-m3)
- RAG-агент с цитатами [doc_id:pN]
- Tool-агент (SupplierPortal API)
- Supervisor (маршрутизация auto / rag / tool / rag+tool)
- FastAPI + SSE
- Расширение соседями для технических описаний

## Стек

| Компонент | Технология |
|---|---|
| LLM (VL) | Qwen2.5-VL-7B-AWQ (vLLM) |
| LLM (Text) | Qwen3-8B-AWQ (vLLM) |
| Embeddings | BGE-M3 (Triton) |
| Reranker | BGE-reranker-v2-m3 (Triton) |
| Vector DB | Qdrant |
| Graph DB | Neo4j (в разработке) |
| API | FastAPI + SSE |

## Требования

- Docker + Docker Compose
- 2x GPU (RTX A5000 или аналоги)
- NVIDIA Container Toolkit
- Python 3.11
- 50 ГБ свободного места

## Установка

1. Клонировать репозиторий:

   git clone https://github.com/USERNAME/assistant-eportal.git
   cd assistant-eportal

2. Создать окружение:

   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. Настроить .env:

   cp .env.example .env
   nano .env

4. Запустить сервисы:

   docker compose up -d

5. Загрузить документы:

   python -m scripts.ingest_docs --dir data/uploads/

6. Запустить API:

   uvicorn app.main:app --host 0.0.0.0 --port 8080

## API

POST /api/v1/ask/stream - SSE-стриминг
POST /api/v1/ask - обычный JSON
GET /api/v1/health - статус сервисов
GET /api/v1/documents - список документов

## Структура проекта

app/agents/ - RAG, Tool, Synthesis, Supervisor
app/ingest/ - Парсинг PDF/DOCX, чанкинг, индексация
app/llm/ - Клиент vLLM
app/tools/ - Triton, Qdrant, SupplierPortal
app/config.py - Настройки
app/schemas.py - Pydantic-схемы
app/main.py - FastAPI + SSE

## Лицензия

Proprietary - внутренний проект ООО «Евроторг».
