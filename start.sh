#!/bin/bash
# ============================================================
# start.sh — Первый запуск после выключения сервера
#
# Что делает:
#   1. Запускает Docker-контейнеры
#   2. Ждёт готовности vLLM (VL + Text) и Triton
#   3. Запускает FastAPI
#   4. Проверяет health
#
# Использование:
#   ./start.sh              — полный запуск
#   ./start.sh --no-wait    — без ожидания готовности vLLM
#   ./start.sh --fast       — только FastAPI (без контейнеров)
#   ./start.sh --ingest     — после запуска загрузить новые документы
#   ./start.sh --help       — показать справку
# ============================================================

set -e

# === Настройки ===
PROJECT_DIR="/root/assistant-eportal"
API_PORT=8080
VLLM_VL_PORT=8000
VLLM_TEXT_PORT=8010
TRITON_PORT=8001
WAIT_TIMEOUT=300
WAIT_INTERVAL=10

# === Аргументы ===
SKIP_WAIT=false
FAST_MODE=false
RUN_INGEST=false

show_help() {
    echo ""
    echo "Использование: ./start.sh [опции]"
    echo ""
    echo "Опции:"
    echo "  --ingest     После запуска загрузить новые документы из data/uploads/"
    echo "  --no-wait    Не ждать готовности vLLM (3-5 минут)"
    echo "  --fast       Только FastAPI (Docker-контейнеры не трогать)"
    echo "  --help       Показать эту справку"
    echo ""
    exit 0
}

for arg in "$@"; do
    case $arg in
        --no-wait) SKIP_WAIT=true ;;
        --fast)    FAST_MODE=true ;;
        --ingest)  RUN_INGEST=true ;;
        --help)    show_help ;;
        *)
            echo "⚠️  Неизвестный аргумент: $arg"
            show_help
            ;;
    esac
done

# === Переход в проект ===
cd "$PROJECT_DIR"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🚀 Assistant Eportal — запуск"
echo "════════════════════════════════════════════════════════════"
echo ""

# === 1. Активация venv ===
if [ ! -d "venv" ]; then
    echo "❌ venv не найден. Создайте: python3 -m venv venv"
    exit 1
fi
source venv/bin/activate
echo "✅ venv активирован"

# === 2. Запуск Docker-контейнеров ===
if [ "$FAST_MODE" = false ]; then
    echo ""
    echo "📦 Запуск Docker-контейнеров..."
    docker compose up -d

    echo ""
    echo "⏳ Ожидание готовности контейнеров..."
    sleep 10

    echo ""
    docker compose ps
else
    echo ""
    echo "⚡ Fast-режим: контейнеры не трогаем"
fi

# === 3. Ожидание готовности vLLM ===
if [ "$SKIP_WAIT" = false ] && [ "$FAST_MODE" = false ]; then
    echo ""
    echo "⏳ Ожидание vLLM VL (порт $VLLM_VL_PORT)..."

    elapsed=0
    while [ $elapsed -lt $WAIT_TIMEOUT ]; do
        if curl -s -f "http://localhost:$VLLM_VL_PORT/v1/models" > /dev/null 2>&1; then
            echo "✅ vLLM VL готов (${elapsed}с)"
            break
        fi
        sleep $WAIT_INTERVAL
        elapsed=$((elapsed + WAIT_INTERVAL))
        echo "   ... ${elapsed}с"
    done

    if [ $elapsed -ge $WAIT_TIMEOUT ]; then
        echo "⚠️  vLLM VL не готов за ${WAIT_TIMEOUT}с. Логи:"
        echo "   docker compose logs vllm-qwen-vl"
    fi

    echo ""
    echo "⏳ Ожидание vLLM Text (порт $VLLM_TEXT_PORT)..."

    elapsed=0
    while [ $elapsed -lt $WAIT_TIMEOUT ]; do
        if curl -s -f "http://localhost:$VLLM_TEXT_PORT/v1/models" > /dev/null 2>&1; then
            echo "✅ vLLM Text готов (${elapsed}с)"
            break
        fi
        sleep $WAIT_INTERVAL
        elapsed=$((elapsed + WAIT_INTERVAL))
        echo "   ... ${elapsed}с"
    done

    echo ""
    echo "⏳ Ожидание Triton (порт $TRITON_PORT)..."

    elapsed=0
    while [ $elapsed -lt $WAIT_TIMEOUT ]; do
        if curl -s -f "http://localhost:$TRITON_PORT/v2/health/ready" > /dev/null 2>&1; then
            echo "✅ Triton готов (${elapsed}с)"
            break
        fi
        sleep $WAIT_INTERVAL
        elapsed=$((elapsed + WAIT_INTERVAL))
        echo "   ... ${elapsed}с"
    done
fi

# === 4. Запуск FastAPI ===
echo ""
echo "🚀 Запуск FastAPI..."

pkill -f "uvicorn app.main" 2>/dev/null || true

if systemctl is-enabled assistant-api > /dev/null 2>&1; then
    echo "   → через systemd"
    sudo systemctl restart assistant-api
    sleep 3
else
    echo "   → через nohup"
    nohup uvicorn app.main:app --host 0.0.0.0 --port $API_PORT > /tmp/api.log 2>&1 &
    echo $! > /tmp/api.pid
    echo "   API PID: $(cat /tmp/api.pid)"
    sleep 5
fi

# === 5. Проверка health ===
echo ""
echo "⏳ Проверка health API..."

elapsed=0
while [ $elapsed -lt 60 ]; do
    health=$(curl -s "http://localhost:$API_PORT/api/v1/health" 2>/dev/null || echo "")
    if echo "$health" | grep -q '"status":"ok"'; then
        echo "✅ API готов"
        echo ""
        echo "$health" | python -m json.tool
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

if [ $elapsed -ge 60 ]; then
    echo "⚠️  API не ответил за 60с. Логи:"
    echo "   tail -30 /tmp/api.log"
fi

# === 6. Ingest документов (опционально) ===
if [ "$RUN_INGEST" = true ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "📥 Ingest документов"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "⚠️  Будут обработаны только НОВЫЕ документы."
    echo "   Уже загруженные пропускаются (журнал .ingest_state.json)."
    echo ""
    read -p "Продолжить? (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        python -m scripts.ingest_docs --dir data/uploads/
    else
        echo "Отменено."
    fi
fi

# === 7. Итог ===
echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Запуск завершён"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "🌐 API:            http://localhost:$API_PORT"
echo "📊 Health:         http://localhost:$API_PORT/api/v1/health"
echo "📚 Swagger UI:     http://localhost:$API_PORT/docs"
echo "📄 Документы:      http://localhost:$API_PORT/api/v1/documents"
echo ""
echo "🔗 Доступ с Mac:   http://89.111.152.207:$API_PORT"
echo ""
echo "📋 Полезные команды:"
echo "   docker compose ps                    — статус контейнеров"
echo "   docker compose logs -f vllm-qwen-vl  — логи vLLM VL"
echo "   sudo systemctl status assistant-api  — статус API"
echo "   sudo journalctl -u assistant-api -f  — логи API"
echo "   tail -30 /tmp/api.log                — логи (если nohup)"
echo ""
echo "🔄 Перезапуск API после правок:  ./restart.sh"
echo "📊 Проверка состояния:           ./status.sh"
echo ""