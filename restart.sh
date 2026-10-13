#!/bin/bash
# ============================================================
# restart.sh — Перезапуск FastAPI в процессе разработки
#
# Что делает:
#   1. Останавливает FastAPI
#   2. Запускает заново
#   3. Проверяет health
#
# НЕ трогает Docker-контейнеры (vLLM, Triton, Qdrant, Neo4j) —
# они грузятся долго и перезапускать их при каждой правке кода
# не нужно.
#
# Использование:
#   ./restart.sh              — перезапуск API
#   ./restart.sh --logs       — перезапуск + показать логи
#   ./restart.sh --rebuild    — перезапуск + переустановить зависимости
# ============================================================

set -e

PROJECT_DIR="/root/assistant-eportal"
API_PORT=8080

SHOW_LOGS=false
REBUILD=false

for arg in "$@"; do
    case $arg in
        --logs)    SHOW_LOGS=true ;;
        --rebuild) REBUILD=true ;;
    esac
done

cd "$PROJECT_DIR"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🔄 Assistant Eportal — перезапуск FastAPI"
echo "════════════════════════════════════════════════════════════"
echo ""

# === 1. Активация venv ===
source venv/bin/activate
echo "✅ venv активирован"

# === 2. Переустановка зависимостей (опционально) ===
if [ "$REBUILD" = true ]; then
    echo ""
    echo "📦 Переустановка зависимостей..."
    pip install -r requirements.txt
fi

# === 3. Остановка FastAPI ===
echo ""
echo "⏹️  Остановка FastAPI..."

if systemctl is-enabled assistant-api > /dev/null 2>&1; then
    sudo systemctl stop assistant-api
    echo "   → остановлен через systemd"
else
    pkill -f "uvicorn app.main" 2>/dev/null || true
    echo "   → остановлен через pkill"
fi

sleep 2

# === 4. Запуск FastAPI ===
echo ""
echo "🚀 Запуск FastAPI..."

if systemctl is-enabled assistant-api > /dev/null 2>&1; then
    echo "   → через systemd"
    sudo systemctl start assistant-api
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
echo "⏳ Проверка health..."

elapsed=0
while [ $elapsed -lt 30 ]; do
    health=$(curl -s "http://localhost:$API_PORT/api/v1/health" 2>/dev/null || echo "")
    if echo "$health" | grep -q '"status":"ok"'; then
        echo "✅ API готов за ${elapsed}с"
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

if [ $elapsed -ge 30 ]; then
    echo "⚠️  API не ответил за 30с. Логи:"
    if systemctl is-enabled assistant-api > /dev/null 2>&1; then
        sudo journalctl -u assistant-api -n 30 --no-pager
    else
        tail -30 /tmp/api.log
    fi
    exit 1
fi

# === 6. Логи (опционально) ===
if [ "$SHOW_LOGS" = true ]; then
    echo ""
    echo "📋 Логи (Ctrl+C для выхода):"
    echo ""
    if systemctl is-enabled assistant-api > /dev/null 2>&1; then
        sudo journalctl -u assistant-api -f
    else
        tail -f /tmp/api.log
    fi
fi

# === 7. Итог ===
echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Перезапуск завершён за ${elapsed}с"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "🌐 API:  http://localhost:$API_PORT"
echo "📊 Health:  curl http://localhost:$API_PORT/api/v1/health"
echo ""
echo "💡 Показать логи:  ./restart.sh --logs"
echo ""
