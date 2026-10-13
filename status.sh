#!/bin/bash
# ============================================================
# status.sh — Быстрая проверка статуса системы
# ============================================================

PROJECT_DIR="/root/assistant-eportal"
cd "$PROJECT_DIR"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 Assistant Eportal — статус"
echo "════════════════════════════════════════════════════════════"
echo ""

# === Docker ===
echo "🐳 Docker-контейнеры:"
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "   ❌ Docker недоступен"
echo ""

# === FastAPI ===
echo "🚀 FastAPI:"
if systemctl is-active assistant-api > /dev/null 2>&1; then
    echo "   ✅ systemd: active"
elif ps aux | grep -q "[u]vicorn app.main"; then
    echo "   ✅ nohup: работает"
else
    echo "   ❌ не запущен"
fi
echo ""

# === Health ===
echo "📊 Health API:"
health=$(curl -s http://localhost:8080/api/v1/health 2>/dev/null)
if echo "$health" | grep -q '"status":"ok"'; then
    echo "$health" | python -m json.tool | sed 's/^/   /'
else
    echo "   ❌ API не отвечает"
fi
echo ""

# === vLLM ===
echo "🤖 vLLM VL (порт 8000):"
if curl -s -f http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "   ✅ готов"
else
    echo "   ⏳ загружается или не работает"
fi
echo ""

echo "🤖 vLLM Text (порт 8010):"
if curl -s -f http://localhost:8010/v1/models > /dev/null 2>&1; then
    echo "   ✅ готов"
else
    echo "   ⏳ загружается или не работает"
fi
echo ""

# === Triton ===
echo "🔧 Triton (порт 8001):"
if curl -s -f http://localhost:8001/v2/health/ready > /dev/null 2>&1; then
    echo "   ✅ готов"
    bge=$(curl -s http://localhost:8001/v2/models/bge_m3 2>/dev/null | head -c 30)
    reranker=$(curl -s http://localhost:8001/v2/models/bge_reranker 2>/dev/null | head -c 30)
    [ -n "$bge" ] && echo "      bge_m3: ✅"
    [ -n "$reranker" ] && echo "      bge_reranker: ✅"
else
    echo "   ⏳ загружается или не работает"
fi
echo ""

# === Qdrant ===
echo "📦 Qdrant:"
qdrant=$(curl -s http://localhost:6333/collections 2>/dev/null)
if echo "$qdrant" | grep -q "collections"; then
    points=$(curl -s http://localhost:6333/collections/documents 2>/dev/null | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('result',{}).get('points_count', 0))" 2>/dev/null || echo "?")
    echo "   ✅ работает"
    echo "   Чанков в 'documents': $points"
else
    echo "   ❌ не отвечает"
fi
echo ""

# === GPU ===
echo "🎮 GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU %s (%s): %s / %s МБ, загрузка %s%%\n", $1, $2, $3, $4, $5}'
echo ""

# === Диск ===
echo "💾 Диск:"
df -h / | tail -1 | awk '{printf "   %s / %s (%s использовано)\n", $3, $2, $5}'
echo ""

echo "════════════════════════════════════════════════════════════"
echo ""
