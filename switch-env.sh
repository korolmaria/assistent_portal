#!/bin/bash
# ============================================================
# switch-env.sh — Переключение SupplierPortal между dev и prod
#
# Использование:
#   ./switch-env.sh dev       — переключить на dev + рестарт API
#   ./switch-env.sh prod      — переключить на prod + рестарт API
#   ./switch-env.sh status    — показать текущий режим
#   ./switch-env.sh current   — показать активные настройки (без рестарта)
# ============================================================

set -e

PROJECT_DIR="/root/assistant-eportal"
cd "$PROJECT_DIR"

ENV_FILE=".env"

# ============================================================
# Проверка аргументов
# ============================================================
if [ -z "$1" ]; then
    echo ""
    echo "Использование: $0 dev|prod|status|current"
    echo ""
    echo "Опции:"
    echo "  dev      — переключить на dev-окружение + рестарт API"
    echo "  prod     — переключить на prod-окружение + рестарт API"
    echo "  status   — показать текущий режим (из .env)"
    echo "  current  — показать активные настройки (загруженные в config)"
    echo ""
    exit 1
fi

# ============================================================
# Проверка, что .env есть
# ============================================================
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ $ENV_FILE не найден в $PROJECT_DIR"
    exit 1
fi

# ============================================================
# STATUS — показать текущий режим из .env
# ============================================================
if [ "$1" = "status" ]; then
    env=$(grep "^SUPPLIER_PORTAL_ENV=" "$ENV_FILE" | cut -d= -f2)

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "📊 SupplierPortal — текущий режим"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "  ENV: $env"
    echo ""

    if [ "$env" = "prod" ]; then
        url=$(grep "^SUPPLIER_PORTAL_URL_PROD=" "$ENV_FILE" | cut -d= -f2)
        sn=$(grep "^SUPPLIER_PORTAL_SERVICE_NUMBER_PROD=" "$ENV_FILE" | cut -d= -f2)
    else
        url=$(grep "^SUPPLIER_PORTAL_URL_DEV=" "$ENV_FILE" | cut -d= -f2)
        sn=$(grep "^SUPPLIER_PORTAL_SERVICE_NUMBER_DEV=" "$ENV_FILE" | cut -d= -f2)
    fi

    echo "  URL:           $url"
    echo "  ServiceNumber: $sn"
    echo ""

    if [ -z "$sn" ]; then
        echo "  ⚠️  ServiceNumber пустой!"
    fi
    echo ""
    exit 0
fi

# ============================================================
# CURRENT — показать активные настройки (загруженные в config)
# ============================================================
if [ "$1" = "current" ]; then
    source venv/bin/activate 2>/dev/null || true

    python -c "
from app.config import settings
print()
print('════════════════════════════════════════════════════════════')
print('📊 SupplierPortal — активные настройки (в config)')
print('════════════════════════════════════════════════════════════')
print()
print(f'  ENV:           {settings.supplier_portal_env}')
print(f'  URL:           {settings.supplier_portal_url}')
print(f'  ServiceNumber: {settings.supplier_portal_service_number}')
print(f'  Full URL:      {settings.supplier_portal_url}/Json')
print()
"
    exit 0
fi

# ============================================================
# Переключение dev/prod
# ============================================================
if [ "$1" != "dev" ] && [ "$1" != "prod" ]; then
    echo "❌ Неизвестный режим: $1"
    echo "Использование: $0 dev|prod|status|current"
    exit 1
fi

TARGET_ENV="$1"

# Бэкап .env
cp "$ENV_FILE" "$ENV_FILE.bak.$(date +%s)"

# Меняем SUPPLIER_PORTAL_ENV
if grep -q "^SUPPLIER_PORTAL_ENV=" "$ENV_FILE"; then
    sed -i "s/^SUPPLIER_PORTAL_ENV=.*/SUPPLIER_PORTAL_ENV=$TARGET_ENV/" "$ENV_FILE"
else
    echo "" >> "$ENV_FILE"
    echo "SUPPLIER_PORTAL_ENV=$TARGET_ENV" >> "$ENV_FILE"
fi

# Получаем URL и ServiceNumber для выбранного режима
if [ "$TARGET_ENV" = "prod" ]; then
    url=$(grep "^SUPPLIER_PORTAL_URL_PROD=" "$ENV_FILE" | cut -d= -f2)
    sn=$(grep "^SUPPLIER_PORTAL_SERVICE_NUMBER_PROD=" "$ENV_FILE" | cut -d= -f2)
else
    url=$(grep "^SUPPLIER_PORTAL_URL_DEV=" "$ENV_FILE" | cut -d= -f2)
    sn=$(grep "^SUPPLIER_PORTAL_SERVICE_NUMBER_DEV=" "$ENV_FILE" | cut -d= -f2)
fi

# Вывод
echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Режим переключён на: $TARGET_ENV"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  URL:           $url"
echo "  ServiceNumber: $sn"
echo ""

# Проверки
if [ -z "$url" ]; then
    echo "  ⚠️  URL пустой! Проверьте .env"
fi

if [ -z "$sn" ]; then
    echo "  ⚠️  ServiceNumber пустой — API может не работать"
    echo "     Заполните SUPPLIER_PORTAL_SERVICE_NUMBER_$(echo $TARGET_ENV | tr '[:lower:]' '[:upper:]') в .env"
fi

# ============================================================
# Перезапуск FastAPI
# ============================================================
echo ""
echo "🔄 Перезапуск FastAPI..."

if systemctl is-enabled assistant-api > /dev/null 2>&1; then
    echo "   → через systemd"
    sudo systemctl restart assistant-api
    sleep 3
else
    echo "   → через nohup"
    pkill -f "uvicorn app.main" 2>/dev/null || true
    sleep 1
    nohup uvicorn app.main:app --host 0.0.0.0 --port 8080 > /tmp/api.log 2>&1 &
    echo $! > /tmp/api.pid
    sleep 5
fi

# ============================================================
# Проверка health
# ============================================================
echo ""
echo "⏳ Проверка health API..."

elapsed=0
while [ $elapsed -lt 30 ]; do
    health=$(curl -s http://localhost:8080/api/v1/health 2>/dev/null || echo "")
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
        echo "   sudo journalctl -u assistant-api -n 20 --no-pager"
    else
        echo "   tail -20 /tmp/api.log"
    fi
fi

# ============================================================
# Проверка доступности SupplierPortal
# ============================================================
if [ -n "$url" ]; then
    echo ""
    echo "🌐 Проверка доступности SupplierPortal..."
    if curl -k -s -o /dev/null -w "%{http_code}" --max-time 10 "$url/Json" 2>/dev/null | grep -qE "200|400|401|403"; then
        echo "✅ SupplierPortal доступен ($url/Json)"
    else
        code=$(curl -k -s -o /dev/null -w "%{http_code}" --max-time 10 "$url/Json" 2>/dev/null || echo "timeout")
        echo "⚠️  SupplierPortal вернул: $code"
        echo "   Возможно нужен VPN или URL недоступен с этого сервера"
    fi
fi

# ============================================================
# Итог
# ============================================================
echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Переключение завершено: $TARGET_ENV"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📋 Команды:"
echo "   ./switch-env.sh status    — текущий режим"
echo "   ./switch-env.sh current   — активные настройки в config"
echo "   ./switch-env.sh dev       — переключить на dev"
echo "   ./switch-env.sh prod      — переключить на prod"
echo ""
