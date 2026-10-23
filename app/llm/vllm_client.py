"""
Клиент к vLLM (OpenAI-совместимый API).

Поддерживает:
- Текстовую генерацию через Qwen3.8-27B
- Мультимодальные запросы через Qwen2.5-VL (base64 картинки)
- Tool calling для агентов (function calling)
- Retry-логику через tenacity
"""
import base64
from pathlib import Path
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings


class VLLMClient:
    """Асинхронный клиент к двум инстансам vLLM (VL + Text)."""

    def __init__(self) -> None:
        # Клиент для VL-модели (парсинг документов)
        self.vl_client = AsyncOpenAI(
            base_url=settings.vllm_vl_url,
            api_key="EMPTY",  # vLLM не требует ключ
        )
        # Клиент для текстовой модели (агенты)
        self.text_client = AsyncOpenAI(
            base_url=settings.vllm_text_url,
            api_key="EMPTY",
        )

    # ------------------------------------------------------------------
    # Текстовая генерация
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate_text(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        top_p: float = 0.95,
        response_format: dict | None = None,
        enable_thinking: bool = False,  # <-- НОВЫЙ ПАРАМЕТР
    ) -> str:
        """Простая текстовая генерация через Qwen3.8-27B."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {}
        if response_format:
            kwargs["response_format"] = response_format

        # ⚠️ Отключаем/включаем размышления
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

        response = await self.text_client.chat.completions.create(
            model=settings.text_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,  # <-- ПЕРЕДАЁМ ФЛАГ
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        logger.debug(
            f"[generate_text] tokens in/out: "
            f"{response.usage.prompt_tokens}/{response.usage.completion_tokens} "
            f"(thinking={enable_thinking})"
        )
        return content

    # ------------------------------------------------------------------
    # Мультимодальная генерация (VL)
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate_with_image(
        self,
        image_path: str | Path,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        """
        Отправляет изображение + текст в Qwen2.5-VL.
        image_path: путь к картинке (PNG/JPG).
        prompt: инструкция, что извлечь из картинки.
        """
        # Читаем картинку и кодируем в base64
        img_bytes = Path(image_path).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        # Определяем mime-тип (по умолчанию PNG)
        ext = Path(image_path).suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(ext, "image/png")

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                },
            ],
        })

        response = await self.vl_client.chat.completions.create(
            model=settings.vl_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        logger.debug(
            f"[generate_with_image] tokens in/out: "
            f"{response.usage.prompt_tokens}/{response.usage.completion_tokens}"
        )
        return content

    # ------------------------------------------------------------------
    # Tool calling (для Supervisor)
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.1,
        tool_choice: str = "auto",
        enable_thinking: bool = False,  # <-- НОВЫЙ ПАРАМЕТР
    ) -> Any:
        """
        Чат с поддержкой function calling.
        Возвращает response.choices[0].message целиком (там может быть tool_calls).
        """
        # ⚠️ Отключаем/включаем размышления
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

        response = await self.text_client.chat.completions.create(
            model=settings.text_model_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,  # <-- ПЕРЕДАЁМ ФЛАГ
        )
        return response.choices[0].message

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    async def health_check(self) -> dict[str, bool]:
        """Проверка доступности обоих vLLM-инстансов."""
        result = {"vl": False, "text": False}
        try:
            await self.vl_client.models.list()
            result["vl"] = True
        except Exception as e:
            logger.warning(f"VL health check failed: {e}")
        try:
            await self.text_client.models.list()
            result["text"] = True
        except Exception as e:
            logger.warning(f"Text health check failed: {e}")
        return result


# Singleton
vllm_client = VLLMClient()