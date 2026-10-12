"""
Клиент к Triton Inference Server.

Две модели:
- bge_m3: dense-эмбеддинги (1024-dim), L2-нормализованные
- bge_reranker: cross-encoder BGE-reranker-v2-m3, скор релевантности пары (query, doc)

Токенизаторы загружаются локально из triton-models/<model>/1/.
"""
from functools import lru_cache

import numpy as np
import tritonclient.http as httpclient
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer

from app.config import settings


# ==========================================================================
# Пути к токенизаторам
# ==========================================================================
BGE_M3_TOKENIZER_PATH = "triton-models/bge_m3/1"
RERANKER_TOKENIZER_PATH = "triton-models/bge_reranker/1"


# ==========================================================================
# Токенизаторы (кэшируем)
# ==========================================================================
@lru_cache(maxsize=1)
def get_bge_tokenizer() -> AutoTokenizer:
    """Токенизатор BGE-M3 (XLM-RoBERTa). pad_token задаём вручную."""
    tok = AutoTokenizer.from_pretrained(BGE_M3_TOKENIZER_PATH)
    if tok.pad_token is None:
        tok.pad_token = "</s>"
    logger.info(f"[triton] BGE-M3 tokenizer loaded, pad_token={tok.pad_token!r}")
    return tok


@lru_cache(maxsize=1)
def get_reranker_tokenizer() -> AutoTokenizer:
    """Токенизатор BGE-reranker-v2-m3 (XLM-RoBERTa)."""
    tok = AutoTokenizer.from_pretrained(RERANKER_TOKENIZER_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "</s>"
    logger.info(f"[triton] Reranker tokenizer loaded, pad_token={tok.pad_token!r}")
    return tok


# ==========================================================================
# Клиент
# ==========================================================================
class TritonClient:
    """Обёртка над Triton (синхронный HTTP-клиент)."""

    def __init__(self) -> None:
        self.url = settings.triton_url
        self.bge_model = settings.triton_bge_model
        self.reranker_model = settings.triton_reranker_model
        self._client: httpclient.InferenceServerClient | None = None

    @property
    def client(self) -> httpclient.InferenceServerClient:
        if self._client is None:
            self._client = httpclient.InferenceServerClient(url=self.url)
        return self._client

    # ----------------------------------------------------------------------
    # Эмбеддинги (BGE-M3, dense, 1024-dim, L2-нормализованные)
    # ----------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    def embed(self, texts: list[str], max_length: int = 512) -> np.ndarray:
        """
        Считает dense-эмбеддинги для списка текстов.
        Возвращает np.ndarray формы (N, 1024), L2-нормализованные.
        """
        if not texts:
            return np.zeros((0, settings.embedding_dim), dtype=np.float32)

        tok = get_bge_tokenizer()
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [httpclient.InferRequestedOutput("sentence_embedding")]
        result = self.client.infer(self.bge_model, inputs, outputs=outputs)
        emb = result.as_numpy("sentence_embedding")

        # L2-нормализация (BGE-M3 отдаёт нормализованные, но подстрахуемся)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms

        logger.debug(f"[triton] embed: {len(texts)} texts → shape {emb.shape}")
        return emb.astype(np.float32)

    # ----------------------------------------------------------------------
    # Реранкинг (BGE-reranker-v2-m3)
    # ----------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    def rerank(
        self,
        query: str,
        documents: list[str],
        max_length: int = 512,
    ) -> list[float]:
        """
        BGE-reranker-v2-m3: скор релевантности каждой пары (query, doc).
        Возвращает список float в [0, 1] в порядке documents.
        """
        if not documents:
            return []

        tok = get_reranker_tokenizer()

        # Пары (query, doc) — BGE-reranker принимает обе части
        pairs = [(query, doc) for doc in documents]
        enc = tok(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [httpclient.InferRequestedOutput("logits")]
        result = self.client.infer(self.reranker_model, inputs, outputs=outputs)
        logits = result.as_numpy("logits")  # (N, 1)

        # Сигмоида → [0, 1]
        scores = 1 / (1 + np.exp(-logits[:, 0]))

        logger.debug(
            f"[triton] rerank: {len(documents)} docs, "
            f"min={scores.min():.4f}, max={scores.max():.4f}"
        )
        return scores.tolist()

    # ----------------------------------------------------------------------
    # Health
    # ----------------------------------------------------------------------
    def health_check(self) -> dict[str, bool]:
        """Проверка готовности Triton и обеих моделей."""
        result = {"triton": False, "bge_m3": False, "reranker": False}
        try:
            if self.client.is_server_ready():
                result["triton"] = True
            if self.client.is_model_ready(self.bge_model):
                result["bge_m3"] = True
            if self.client.is_model_ready(self.reranker_model):
                result["reranker"] = True
        except Exception as e:
            logger.warning(f"[triton] health check failed: {e}")
        return result


# Singleton
triton_client = TritonClient()
