"""Lightweight helpers shared by CORE-T scripts."""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _sanitize_path_component(value: object) -> str:
    """Best-effort filesystem-safe-ish component for cache/results paths."""
    s = str(value or "").strip()
    if not s:
        return "unknown"
    return "".join(c if c.isalnum() else "_" for c in s)


def make_run_tag(*, llm_model: str, embedding_model: str) -> str:
    """Return `{llm}_{embedding_model}` with safe path characters."""
    return f"{_sanitize_path_component(llm_model)}_{_sanitize_path_component(embedding_model)}"


def get_results_run_dir(
    *,
    dataset: str,
    step_dirname: str,
    llm_model: str,
    embedding_model: str,
    project_root: Optional[Path] = None,
) -> Path:
    """Return `<project_root>/results/<dataset>/{llm}_{embedding_model}/<step_dirname>`."""
    root = project_root or _detect_project_root()
    run_tag = make_run_tag(llm_model=llm_model, embedding_model=embedding_model)
    return (root / "results" / str(dataset) / run_tag / str(step_dirname)).resolve()


class DatabaseType(str, Enum):
    """Supported database types."""

    SPIDER = "spider"
    BIRD = "bird"
    MMQA = "mmqa"


class Configuration:
    """Minimal configuration used by CORE-T scripts."""

    def __init__(
        self,
        database_type: DatabaseType = DatabaseType.BIRD,
        llm_model: str = "openai:gpt-4o-mini",
        embedding_model: str = "fireworks:WhereIsAI/UAE-Large-V1",
        temperature: float = 0.0,
        **_kwargs,
    ):
        self.database_type = database_type
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature

    def get_database_cache_dir(self, name: str) -> str:
        """Return `<project_root>/cache/<dataset>/{llm}_{embedding_model}/<name>`."""
        root = _detect_project_root()
        run_tag = make_run_tag(llm_model=self.llm_model, embedding_model=self.embedding_model)
        return str(root / "cache" / self.database_type.value / run_tag / name)

    def create_llm(self):
        """Create a chat model used by standalone scripts.

        Supports common LangChain providers (OpenAI, Together, Gemini, etc.) via
        `init_chat_model`, and additionally supports local Hugging Face models
        (e.g., `huggingface:meta-llama/Llama-3.1-8B-Instruct`) via a transformers
        pipeline wrapped as a LangChain chat model.
        """

        def _is_huggingface_model(model_name: str) -> bool:
            return isinstance(model_name, str) and (
                model_name.startswith("huggingface:") or model_name.startswith("hf:")
            )

        def _create_huggingface_chat(model_name: str):
            """Create a ChatHuggingFace model backed by a transformers pipeline.

            This mirrors the cluster setup used for:
            - meta-llama/Llama-3.1-8B-Instruct
            - Qwen/Qwen2.5-7B-Instruct
            """
            try:
                import torch  # type: ignore
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
                from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline  # type: ignore
            except Exception as import_exc:  # pragma: no cover
                raise ImportError(
                    "Hugging Face support requires 'transformers', 'torch', and 'langchain-huggingface'. "
                    "Install with: pip install -U transformers torch langchain-huggingface"
                ) from import_exc

            # Strip provider prefix to get the HF repo id.
            repo_id = model_name.split(":", 1)[1] if ":" in model_name else model_name

            is_llama_3_1_8b = repo_id == "meta-llama/Llama-3.1-8B-Instruct"
            is_qwen_2_5_7b = repo_id == "Qwen/Qwen2.5-7B-Instruct"

            # Ensure HF caches are writable inside the project directory.
            try:
                project_root = _detect_project_root()
                hf_home = project_root / ".hf_cache"
                transformers_cache = hf_home / "transformers"
                hub_cache = hf_home / "hub"
                hf_home.mkdir(parents=True, exist_ok=True)
                transformers_cache.mkdir(parents=True, exist_ok=True)
                hub_cache.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("HF_HOME", str(hf_home))
                os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
                os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
            except Exception:
                pass

            # Respect an HF token from environment for gated repos.
            try:
                hf_token = (
                    os.environ.get("HUGGINGFACEHUB_API_TOKEN")
                    or os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                )
                if hf_token:
                    os.environ.setdefault("HF_TOKEN", hf_token)
                    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
            except Exception:
                pass

            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                use_fast=True,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
            )
            if is_llama_3_1_8b and getattr(tokenizer, "pad_token_id", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Use device_map="auto" only if accelerate is installed.
            has_accelerate = False
            try:
                import accelerate  # type: ignore  # noqa: F401

                has_accelerate = True
            except Exception:
                has_accelerate = False

            try:
                model_kwargs = {
                    "torch_dtype": torch.bfloat16 if (is_llama_3_1_8b or is_qwen_2_5_7b) else "auto",
                    "cache_dir": os.environ.get("TRANSFORMERS_CACHE"),
                }
                if has_accelerate:
                    model_kwargs["device_map"] = "auto"
                model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
                )

            do_sample = bool(self.temperature and self.temperature > 0.0)
            gen_kwargs = {
                "max_new_tokens": 16384,
                "do_sample": do_sample,
                "pad_token_id": getattr(tokenizer, "eos_token_id", None),
                "return_full_text": False,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(0.0, float(self.temperature or 0.0))

            gen_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **gen_kwargs,
            )
            hf_llm = HuggingFacePipeline(pipeline=gen_pipe)

            # For these instruct models, apply the tokenizer chat template.
            if is_llama_3_1_8b or is_qwen_2_5_7b:

                def _convert_messages_to_prompt(messages):
                    chat = []
                    for m in messages:
                        m_type = getattr(m, "type", "human")
                        if m_type == "system":
                            role = "system"
                        elif m_type == "human":
                            role = "user"
                        elif m_type in ("ai", "assistant"):
                            role = "assistant"
                        else:
                            role = "user"
                        chat.append({"role": role, "content": getattr(m, "content", "")})
                    return tokenizer.apply_chat_template(
                        chat,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                return ChatHuggingFace(llm=hf_llm, convert_messages_to_prompt=_convert_messages_to_prompt)

            return ChatHuggingFace(llm=hf_llm)

        # Hugging Face (local) path first: `init_chat_model` doesn't handle this prefix.
        if _is_huggingface_model(self.llm_model):
            return _create_huggingface_chat(self.llm_model)

        try:
            from langchain.chat_models.base import init_chat_model
        except Exception as e:  # pragma: no cover
            raise ImportError("Missing LangChain dependency for LLM initialization.") from e
        return init_chat_model(self.llm_model, temperature=self.temperature)

    def format_prompt(self, prompt: str) -> str:
        """Hook to adapt prompts if needed. For now, keep as plain text."""
        return prompt


def _detect_project_root() -> Path:
    """Best-effort project root detection (looks for `data/` or `sql_database/`)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").exists() or (parent / "sql_database").exists():
            return parent
    return here.parent.parent


def create_embeddings(model_name: str):
    """Create embeddings using LangChain initializers."""
    from langchain.embeddings.base import init_embeddings

    if not model_name:
        raise ValueError("embedding model name is required")

    lower = model_name.lower()
    snowflake_aliases = {
        "snowflake",
        "arctic",
        "arctic-embed",
        "snowflake-arctic-embed-m-v2",
        "snowflake-arctic-embed-m-v2.0",
        "snowflake/snowflake-arctic-embed-m-v2.0",
    }
    if (
        lower in snowflake_aliases
        or lower.startswith("snowflake:")
        or lower.startswith("snowflake/")
    ):
        normalized = (
            "Snowflake/snowflake-arctic-embed-m-v2.0"
            if lower in snowflake_aliases
            else model_name.split(":", 1)[1]
            if lower.startswith("snowflake:")
            else model_name
        )
        return _create_snowflake_embeddings(normalized)

    selected_model = model_name
    if ":" not in selected_model:
        selected_model = f"openai:{selected_model}"

    if selected_model.startswith("fireworks:"):
        fireworks_model = selected_model.split(":", 1)[1]
        from langchain_fireworks import FireworksEmbeddings

        return FireworksEmbeddings(model=fireworks_model)

    return init_embeddings(selected_model)


def _create_snowflake_embeddings(model_name: str):
    """Snowflake Arctic embeddings adapter."""
    try:
        import torch
    except Exception:
        torch = None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError(
            "sentence-transformers is required for Snowflake embeddings. Install with: pip install sentence-transformers"
        ) from e

    has_cuda = bool(
        torch is not None
        and hasattr(torch, "cuda")
        and callable(getattr(torch.cuda, "is_available", None))
        and torch.cuda.is_available()
    )
    device = "cuda" if has_cuda else "cpu"

    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        config_kwargs={"use_memory_efficient_attention": False},
    )
    model = model.to(device)

    class SnowflakeEmbeddingsAdapter:
        """Minimal adapter exposing LangChain's embeddings interface."""

        def __init__(self, _model):
            self._model = _model

        def embed_documents(self, texts):
            if not texts:
                return []
            vecs = self._model.encode(
                texts,
                prompt_name="document",
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vecs.tolist()

        def embed_query(self, text):
            if not text:
                return []
            vec = self._model.encode(
                [text],
                prompt_name="query",
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vec[0].tolist()

    return SnowflakeEmbeddingsAdapter(model)


def create_reranker(model_name: str):
    """Create a reranker adapter for Together, Cohere, or Fireworks."""
    if not model_name:
        raise ValueError("reranker_model is required when reranking is enabled")

    if model_name.startswith("fireworks:"):
        fw_model = model_name.split(":", 1)[1].strip()
        import requests

        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError("Missing FIREWORKS_API_KEY env var for Fireworks reranking")

        url = "https://api.fireworks.ai/inference/v1/rerank"

        class FireworksRerankerAdapter:
            def __init__(self, model_str: str, api_key: str):
                self._model = model_str
                self._api_key = api_key

            def rerank(self, *, query: str, documents: list, top_n: int | None = None):
                payload = {
                    "model": self._model,
                    "query": query,
                    "documents": documents,
                    "return_documents": False,
                }
                if top_n is not None:
                    payload["top_n"] = int(top_n)

                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }

                r = requests.post(url, json=payload, headers=headers, timeout=60)
                r.raise_for_status()
                data = r.json()

                out = []
                for item in data.get("data", []):
                    out.append(
                        {
                            "index": int(item["index"]),
                            "score": float(item.get("relevance_score", 0.0)),
                        }
                    )
                out.sort(key=lambda x: x["score"], reverse=True)
                if top_n is not None:
                    out = out[: int(top_n)]
                return out

        return FireworksRerankerAdapter(fw_model, api_key)

    if model_name.startswith("cohere:"):
        cohere_model = model_name.split(":", 1)[1]
        from cohere import Client 

        client = Client()  # picks up COHERE_API_KEY

        class CohereRerankerAdapter:
            def __init__(self, client_inst, model_str):
                self._client = client_inst
                self._model = model_str

            def rerank(self, *, query: str, documents: list, top_n: int | None = None):
                resp = self._client.rerank(
                    model=self._model,
                    query=query,
                    documents=documents,
                    top_n=top_n,
                )
                results = sorted(
                    [
                        {"index": r.index, "score": getattr(r, "relevance_score", 0.0)}
                        for r in getattr(resp, "results", [])
                    ],
                    key=lambda x: x["score"],
                    reverse=True,
                )
                if top_n is not None:
                    results = results[: top_n]
                return results

        return CohereRerankerAdapter(client, cohere_model)

    together_model = model_name.split(":", 1)[1] if model_name.startswith("together:") else model_name
    from together import Together

    client = Together()

    class TogetherRerankerAdapter:
        def __init__(self, client_inst, model_str):
            self._client = client_inst
            self._model = model_str

        def rerank(self, *, query: str, documents: list, top_n: int | None = None):
            resp = self._client.rerank.create(
                model=self._model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=False,
            )
            results = sorted(
                [
                    {"index": r.index, "score": getattr(r, "relevance_score", 0.0)}
                    for r in getattr(resp, "results", [])
                ],
                key=lambda x: x["score"],
                reverse=True,
            )
            if top_n is not None:
                results = results[: top_n]
            return results

    return TogetherRerankerAdapter(client, together_model)