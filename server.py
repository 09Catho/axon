"""AXON backend.

Loads gpt2-small + Joseph Bloom's pretrained SAE at layer 8 (residual stream),
streams per-token top-K SAE feature activations over a WebSocket to the browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import requests
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sae_lens import SAE, HookedSAETransformer


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_PATH = Path(__file__).parent / "axon.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("axon")


# ---------------------------------------------------------------------------
# Model / SAE
# ---------------------------------------------------------------------------

SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.8.hook_resid_pre"
LAYER = 8
HOOK_NAME = "blocks.8.hook_resid_pre"
TOP_K = 15
ACT_THRESHOLD = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Using device: %s", device)

log.info("Loading SAE %s / %s ...", SAE_RELEASE, SAE_ID)
sae_result = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
# sae_lens API has varied; normalize to (sae, cfg_dict, sparsity)
if isinstance(sae_result, tuple):
    sae = sae_result[0]
    sae_cfg_dict = sae_result[1] if len(sae_result) > 1 else {}
else:
    sae = sae_result
    sae_cfg_dict = {}

model_kwargs = {}
if isinstance(sae_cfg_dict, dict):
    model_kwargs = dict(sae_cfg_dict.get("model_from_pretrained_kwargs") or {})

log.info("Loading HookedSAETransformer gpt2-small (no_processing, kwargs=%s) ...", model_kwargs)
model = HookedSAETransformer.from_pretrained_no_processing(
    "gpt2-small", device=device, **model_kwargs
)
model.eval()

# Cast SAE to a sensible dtype and resolve target dtype/device once.
sae = sae.to(device)
SAE_DEVICE = next(sae.parameters()).device
SAE_DTYPE = next(sae.parameters()).dtype
log.info("SAE on %s dtype %s", SAE_DEVICE, SAE_DTYPE)

ready = True
log.info("Models loaded successfully.")


# ---------------------------------------------------------------------------
# Feature label cache (persistent)
# ---------------------------------------------------------------------------

CACHE_PATH = Path(__file__).parent / "feature_cache.json"
NEURONPEDIA_URL = "https://www.neuronpedia.org/api/feature/gpt2-small/8-res-jb/{idx}"

if CACHE_PATH.exists():
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            feature_cache: dict[str, str] = json.load(f)
        log.info("Loaded %d cached feature labels.", len(feature_cache))
    except Exception:
        feature_cache = {}
else:
    feature_cache = {}

_cache_dirty_count = 0


def _persist_cache() -> None:
    try:
        tmp = CACHE_PATH.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(feature_cache, f)
        os.replace(tmp, CACHE_PATH)
    except Exception:
        log.exception("Failed to persist feature cache.")


def _fetch_label_sync(idx: int) -> str:
    """Blocking HTTP fetch — run inside an executor."""
    key = str(idx)
    if key in feature_cache:
        return feature_cache[key]
    label = f"Feature {idx}"
    try:
        res = requests.get(NEURONPEDIA_URL.format(idx=idx), timeout=3)
        if res.status_code == 200:
            data = res.json()
            for exp in data.get("explanations") or []:
                desc = exp.get("description")
                if desc:
                    label = desc.strip()
                    break
    except Exception as exc:  # noqa: BLE001 — log and fall back gracefully
        log.debug("Neuronpedia fetch failed for %d: %s", idx, exc)
    feature_cache[key] = label
    return label


async def get_labels(indices: list[int]) -> list[str]:
    """Resolve labels for a list of feature indices in parallel."""
    global _cache_dirty_count
    loop = asyncio.get_running_loop()
    # Split cached vs uncached so cached lookups are instant.
    todo: list[tuple[int, int]] = []  # (position, idx)
    out: list[Optional[str]] = [None] * len(indices)
    for i, idx in enumerate(indices):
        cached = feature_cache.get(str(idx))
        if cached is not None:
            out[i] = cached
        else:
            todo.append((i, idx))

    if todo:
        results = await asyncio.gather(
            *(loop.run_in_executor(None, _fetch_label_sync, idx) for _, idx in todo)
        )
        for (i, _), label in zip(todo, results):
            out[i] = label
        _cache_dirty_count += len(todo)
        if _cache_dirty_count >= 25:
            _persist_cache()
            _cache_dirty_count = 0

    return [s or "" for s in out]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AXON")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return RedirectResponse("/static/index.html")


@app.get("/health")
def health():
    return {"ready": ready, "device": device, "cached_labels": len(feature_cache)}


@app.on_event("shutdown")
def _on_shutdown():
    _persist_cache()


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def step_once(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one forward pass; return (logits[:, -1, :], last_token_residual)."""
    captured: dict[str, torch.Tensor] = {}

    def hook(activation: torch.Tensor, hook):  # noqa: ARG001
        # activation: [batch, pos, d_model]
        captured["resid"] = activation[:, -1, :].detach()

    logits = model.run_with_hooks(
        tokens,
        return_type="logits",
        fwd_hooks=[(HOOK_NAME, hook)],
    )
    return logits[:, -1, :], captured["resid"][0]  # [d_model]


@torch.no_grad()
def sae_topk(resid: torch.Tensor, k: int = TOP_K) -> tuple[list[float], list[int]]:
    x = resid.to(device=SAE_DEVICE, dtype=SAE_DTYPE)
    feats = sae.encode(x)
    if feats.dim() == 2:
        feats = feats[0]
    k = min(k, feats.shape[0])
    top_vals, top_idx = torch.topk(feats, k=k)
    return top_vals.detach().cpu().tolist(), top_idx.detach().cpu().tolist()


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    past_ids: Optional[torch.Tensor] = None,
) -> int:
    """Sample (or greedily decode) the next token from logits."""
    logits = logits.clone().float()

    # Repetition penalty — reduce probability of already-seen tokens
    if repetition_penalty != 1.0 and past_ids is not None:
        for tid in past_ids[0].tolist():
            if logits[tid] < 0:
                logits[tid] *= repetition_penalty
            else:
                logits[tid] /= repetition_penalty

    # Greedy when temperature is effectively 0
    if temperature < 1e-4:
        return int(logits.argmax().item())

    logits = logits / temperature

    # Top-K filter
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k).values[-1]
        logits[logits < threshold] = float("-inf")

    # Top-P (nucleus) filter
    if top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens whose cumulative prob exceeds top_p
        remove = (cumulative - sorted_probs) > top_p
        sorted_probs[remove] = 0.0
        probs.scatter_(0, sorted_idx, sorted_probs)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return int(torch.multinomial(probs, num_samples=1).item())

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid_json"}))
                continue

            prompt = (req.get("prompt") or "").strip()
            max_tokens = int(req.get("max_tokens", 40))
            max_tokens = max(1, min(max_tokens, 500))
            temperature = float(req.get("temperature", 0.8))
            top_k = int(req.get("top_k", 50))
            top_p = float(req.get("top_p", 0.9))
            rep_penalty = float(req.get("repetition_penalty", 1.1))
            act_threshold = float(req.get("act_threshold", ACT_THRESHOLD))
            feat_k = int(req.get("feat_k", TOP_K))

            if not prompt:
                await websocket.send_text(json.dumps({"error": "empty_prompt"}))
                await websocket.send_text(json.dumps({"status": "done"}))
                continue

            try:
                tokens = model.to_tokens(prompt).to(device)
                for step in range(max_tokens):
                    last_logits, resid = step_once(tokens)
                    top_acts, top_indices = sae_topk(resid, k=feat_k)

                    labels = await get_labels(top_indices)
                    features = [
                        {
                            "id": int(idx),
                            "label": label,
                            "activation": float(act),
                            "layer": LAYER,
                        }
                        for idx, act, label in zip(top_indices, top_acts, labels)
                        if act > act_threshold
                    ]

                    next_id = sample_token(
                        last_logits[0],
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=rep_penalty,
                        past_ids=tokens,
                    )
                    next_str = model.tokenizer.decode([next_id])
                    tokens = torch.cat(
                        [tokens, torch.tensor([[next_id]], device=tokens.device)],
                        dim=-1,
                    )

                    await websocket.send_text(
                        json.dumps(
                            {
                                "token": next_str,
                                "step": step,
                                "features": features,
                            }
                        )
                    )
                    # yield control so the browser can render between frames
                    await asyncio.sleep(0)

                await websocket.send_text(json.dumps({"status": "done"}))
            except WebSocketDisconnect:
                # Client navigated away mid-generation — not an error, just stop.
                raise
            except Exception:
                log.exception("Generation error")
                # Only try to send if the socket is still open.
                try:
                    await websocket.send_text(json.dumps({"error": "generation_failed"}))
                    await websocket.send_text(json.dumps({"status": "done"}))
                except Exception:
                    pass

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected.")
    except Exception:
        log.exception("WebSocket loop crashed.")
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
