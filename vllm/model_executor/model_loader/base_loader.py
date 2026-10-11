# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import contextlib
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_processing
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            log_model_inspection(model)

            logger.debug("Loading weights on %s ...", load_device)
            with _impala_count_first_load(model):
                self.load_weights(model, model_config)

            # Log peak GPU memory after loading weights. This is needed
            # to have test coverage on peak memory for online quantization.
            if current_platform.is_cuda():
                peak_memory = torch.accelerator.max_memory_allocated()
                logger.debug_once(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                )

            # Process weights into kernel format. Note that when using online
            # quantization, weights are (typically) quantized as they are loaded.
            if _has_online_quant(model):
                finalize_layerwise_processing(model, model_config)

            process_weights_after_loading(model, model_config, target_device)

            _impala_weight_checksum(model)

        return model.eval()


def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from vllm.model_inspection import format_model_inspection

    logger.info("vLLM model structure:\n%s", format_model_inspection(model))


def _has_online_quant(model: nn.Module):
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if getattr(quant_method, "uses_meta_device", False):
            return True

    return False


_IMPALA_CHECKSUM_VERSION = 2


def _impala_h64(t: torch.Tensor) -> tuple[str, int]:
    """64-bit position-mixed hash of a CUDA tensor's bytes (device side), plus its byte count."""
    mask = (1 << 64) - 1
    chunk = 1 << 26
    u8 = t.detach().contiguous().view(-1).view(torch.uint8)
    nb = u8.numel()
    pad = (-nb) % 8
    if pad:
        u8 = torch.cat([u8, torch.zeros(pad, dtype=torch.uint8, device=u8.device)])
    x = u8.view(torch.int64)
    h = 0
    for off in range(0, x.numel(), chunk):
        c = x[off : off + chunk]
        idx = torch.arange(off, off + c.numel(), device=c.device, dtype=torch.int64)
        h = (h + int(((c * 0x5BD1E995) ^ (idx * 0x1B873593)).sum().item())) & mask
    return f"{h:016x}", nb


def _impala_derived_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Derived (non-parameter, non-buffer) CUDA tensors reachable from the modules: plain tensor attributes, fused-MoE quant
    config descriptors, and the experts objects' tensor attributes. These are what captured CUDA graphs read besides params."""
    out: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()

    def add(name: str, t) -> None:
        if isinstance(t, torch.Tensor) and t.device.type == "cuda" and t.numel() > 0 and id(t) not in seen:
            seen.add(id(t))
            out.append((name, t))

    for mod_name, mod in model.named_modules():
        pb = set(mod._parameters.keys()) | set(mod._buffers.keys())
        for k, v in list(vars(mod).items()):
            if k in pb or k == "kv_cache":  # kv_cache: the attention layers' KV tensors (bound after load), not weights
                continue
            add(f"{mod_name}.<attr>{k}", v)
        qm = getattr(mod, "quant_method", None)
        qc = getattr(qm, "moe_quant_config", None)
        if qc is not None:
            for desc_name in ("_a1", "_a2", "_w1", "_w2"):
                d = getattr(qc, desc_name, None)
                if d is None:
                    continue
                for f, v in list(vars(d).items()):
                    add(f"{mod_name}.<qc>{desc_name}.{f}", v)
        kern = getattr(qm, "moe_kernel", None)
        ex = getattr(kern, "fused_experts", None)
        if ex is not None:
            for f, v in list(vars(ex).items()):
                add(f"{mod_name}.<experts>{f}", v)
    out.sort(key=lambda kv: kv[0])
    return out


def _impala_weight_checksum(model: nn.Module, tag: str = "load") -> str | None:
    """Experiment 4/5 (Impala): IMPALA_WEIGHT_CHECKSUM=<dir> -> write a per-rank report hashing every GPU parameter/buffer
    (digest `all_tensors_sha256`, Exp-4 definition) and every derived CUDA tensor (digest `derived_sha256`), with data_ptr per
    tensor, plus sha256 of a fixed sample. Returns the report path (None when the hook is inert)."""
    import hashlib
    import json
    import os
    import time

    out_dir = os.environ.get("IMPALA_WEIGHT_CHECKSUM")
    if not out_dir:
        return None
    from vllm.distributed import get_tensor_model_parallel_rank

    t0 = time.time()
    rank = get_tensor_model_parallel_rank()
    items = [(n, p.data) for n, p in model.named_parameters()]
    items += [(n, b) for n, b in model.named_buffers()]
    items.sort(key=lambda kv: kv[0])
    res: dict[str, dict] = {}
    total = 0
    for name, t in items:
        entry = {"shape": list(t.shape), "dtype": str(t.dtype), "dev": str(t.device)}
        if t.device.type == "cuda" and t.numel() > 0:
            h, nb = _impala_h64(t)
            entry["nbytes"] = nb
            entry["h64"] = h
            entry["ptr"] = t.data_ptr()
            total += nb
        res[name] = entry
    names = [n for n, t in items if t.device.type == "cuda" and t.numel() > 0]
    k = max(1, len(names) // 20)
    chosen = list(dict.fromkeys(names[::k][:20] + [n for n in names if "scale" in n][:6]))
    by_name = dict(items)
    sample = {}
    for n in chosen:
        t = by_name[n]
        sample[n] = hashlib.sha256(
            t.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
    # digest over h64/shape/dtype only (ptr excluded) so it is comparable across processes
    digest_src = {n: {k2: v for k2, v in e.items() if k2 != "ptr"} for n, e in res.items()}
    digest = hashlib.sha256(json.dumps(digest_src, sort_keys=True).encode()).hexdigest()
    derived: dict[str, dict] = {}
    dtotal = 0
    for name, t in _impala_derived_tensors(model):
        h, nb = _impala_h64(t)
        derived[name] = {"shape": list(t.shape), "dtype": str(t.dtype), "nbytes": nb, "h64": h, "ptr": t.data_ptr()}
        dtotal += nb
    dsrc = {n: {k2: v for k2, v in e.items() if k2 != "ptr"} for n, e in derived.items()}
    ddigest = hashlib.sha256(json.dumps(dsrc, sort_keys=True).encode()).hexdigest()
    report = {
        "version": _IMPALA_CHECKSUM_VERSION,
        "tag": tag,
        "rank": rank,
        "pid": os.getpid(),
        "n_tensors": len(res),
        "n_hashed": len(names),
        "gpu_bytes": total,
        "all_tensors_sha256": digest,
        "n_derived": len(derived),
        "derived_bytes": dtotal,
        "derived_sha256": ddigest,
        "sample_sha256": sample,
        "tensors": res,
        "derived": derived,
        "elapsed_s": round(time.time() - t0, 1),
        "pinned_loader": os.environ.get("IMPALA_PINNED_LOADER", "0"),
        "reload_fix": os.environ.get("IMPALA_RELOAD_FIX", "0"),
    }
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if tag == "load" else f"-{tag}"
    path = os.path.join(out_dir, f"wchk-rank{rank}-{os.getpid()}{suffix}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(report, f, indent=0)
    os.replace(path + ".tmp", path)
    logger.info(
        "[impala-checksum] v2 tag=%s rank %d: %d tensors / %.1f GiB hashed, all-tensors %s, derived %d / %.1f MiB %s (%.1f s) -> %s",
        tag, rank, len(names), total / 1024**3, digest[:16], len(derived), dtotal / 1024**2, ddigest[:16],
        time.time() - t0, path,
    )
    return path


@contextlib.contextmanager
def _impala_count_first_load(model: nn.Module):
    """Experiment 5 (Impala): count every layer's loaded elements during the first load (IMPALA_RELOAD_FIX=1) and store the
    measured total as layer._impala_first_load_numel, the exact trigger total for a later layerwise reload."""
    import os as _os

    if _os.environ.get("IMPALA_RELOAD_FIX", "0") != "1":
        yield
        return
    import inspect

    from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS, get_numel_loaded
    from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    counts: dict[int, int] = {}
    wrapped: list[tuple[torch.Tensor, object]] = []

    def make(layer, tensor):
        original = getattr(tensor, "weight_loader", default_weight_loader)
        sig = inspect.signature(original)

        def counting_loader(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            n, ret = get_numel_loaded(original, bound)
            counts[id(layer)] = counts.get(id(layer), 0) + n
            return ret

        return original, counting_loader

    layers = list(model.modules())
    for layer in layers:
        for name, tensor in get_layer_tensors(layer).items():
            if name in SKIP_TENSORS:
                continue
            original, cl = make(layer, tensor)
            tensor.weight_loader = cl
            wrapped.append((tensor, original))
    try:
        yield
    finally:
        for tensor, original in wrapped:
            try:
                if original is default_weight_loader and "weight_loader" not in tensor.__dict__:
                    continue
                tensor.weight_loader = original
            except Exception:  # noqa: BLE001
                pass
        n_layers = 0
        for layer in layers:
            c = counts.get(id(layer), 0)
            if c > 0:
                layer._impala_first_load_numel = c
                n_layers += 1
        logger.info("[impala-reload] first-load totals recorded for %d layers", n_layers)
