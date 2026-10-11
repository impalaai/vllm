# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

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


def _impala_weight_checksum(model: nn.Module) -> None:
    """Experiment 4 (Impala): IMPALA_WEIGHT_CHECKSUM=<dir> -> write a per-rank report hashing every GPU parameter/buffer
    (64-bit position-mixed sum on the device) plus sha256 of a fixed sample, after process_weights_after_loading."""
    import hashlib
    import json
    import os
    import time

    out_dir = os.environ.get("IMPALA_WEIGHT_CHECKSUM")
    if not out_dir:
        return
    from vllm.distributed import get_tensor_model_parallel_rank

    t0 = time.time()
    rank = get_tensor_model_parallel_rank()
    items = [(n, p.data) for n, p in model.named_parameters()]
    items += [(n, b) for n, b in model.named_buffers()]
    items.sort(key=lambda kv: kv[0])
    mask = (1 << 64) - 1
    chunk = 1 << 26
    res: dict[str, dict] = {}
    total = 0
    for name, t in items:
        entry = {"shape": list(t.shape), "dtype": str(t.dtype), "dev": str(t.device)}
        if t.device.type == "cuda" and t.numel() > 0:
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
            entry["nbytes"] = nb
            entry["h64"] = f"{h:016x}"
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
    digest = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    report = {
        "rank": rank,
        "pid": os.getpid(),
        "n_tensors": len(res),
        "n_hashed": len(names),
        "gpu_bytes": total,
        "all_tensors_sha256": digest,
        "sample_sha256": sample,
        "tensors": res,
        "elapsed_s": round(time.time() - t0, 1),
        "pinned_loader": os.environ.get("IMPALA_PINNED_LOADER", "0"),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"wchk-rank{rank}-{os.getpid()}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(report, f, indent=0)
    os.replace(path + ".tmp", path)
    logger.info(
        "[impala-checksum] rank %d: %d tensors, %.1f GiB hashed, all-tensors sha256 %s (%.1f s) -> %s",
        rank, len(names), total / 1024**3, digest[:16], time.time() - t0, path,
    )
