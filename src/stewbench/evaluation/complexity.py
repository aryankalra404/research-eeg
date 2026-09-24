"""Model cost: parameters, FLOPs per window, latency and peak memory."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict:
    return {"parameters": int(sum(p.numel() for p in model.parameters())),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}


def _recurrent_flops(module: nn.RNNBase, seq_len: int) -> int:
    gates = {"RNN_TANH": 1, "RNN_RELU": 1, "LSTM": 4, "GRU": 3}[module.mode]
    directions = 2 if module.bidirectional else 1
    total, input_size = 0, module.input_size
    for _ in range(module.num_layers):
        macs = gates * module.hidden_size * (input_size + module.hidden_size) * seq_len
        total += 2 * macs * directions
        input_size = module.hidden_size * directions
    return total


def flops_per_window(model: nn.Module, input_shape: tuple[int, int]) -> int | None:
    """Forward FLOPs for one window (multiply-add = 2 FLOPs) measured with
    torch's FlopCounterMode; recurrent layers are added analytically because
    the counter does not cover cuDNN/native RNN kernels."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:  # pragma: no cover
        return None
    model = model.eval().cpu()
    x = torch.zeros(1, *input_shape)
    rnn_modules = {name: m for name, m in model.named_modules() if isinstance(m, nn.RNNBase)}
    analytic = [0]
    handles = [m.register_forward_hook(
        lambda mod, inp, out: analytic.__setitem__(0, analytic[0] + _recurrent_flops(mod, inp[0].shape[1])))
        for m in rnn_modules.values()]
    # The fused inference fast path of nn.TransformerEncoderLayer / MHA is
    # invisible to the counter; disable it while counting.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(x)
        total = counter.get_total_flops()
        # Replace whatever the counter attributed to recurrent layers (backend
        # dependent: sometimes all, sometimes nothing) by the analytic count.
        counts = counter.get_flop_counts()
        prefix = type(model).__name__
        for name in rnn_modules:
            key = f"{prefix}.{name}"
            if key in counts:
                total -= sum(counts[key].values())
        return int(total + analytic[0])
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        for h in handles:
            h.remove()


def latency_ms(model: nn.Module, input_shape: tuple[int, int], device, batch_size: int = 1,
               warmup: int = 10, repeats: int = 50) -> float:
    """Median wall-clock forward latency per batch (ms)."""
    model = model.eval().to(device)
    x = torch.randn(batch_size, *input_shape, device=device)
    timings = []
    with torch.no_grad():
        for i in range(warmup + repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if i >= warmup:
                timings.append((time.perf_counter() - t0) * 1000)
    return float(np.median(timings))


def profile(model: nn.Module, input_shape: tuple[int, int], device) -> dict:
    info = count_parameters(model)
    info["flops_per_window"] = flops_per_window(model, input_shape)
    info["latency_ms_cpu_batch1"] = latency_ms(model, input_shape, torch.device("cpu"), repeats=30)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        info["latency_ms_gpu_batch1"] = latency_ms(model, input_shape, device)
        info["latency_ms_gpu_batch256_per_window"] = latency_ms(model, input_shape, device, batch_size=256) / 256
        info["peak_gpu_memory_mb_inference"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
    model.cpu()
    return info
