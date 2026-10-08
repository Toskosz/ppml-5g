"""FHE GPU/CUDA device selection helpers for Concrete ML."""

from __future__ import annotations

import os
import sys


DEVICE_ENV = "PPML_FHE_DEVICE"  # auto | cpu | cuda


def _default_log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def probe_gpu(log_fn=None) -> dict:
    """Return Concrete compiler GPU capability flags."""
    log = log_fn or _default_log
    info = {
        "gpu_available": False,
        "gpu_enabled": False,
        "error": None,
    }
    try:
        import concrete.compiler as cc

        info["gpu_available"] = bool(cc.check_gpu_available())
        info["gpu_enabled"] = bool(cc.check_gpu_enabled())
    except Exception as exc:  # pragma: no cover - import/runtime edge cases
        info["error"] = str(exc)
        log(f"GPU probe failed: {exc}")
    return info


def resolve_fhe_device(requested: str | None = None, log_fn=None) -> str:
    """
    Resolve FHE compilation device to 'cpu' or 'cuda'.

    Priority:
      1. explicit `requested` argument
      2. PPML_FHE_DEVICE env var
      3. 'auto' (cuda if GPU wheel + GPU present, else cpu)
    """
    log = log_fn or _default_log
    choice = (requested or os.environ.get(DEVICE_ENV) or "auto").strip().lower()
    if choice not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"Invalid FHE device '{choice}'. Expected one of: auto, cpu, cuda."
        )

    info = probe_gpu(log_fn=log)
    log(
        f"GPU status — available={info['gpu_available']}, "
        f"enabled(runtime)={info['gpu_enabled']}"
        + (f", probe_error={info['error']}" if info["error"] else "")
    )

    if choice == "cpu":
        log("FHE device: cpu (requested)")
        return "cpu"

    if choice == "cuda":
        if not info["gpu_available"]:
            raise RuntimeError(
                "FHE device='cuda' was requested, but no CUDA GPU is available / "
                "the GPU-enabled concrete-python wheel is not installed.\n"
                "Install with:\n"
                "  pip uninstall concrete-python\n"
                "  pip install --extra-index-url https://pypi.zama.ai/gpu "
                "concrete-python==2.10.0\n"
                "Or set PPML_FHE_DEVICE=cpu / pass device='cpu'."
            )
        if not info["gpu_enabled"]:
            log(
                "WARNING: GPU hardware looks available but the Concrete runtime "
                "reports GPU disabled. You likely have the CPU wheel installed. "
                "Reinstall the GPU wheel from https://pypi.zama.ai/gpu"
            )
            raise RuntimeError(
                "FHE device='cuda' requested but concrete.compiler.check_gpu_enabled() "
                "is False. Install the GPU-enabled concrete-python wheel."
            )
        log("FHE device: cuda (requested)")
        return "cuda"

    # auto
    if info["gpu_available"] and info["gpu_enabled"]:
        log("FHE device: cuda (auto — GPU available and runtime enabled)")
        return "cuda"

    if info["gpu_available"] and not info["gpu_enabled"]:
        log(
            "FHE device: cpu (auto — GPU present but Concrete GPU runtime disabled; "
            "install GPU wheel from https://pypi.zama.ai/gpu to enable)"
        )
    else:
        log("FHE device: cpu (auto — no CUDA GPU / GPU runtime)")
    return "cpu"


def compile_for_device(classifier, X, device="auto", log_fn=None, fallback_cpu=True, **compile_kwargs):
    """
    Compile a Concrete ML model for cpu or cuda.

    On GPU parameter-selection failures (common with constrained GPU crypto params),
    optionally falls back to CPU compilation when device resolution was 'auto' or
    fallback_cpu=True for explicit cuda requests only if PPML_FHE_GPU_FALLBACK=1.
    """
    log = log_fn or _default_log
    requested = (device or os.environ.get(DEVICE_ENV) or "auto").strip().lower()
    resolved = resolve_fhe_device(requested, log_fn=log)

    allow_fallback = fallback_cpu and (
        requested == "auto"
        or os.environ.get("PPML_FHE_GPU_FALLBACK", "0") == "1"
    )

    log(f"Compiling FHE circuit with device='{resolved}'...")
    try:
        circuit = classifier.compile(X, device=resolved, **compile_kwargs)
        log(f"FHE compilation succeeded on device='{resolved}'.")
        return circuit, resolved
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        gpu_param_issue = resolved == "cuda" and (
            "NoParametersFound" in msg
            or "no parameters" in msg.lower()
            or "Unfeasible noise constraint" in msg
            or "noise constraint" in msg.lower()
        )
        if gpu_param_issue and allow_fallback:
            log(
                f"WARNING: CUDA compilation failed ({msg}). "
                "Falling back to device='cpu'."
            )
            circuit = classifier.compile(X, device="cpu", **compile_kwargs)
            log("FHE compilation succeeded on device='cpu' (fallback).")
            return circuit, "cpu"
        raise


def write_device_marker(model_dir: str, device: str) -> None:
    """Record which device a saved FHE model was compiled for."""
    os.makedirs(model_dir, exist_ok=True)
    marker = os.path.join(model_dir, "fhe_device.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write(device.strip().lower() + "\n")


def project_sparse_svd(svd, X_sparse, log_fn=None, chunk_size=50_000):
    """Project a wide sparse matrix with a fitted TruncatedSVD.

    One float32 buffer is filled in place. The previous path kept every
    chunk and then np.vstack'd them, so the 200-component training matrix
    (~10 GB) was allocated twice and the kernel sent SIGKILL.
    """
    import numpy as np

    n_rows = int(X_sparse.shape[0])
    n_components = int(svd.components_.shape[0])
    components_t = np.ascontiguousarray(svd.components_.T, dtype=np.float32)
    out = np.empty((n_rows, n_components), dtype=np.float32)
    if log_fn is not None:
        log_fn(
            f"Projecting {n_rows:,} x {X_sparse.shape[1]:,} -> {n_components} components "
            f"in chunks of {chunk_size:,}..."
        )
    next_log = 0
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        block = X_sparse[start:end]
        if block.dtype != np.float32:
            block = block.astype(np.float32)
        out[start:end] = np.asarray(block @ components_t, dtype=np.float32)
        del block
        if log_fn is not None and end >= next_log:
            log_fn(f"Projected rows {end:,} / {n_rows:,}")
            next_log = end + 1_000_000
    del components_t
    release_memory()
    return out


def release_memory() -> None:
    """Return freed arrays to the OS before the next forest is trained.

    Large numpy blocks are released on free. malloc_trim covers the smaller
    glibc heap that sklearn and Concrete ML leave behind, which otherwise
    keeps RSS high enough for the OOM killer to send SIGKILL.
    """
    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
