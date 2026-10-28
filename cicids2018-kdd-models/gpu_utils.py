"""FHE GPU/CUDA device selection helpers for Concrete ML."""

from __future__ import annotations

import json
import os
import shutil
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


def read_fhe_device(model_dir: str) -> str:
    """Return the device recorded by write_device_marker."""
    marker = os.path.join(model_dir.rstrip("/"), "fhe_device.txt")
    with open(marker, encoding="utf-8") as f:
        return f.read().strip() or "unknown"


def _runtime_device(device: str) -> str:
    device = (device or "").strip().lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError(
            f"Invalid compiled-circuit device '{device}'. Expected 'cpu' or 'cuda'."
        )
    return device


def device_circuit_dir(model_dir: str, device: str) -> str:
    """``<model_dir>/cpu`` or ``<model_dir>/cuda``."""
    device = _runtime_device(device)
    return os.path.join(model_dir.rstrip("/"), device)


def fhe_circuit_dirs(model_dir: str) -> tuple[str, str, str]:
    """Legacy flat directory, then the cpu and cuda folders."""
    base = model_dir.rstrip("/")
    return (
        base,
        device_circuit_dir(base, "cpu"),
        device_circuit_dir(base, "cuda"),
    )


def migrate_legacy_circuit(model_dir: str, log_fn=None) -> str | None:
    """Move a flat circuit directory into ``<model_dir>/<device>/``.

    Older runs stored the circuit files directly in the model directory.
    Returns the device folder, or None when the directory is not a flat circuit.
    """
    base = model_dir.rstrip("/")
    if not fhe_assets_ready(base):
        return None
    device = read_fhe_device(base)
    if device not in {"cpu", "cuda"}:
        return None
    dest = device_circuit_dir(base, device)
    if fhe_assets_ready(dest):
        return dest
    if os.path.lexists(dest):
        return None
    staging = base + ".layout"
    if os.path.lexists(staging):
        return None
    if log_fn is not None:
        log_fn(f"Moving FHE circuit '{base}' to '{dest}' (device='{device}').")
    os.rename(base, staging)
    try:
        os.makedirs(base)
        os.rename(staging, dest)
    except Exception:
        if os.path.isdir(base) and not os.listdir(base):
            os.rmdir(base)
        if os.path.isdir(staging) and not os.path.isdir(base):
            os.rename(staging, base)
        raise
    return dest


def compiled_circuit_dir(model_dir: str, device: str, log_fn=None) -> str | None:
    """Return ``<model_dir>/<device>`` when a finished circuit for `device` exists.

    A flat circuit from an older run is moved into that device folder first.
    """
    device = _runtime_device(device)
    base = model_dir.rstrip("/")
    migrate_legacy_circuit(base, log_fn=log_fn)
    specific = device_circuit_dir(base, device)
    if fhe_assets_ready(specific) and read_fhe_device(specific) == device:
        return specific
    if fhe_assets_ready(base) and read_fhe_device(base) == device:
        return base
    return None


def circuit_save_dir(model_dir: str, device: str, log_fn=None) -> str:
    """Folder where a circuit for `device` is written: ``<model_dir>/<device>``."""
    device = _runtime_device(device)
    base = model_dir.rstrip("/")
    migrate_legacy_circuit(base, log_fn=log_fn)
    if fhe_assets_ready(base):
        saved = read_fhe_device(base)
        raise RuntimeError(
            f"FHE circuit '{base}' is still in the flat layout "
            f"(device='{saved}') and could not be moved into "
            f"'{device_circuit_dir(base, saved)}'."
        )
    return device_circuit_dir(base, device)


def any_fhe_assets(model_dir: str) -> bool:
    """True when a finished circuit exists for either runtime device."""
    return any(fhe_assets_ready(path) for path in fhe_circuit_dirs(model_dir))


def circuit_reuse_status(model_dir: str, device: str, log_fn=None) -> tuple[str | None, str | None]:
    """Return ``(matched_dir, None)`` or ``(None, mismatch_reason)``.

    ``mismatch_reason`` is None when no circuit is saved. A reason means a
    saved circuit was compiled for a different device.
    """
    device = _runtime_device(device)
    matched = compiled_circuit_dir(model_dir, device, log_fn=log_fn)
    if matched is not None:
        return matched, None
    others = []
    for path in fhe_circuit_dirs(model_dir):
        if not fhe_assets_ready(path):
            continue
        saved = read_fhe_device(path)
        if saved == device:
            continue
        others.append(f"'{path}' (device='{saved}')")
    if not others:
        return None, None
    listed = ", ".join(others)
    noun = "circuit" if len(others) == 1 else "circuits"
    verb = "does" if len(others) == 1 else "do"
    return None, (
        f"Saved FHE {noun} {listed} {verb} not match device='{device}'."
    )


FHE_ASSET_NAMES = ("client.zip", "server.zip", "fhe_device.txt")


def _nonempty_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def fhe_assets_ready(model_dir: str) -> bool:
    """True when a previous save finished (circuit zips plus device marker)."""
    root = model_dir.rstrip("/")
    return all(_nonempty_file(os.path.join(root, name)) for name in FHE_ASSET_NAMES)


def recover_interrupted_fhe_save(model_dir: str) -> str | None:
    """Finish or roll back a save that died while swapping directories.

    Returns ``promoted-tmp``, ``restored-old``, ``kept``, or None when there
    was nothing to repair.
    """
    model_dir = model_dir.rstrip("/")
    tmp_dir = model_dir + ".tmp"
    old_dir = model_dir + ".old"
    # A finished temp dir is a newer save that died before the swap.
    if fhe_assets_ready(tmp_dir):
        if os.path.isdir(model_dir):
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir)
            os.rename(model_dir, old_dir)
        os.rename(tmp_dir, model_dir)
        if os.path.isdir(old_dir):
            shutil.rmtree(old_dir)
        return "promoted-tmp"
    if fhe_assets_ready(model_dir):
        removed = False
        for leftover in (tmp_dir, old_dir):
            if os.path.isdir(leftover):
                shutil.rmtree(leftover)
                removed = True
        return "kept" if removed else None
    if fhe_assets_ready(old_dir):
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        os.rename(old_dir, model_dir)
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        return "restored-old"
    return None


def save_fhe_model(model_dir: str, classifier, device: str, meta: dict | None = None) -> None:
    """Save a compiled circuit, replacing any previous directory.

    Concrete ML refuses to save into a non-empty directory. The circuit is
    written to a sibling temp directory and swapped into place only after
    ``client.zip``, ``server.zip``, and ``fhe_device.txt`` exist, so a rerun
    neither crashes nor deletes a good model halfway through the write.
    """
    from concrete.ml.deployment import FHEModelDev

    model_dir = model_dir.rstrip("/")
    parent = os.path.dirname(os.path.abspath(model_dir)) or "."
    os.makedirs(parent, exist_ok=True)
    recover_interrupted_fhe_save(model_dir)
    tmp_dir = model_dir + ".tmp"
    old_dir = model_dir + ".old"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    dev = FHEModelDev(tmp_dir, classifier)
    try:
        dev.save()
        write_device_marker(tmp_dir, device)
        if meta is not None:
            with open(os.path.join(tmp_dir, "run_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
        if not fhe_assets_ready(tmp_dir):
            raise RuntimeError(f"FHE save to '{tmp_dir}' did not produce the expected files.")
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        del dev
    if os.path.isdir(old_dir):
        shutil.rmtree(old_dir)
    if os.path.isdir(model_dir):
        os.rename(model_dir, old_dir)
    os.rename(tmp_dir, model_dir)
    if os.path.isdir(old_dir):
        shutil.rmtree(old_dir)


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


def save_projection(svd, X_sparse, path, log_fn=None):
    """Project *X_sparse* and write one float32 ``.npy``, then free the buffer.

    Callers keep the sparse one-hot matrix only while projections are written,
    then delete it before training and FHE compilation.
    """
    import numpy as np

    projected = project_sparse_svd(svd, X_sparse, log_fn=log_fn)
    if log_fn is not None:
        log_fn(
            f"Writing {projected.shape} float32 projection "
            f"({projected.nbytes / 1e9:.2f} GB) to '{path}'..."
        )
    np.save(path, projected)
    del projected
    release_memory()


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


# Host RAM ceiling for a training / FHE run. GPU device memory is separate.
# Override with PPML_MEMORY_LIMIT_GB=0 to disable, or pass --memory-gb.
MEMORY_LIMIT_ENV = "PPML_MEMORY_LIMIT_GB"
DEFAULT_MEMORY_LIMIT_GB = 20
_MEM_CGROUP_NAME = "ppml-mem"


def memory_limit_gb_from_env() -> float | None:
    """Return the configured RAM cap in GiB, or None when the cap is disabled."""
    raw = os.environ.get(MEMORY_LIMIT_ENV, str(DEFAULT_MEMORY_LIMIT_GB)).strip().lower()
    if raw in {"", "0", "none", "off", "unlimited"}:
        return None
    gb = float(raw)
    if gb <= 0:
        return None
    return gb


def _self_cgroup_dir() -> str | None:
    """Absolute cgroup v2 directory for this process."""
    try:
        text = open("/proc/self/cgroup", encoding="utf-8").read()
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            rel = parts[2] if parts[2].startswith("/") else "/" + parts[2]
            return "/sys/fs/cgroup" if rel == "/" else "/sys/fs/cgroup" + rel
    return None


def _read_memory_max(cgroup_dir: str) -> int | None:
    raw = open(os.path.join(cgroup_dir, "memory.max"), encoding="utf-8").read().strip()
    if raw == "max":
        return None
    return int(raw)


def _cgroup_procs(cgroup_dir: str) -> list[int]:
    procs = []
    with open(os.path.join(cgroup_dir, "cgroup.procs"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                procs.append(int(line))
    return procs


def _move_pid(cgroup_dir: str, pid: int) -> None:
    with open(os.path.join(cgroup_dir, "cgroup.procs"), "w", encoding="utf-8") as f:
        f.write(str(pid))


def _write_memory_limit(cgroup_dir: str, nbytes: int) -> None:
    with open(os.path.join(cgroup_dir, "memory.max"), "w", encoding="utf-8") as f:
        f.write(str(nbytes))
    high_path = os.path.join(cgroup_dir, "memory.high")
    try:
        with open(high_path, "w", encoding="utf-8") as f:
            f.write(str(nbytes))
    except OSError:
        pass


def _enable_memory_controller(current: str, log) -> bool:
    """Delegate the memory controller when this process is alone in `current`."""
    del log
    controllers_path = os.path.join(current, "cgroup.controllers")
    controllers = set(open(controllers_path, encoding="utf-8").read().split())
    if "memory" not in controllers:
        return False
    others = [pid for pid in _cgroup_procs(current) if pid != os.getpid()]
    if others:
        return False
    hold = os.path.join(current, "ppml-mem-hold")
    os.makedirs(hold, exist_ok=True)
    _move_pid(hold, os.getpid())
    try:
        with open(os.path.join(current, "cgroup.subtree_control"), "w", encoding="utf-8") as f:
            f.write("+memory\n")
    except OSError:
        _move_pid(current, os.getpid())
        try:
            os.rmdir(hold)
        except OSError:
            pass
        raise
    return True


def _systemd_user_running() -> bool:
    if shutil.which("systemd-run") is None or shutil.which("systemctl") is None:
        return False
    import subprocess

    probe = subprocess.run(
        ["systemctl", "--user", "is-system-running"],
        capture_output=True,
        text=True,
    )
    return (probe.stdout or "").strip() in {"running", "degraded"}


def _reexec_under_systemd(gb: float, log) -> int | None:
    """Re-run this command inside a systemd user scope limited to `gb` GiB.

    Returns the child exit code, or None when systemd cannot take over.
    """
    if os.environ.get("PPML_MEMORY_CGROUP_REEXEC") == "1":
        return None
    if not _systemd_user_running():
        return None
    import subprocess

    log(f"Restarting under a {gb:g} GiB systemd scope.")
    env = os.environ.copy()
    env["PPML_MEMORY_CGROUP_REEXEC"] = "1"
    return subprocess.call(
        [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            "-p", f"MemoryMax={gb:g}G",
            "-p", f"MemoryHigh={gb:g}G",
            "--",
            sys.executable,
            *sys.argv,
        ],
        env=env,
    )


def apply_memory_limit(gb: float | None = None, log_fn=None) -> bool:
    """Keep this process and its children at or below `gb` GiB of host RAM.

    Uses a cgroup v2 memory.max so the cap tracks resident memory. A virtual
    address-space rlimit is not used: CUDA reserves large virtual ranges and
    would fail the run before any model is compiled. Pass gb=0 or set
    PPML_MEMORY_LIMIT_GB=0 to leave the process uncapped.
    """
    log = log_fn or (lambda msg: print(msg, flush=True))
    if gb is None:
        gb = memory_limit_gb_from_env()
    if gb is None or gb <= 0:
        log("RAM cap disabled.")
        return False

    nbytes = int(gb * 1024**3)
    current = _self_cgroup_dir()
    if current is None:
        log("WARNING: RAM cap not applied (cgroup v2 is unavailable).")
        return False

    try:
        current_max = _read_memory_max(current)
    except OSError:
        current_max = None
    if current_max is not None and current_max <= nbytes:
        log(f"RAM cap already {current_max / 1024**3:.1f} GiB ({current}).")
        return True

    try:
        if os.path.basename(current.rstrip("/")) == _MEM_CGROUP_NAME:
            _write_memory_limit(current, nbytes)
            log(f"RAM capped at {gb:g} GiB ({current}).")
            return True

        subtree_path = os.path.join(current, "cgroup.subtree_control")
        subtree = set(open(subtree_path, encoding="utf-8").read().split())
        if "memory" not in subtree and not _enable_memory_controller(current, log):
            raise OSError("memory controller is not delegated to this cgroup")

        child = os.path.join(current, _MEM_CGROUP_NAME)
        os.makedirs(child, exist_ok=True)
        _write_memory_limit(child, nbytes)
        _move_pid(child, os.getpid())
        hold = os.path.join(current, "ppml-mem-hold")
        if os.path.isdir(hold):
            try:
                os.rmdir(hold)
            except OSError:
                pass
    except OSError:
        code = _reexec_under_systemd(gb, log)
        if code is None:
            log(
                "WARNING: RAM cap not applied. Host RAM can still grow until the "
                "OOM killer stops the run. A user systemd session can apply the "
                f"{gb:g} GiB cap, or set PPML_MEMORY_LIMIT_GB=0 to skip it."
            )
            return False
        raise SystemExit(code)

    log(f"RAM capped at {gb:g} GiB ({child}).")
    return True
