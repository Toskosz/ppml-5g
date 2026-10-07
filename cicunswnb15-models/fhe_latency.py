"""Per-record FHE encrypt / inference / decrypt latency measurement."""

from concrete.ml.deployment import FHEModelClient, FHEModelServer
import numpy as np
import os
import sys
import time as _time


def _default_log(msg):
    print(msg)
    sys.stdout.flush()


def measure_fhe_roundtrip(model_dir, X, n_records=100, log_fn=None, return_predictions=False):
    """
    Time encrypt → FHE inference → decrypt for up to n_records rows of X.

    Returns a dict with per-phase totals, means, stds, and the raw timing lists.
    If return_predictions=True, also includes a binary `predictions` list.
    """
    log = log_fn or _default_log
    n_records = min(int(n_records), int(X.shape[0]))
    if n_records <= 0:
        raise ValueError("n_records must be >= 1 and X must be non-empty")

    log(f"Loading FHE client/server from '{model_dir}' for latency measurement...")
    device_marker = os.path.join(model_dir, "fhe_device.txt")
    if os.path.isfile(device_marker):
        with open(device_marker, "r", encoding="utf-8") as f:
            compiled_device = f.read().strip() or "unknown"
        log(f"Model compiled for device='{compiled_device}' (from fhe_device.txt).")
        if compiled_device == "cuda":
            try:
                import concrete.compiler as cc
                if not cc.check_gpu_enabled():
                    log(
                        "WARNING: model was compiled for CUDA but Concrete GPU runtime "
                        "is disabled. Install the GPU wheel from https://pypi.zama.ai/gpu"
                    )
            except Exception as exc:
                log(f"WARNING: could not verify GPU runtime: {exc}")

    server = FHEModelServer(model_dir)
    server.load()
    client = FHEModelClient(model_dir)

    log("Generating evaluation keys...")
    t0 = _time.time()
    evaluation_keys = client.get_serialized_evaluation_keys()
    keygen_dur = _time.time() - t0
    log(f"Evaluation keys ready in {keygen_dur:,.3f}s ({len(evaluation_keys):,} bytes).")

    encrypt_times = []
    inference_times = []
    decrypt_times = []
    predictions = []
    report_interval = max(1, n_records // 10)

    log(f"Measuring encrypt / inference / decrypt on {n_records:,} records...")
    for i in range(n_records):
        single_record = X[i:i + 1]

        t0 = _time.time()
        encrypted_input = client.quantize_encrypt_serialize(single_record)
        encrypt_times.append(_time.time() - t0)

        t0 = _time.time()
        encrypted_output = server.run(encrypted_input, evaluation_keys)
        inference_times.append(_time.time() - t0)

        t0 = _time.time()
        decrypted = client.deserialize_decrypt_dequantize(encrypted_output)
        decrypt_times.append(_time.time() - t0)

        if return_predictions:
            predictions.append(1 if decrypted[0][1] > 0.5 else 0)

        if (i + 1) % report_interval == 0:
            log(
                f"  {i + 1}/{n_records} — "
                f"avg encrypt={np.mean(encrypt_times):.6f}s, "
                f"avg infer={np.mean(inference_times):.6f}s, "
                f"avg decrypt={np.mean(decrypt_times):.6f}s"
            )

    result = {
        "n_records": n_records,
        "keygen_s": keygen_dur,
        "encrypt_times": encrypt_times,
        "inference_times": inference_times,
        "decrypt_times": decrypt_times,
        "total_encrypt_s": float(np.sum(encrypt_times)),
        "total_inference_s": float(np.sum(inference_times)),
        "total_decrypt_s": float(np.sum(decrypt_times)),
        "mean_encrypt_s": float(np.mean(encrypt_times)),
        "mean_inference_s": float(np.mean(inference_times)),
        "mean_decrypt_s": float(np.mean(decrypt_times)),
        "std_encrypt_s": float(np.std(encrypt_times)),
        "std_inference_s": float(np.std(inference_times)),
        "std_decrypt_s": float(np.std(decrypt_times)),
    }
    result["mean_e2e_s"] = (
        result["mean_encrypt_s"] + result["mean_inference_s"] + result["mean_decrypt_s"]
    )
    result["total_e2e_s"] = (
        result["total_encrypt_s"] + result["total_inference_s"] + result["total_decrypt_s"]
    )
    if return_predictions:
        result["predictions"] = predictions
    return result


def print_latency_summary(result, title="FHE LATENCY SUMMARY"):
    """Pretty-print a measure_fhe_roundtrip result."""
    n = result["n_records"]
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)
    print(f"Records measured: {n}")
    print(f"Key generation:   {result['keygen_s']:.4f} s (once)")
    print(
        f"{'Phase':>12s} | {'Total (s)':>12s} | {'Mean (s)':>12s} | {'Std (s)':>12s}"
    )
    print("-" * 58)
    for phase, total_k, mean_k, std_k in [
        ("encrypt", "total_encrypt_s", "mean_encrypt_s", "std_encrypt_s"),
        ("inference", "total_inference_s", "mean_inference_s", "std_inference_s"),
        ("decrypt", "total_decrypt_s", "mean_decrypt_s", "std_decrypt_s"),
        ("end-to-end", "total_e2e_s", "mean_e2e_s", None),
    ]:
        total = result[total_k]
        mean = result[mean_k]
        if std_k is None:
            print(f"{phase:>12s} | {total:>12.4f} | {mean:>12.6f} | {'n/a':>12s}")
        else:
            print(f"{phase:>12s} | {total:>12.4f} | {mean:>12.6f} | {result[std_k]:>12.6f}")
    print("=" * 70 + "\n")
    sys.stdout.flush()
