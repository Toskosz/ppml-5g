#!/usr/bin/env python3
"""
Run all cicids2018-kdd-models experiment scripts and write report.txt.

Usage (from this directory):
    python run_all_and_report.py
    python run_all_and_report.py --skip-fast          # only svd100/200 + inference
    python run_all_and_report.py --only-inference     # best_model_single_record.py only
    python run_all_and_report.py --report-only DIR    # rebuild report from existing logs

Outputs:
    report.txt
    run_logs/<timestamp>/<script>.log   (full stdout+stderr per script)
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

# Runnable experiment scripts (fhe_latency.py is a helper module, not listed).
ALL_SCRIPTS = [
    "svd100_model.py",
    "svd200_model.py",
    "svd100_model_fast.py",
    "svd200_model_fast.py",
    "best_model_single_record.py",
]

DEFAULT_SCRIPTS = ALL_SCRIPTS


def now_brasilia():
    return datetime.datetime.now(ZoneInfo("America/Sao_Paulo"))


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{seconds:.1f}s"


def run_script(script: str, log_path: Path, python_exe: str) -> dict:
    """Run one script, tee output to console and log_path."""
    print("\n" + "#" * 70)
    print(f"#  RUNNING {script}")
    print("#" * 70)
    sys.stdout.flush()

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"# script: {script}\n")
        log_f.write(f"# started: {now_brasilia().isoformat()}\n")
        log_f.write("#" * 70 + "\n")
        log_f.flush()

        proc = subprocess.Popen(
            [python_exe, "-u", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(Path(__file__).resolve().parent),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
        exit_code = proc.wait()
        elapsed = time.time() - t0
        log_f.write("\n" + "#" * 70 + "\n")
        log_f.write(f"# finished: {now_brasilia().isoformat()}\n")
        log_f.write(f"# exit_code: {exit_code}\n")
        log_f.write(f"# wall_time_s: {elapsed:.3f}\n")

    status = "OK" if exit_code == 0 else "FAILED"
    print(f"\n[{status}] {script} finished in {format_duration(elapsed)} (exit={exit_code})")
    sys.stdout.flush()
    return {
        "script": script,
        "log_path": str(log_path),
        "exit_code": exit_code,
        "elapsed_s": elapsed,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

CONFIG_RE = re.compile(
    r"CONFIG[^:]*:\s*n_estimators=(?P<n>\d+),\s*max_depth=(?P<d>\d+),\s*svd=(?P<svd>\d+)",
    re.IGNORECASE,
)
CONFIG_TAG_RE = re.compile(r"n=(?P<n>\d+),\s*d=(?P<d>\d+),\s*svd=(?P<svd>\d+)")
METRICS_RE = re.compile(
    r"Accuracy:\s*(?P<acc>[\d.]+).*?"
    r"Precision:\s*(?P<prec>[\d.]+).*?"
    r"Recall:\s*(?P<rec>[\d.]+).*?"
    r"F1-Score:\s*(?P<f1>[\d.]+)",
    re.DOTALL,
)
LATENCY_BLOCK_RE = re.compile(
    r"=+\s*FHE LATENCY\s*\((?P<tag>[^)]+)\)\s*=+\s*"
    r"Records measured:\s*(?P<nrec>\d+)\s*"
    r"Key generation:\s*(?P<keygen>[\d.]+)\s*s.*?"
    r"encrypt\s*\|\s*(?P<enc_tot>[\d.]+)\s*\|\s*(?P<enc_mean>[\d.]+)\s*\|\s*(?P<enc_std>[\d.]+)\s*"
    r"inference\s*\|\s*(?P<inf_tot>[\d.]+)\s*\|\s*(?P<inf_mean>[\d.]+)\s*\|\s*(?P<inf_std>[\d.]+)\s*"
    r"decrypt\s*\|\s*(?P<dec_tot>[\d.]+)\s*\|\s*(?P<dec_mean>[\d.]+)\s*\|\s*(?P<dec_std>[\d.]+)\s*"
    r"end-to-end\s*\|\s*(?P<e2e_tot>[\d.]+)\s*\|\s*(?P<e2e_mean>[\d.]+)",
    re.DOTALL | re.IGNORECASE,
)
DEVICE_RE = re.compile(
    r"FHE compilation completed in\s*[\d,.]+s on device='(?P<dev>cpu|cuda)'"
    r"|FHE assets saved \(device=(?P<dev2>cpu|cuda)\)"
    r"|FHE device:\s*(?P<dev3>cpu|cuda)",
    re.IGNORECASE,
)
TRAIN_DONE_RE = re.compile(
    r"\[(?P<tag>n=\d+,\s*d=\d+,\s*svd=\d+)\]\s*Training completed in\s*(?P<t>[\d,.]+)s"
)
PRED_DONE_RE = re.compile(
    r"\[(?P<tag>n=\d+,\s*d=\d+,\s*svd=\d+)\]\s*Clear prediction completed in\s*(?P<t>[\d,.]+)s"
)
COMPILE_DONE_RE = re.compile(
    r"\[(?P<tag>n=\d+,\s*d=\d+,\s*svd=\d+)\]\s*FHE compilation completed in\s*(?P<t>[\d,.]+)s"
)
FHE_SIM_DONE_RE = re.compile(
    r"\[(?P<tag>n=\d+,\s*d=\d+,\s*svd=\d+)\]\s*FHE simulation completed in\s*(?P<t>[\d,.]+)s"
)
SUMMARY_ROW_RE = re.compile(
    r"(?P<tag>n=\d+,\s*d=\d+,\s*svd=\d+)\s*\|\s*"
    r"(?P<train>[\d.]+)\s*\|\s*"
    r"(?P<pred>[\d.]+)\s*\|\s*"
    r"(?P<compile>[\d.]+)\s*\|\s*"
    r"(?P<fhesim>[\d.]+)"
    r"(?:\s*\|\s*(?P<enc>[\d.]+)\s*\|\s*(?P<inf>[\d.]+)\s*\|\s*(?P<dec>[\d.]+)"
    r"(?:\s*\|\s*(?P<e2e>[\d.]+))?)?"
)
PLAINTEXT_LAT_RE = re.compile(
    r"=+\s*PLAINTEXT LATENCY SUMMARY\s*=+\s*"
    r"Records measured:\s*(?P<nrec>\d+)\s*"
    r"Total inference time:\s*(?P<tot>[\d.]+)\s*s\s*"
    r"Mean inference time/record:\s*(?P<mean>[\d.]+)\s*s\s*"
    r"Std inference time/record:\s*(?P<std>[\d.]+)\s*s",
    re.DOTALL | re.IGNORECASE,
)
BENCH_FHE_ROW_RE = re.compile(
    r"(?P<tag>n=\d+,\s*d=\d+,\s*svd=\d+)\s*\|\s*"
    r"(?P<enc>[\d.]+)\s*\|\s*(?P<inf>[\d.]+)\s*\|\s*(?P<dec>[\d.]+)\s*\|\s*(?P<e2e>[\d.]+)"
)


def _parse_float(s: str) -> float:
    return float(s.replace(",", ""))


def _metric_kind(text_before: str) -> str:
    window = text_before[-400:].lower()
    if "fhe metrics" in window:
        return "fhe"
    if "plaintext metrics" in window or "clear (plaintext)" in window:
        return "plaintext"
    if "fhe" in window:
        return "fhe"
    return "plaintext"


def parse_log(text: str, script: str) -> dict:
    """Extract structured metrics from a script log."""
    result = {
        "script": script,
        "model_metrics": [],  # {config, kind, accuracy, precision, recall, f1}
        "phase_times": {},    # tag -> {train, pred, compile, fhe_sim}
        "summary_rows": [],   # from GLOBAL/FHE SUMMARY tables
        "fhe_latency": [],    # from FHE LATENCY blocks
        "plaintext_latency": [],
        "bench_fhe_rows": [],
        "devices": [],        # device strings observed in the log
    }

    for m in DEVICE_RE.finditer(text):
        dev = m.group("dev") or m.group("dev2") or m.group("dev3")
        if dev:
            result["devices"].append(dev.lower())

    # Phase timings from log lines
    phase = result["phase_times"]
    for regex, key in [
        (TRAIN_DONE_RE, "train_s"),
        (PRED_DONE_RE, "predict_s"),
        (COMPILE_DONE_RE, "compile_s"),
        (FHE_SIM_DONE_RE, "fhe_sim_s"),
    ]:
        for m in regex.finditer(text):
            tag = re.sub(r"\s+", " ", m.group("tag").strip())
            phase.setdefault(tag, {})[key] = _parse_float(m.group("t"))

    # Classification metrics — associate with nearest preceding config
    for m in METRICS_RE.finditer(text):
        start = m.start()
        before = text[max(0, start - 800):start]
        kind = _metric_kind(before)
        config = None
        cfg_m = list(CONFIG_RE.finditer(before))
        tag_m = list(CONFIG_TAG_RE.finditer(before))
        if cfg_m:
            last = cfg_m[-1]
            config = f"n={last.group('n')}, d={last.group('d')}, svd={last.group('svd')}"
        elif tag_m:
            last = tag_m[-1]
            config = f"n={last.group('n')}, d={last.group('d')}, svd={last.group('svd')}"
        result["model_metrics"].append({
            "config": config or "unknown",
            "kind": kind,
            "accuracy": float(m.group("acc")),
            "precision": float(m.group("prec")),
            "recall": float(m.group("rec")),
            "f1": float(m.group("f1")),
        })

    for m in LATENCY_BLOCK_RE.finditer(text):
        tag = re.sub(r"\s+", " ", m.group("tag").strip())
        result["fhe_latency"].append({
            "config": tag,
            "n_records": int(m.group("nrec")),
            "keygen_s": float(m.group("keygen")),
            "encrypt_total_s": float(m.group("enc_tot")),
            "encrypt_mean_s": float(m.group("enc_mean")),
            "encrypt_std_s": float(m.group("enc_std")),
            "inference_total_s": float(m.group("inf_tot")),
            "inference_mean_s": float(m.group("inf_mean")),
            "inference_std_s": float(m.group("inf_std")),
            "decrypt_total_s": float(m.group("dec_tot")),
            "decrypt_mean_s": float(m.group("dec_mean")),
            "decrypt_std_s": float(m.group("dec_std")),
            "e2e_total_s": float(m.group("e2e_tot")),
            "e2e_mean_s": float(m.group("e2e_mean")),
        })

    for m in PLAINTEXT_LAT_RE.finditer(text):
        # Find nearest config tag before this block
        before = text[max(0, m.start() - 600):m.start()]
        tags = list(CONFIG_TAG_RE.finditer(before))
        config = "unknown"
        if tags:
            last = tags[-1]
            config = f"n={last.group('n')}, d={last.group('d')}, svd={last.group('svd')}"
        result["plaintext_latency"].append({
            "config": config,
            "n_records": int(m.group("nrec")),
            "total_s": float(m.group("tot")),
            "mean_s": float(m.group("mean")),
            "std_s": float(m.group("std")),
        })

    for m in SUMMARY_ROW_RE.finditer(text):
        row = {
            "config": re.sub(r"\s+", " ", m.group("tag").strip()),
            "train_s": float(m.group("train")),
            "predict_s": float(m.group("pred")),
            "compile_s": float(m.group("compile")),
            "fhe_sim_s": float(m.group("fhesim")),
        }
        if m.group("enc") is not None:
            row["encrypt_mean_s"] = float(m.group("enc"))
            row["inference_mean_s"] = float(m.group("inf"))
            row["decrypt_mean_s"] = float(m.group("dec"))
            if m.group("e2e") is not None:
                row["e2e_mean_s"] = float(m.group("e2e"))
        result["summary_rows"].append(row)

    # Deduplicate summary rows (GLOBAL + per-SVD tables repeat)
    seen = set()
    deduped = []
    for row in result["summary_rows"]:
        key = (
            row["config"],
            row.get("train_s"),
            row.get("compile_s"),
            row.get("encrypt_mean_s"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    result["summary_rows"] = deduped

    # Benchmark end summary rows (encrypt/infer/decrypt/e2e only)
    if "FHE LATENCY SUMMARY (per-record means)" in text:
        section = text.split("FHE LATENCY SUMMARY (per-record means)", 1)[1]
        for m in BENCH_FHE_ROW_RE.finditer(section):
            result["bench_fhe_rows"].append({
                "config": re.sub(r"\s+", " ", m.group("tag").strip()),
                "encrypt_mean_s": float(m.group("enc")),
                "inference_mean_s": float(m.group("inf")),
                "decrypt_mean_s": float(m.group("dec")),
                "e2e_mean_s": float(m.group("e2e")),
            })

    return result


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _section(title: str) -> str:
    bar = "=" * 78
    return f"\n{bar}\n{title}\n{bar}\n"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(none)\n"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "-+-".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines) + "\n"


def build_report(run_meta: dict, run_results: list[dict], parsed: list[dict]) -> str:
    lines = []
    lines.append("PPML-5G / CIC-IDS-2018 — Experiment Report")
    lines.append(f"Generated: {run_meta['finished']}")
    lines.append(f"Started:   {run_meta['started']}")
    lines.append(f"Host cwd:  {run_meta['cwd']}")
    lines.append(f"Python:    {run_meta['python']}")
    lines.append(f"FHE device request: {run_meta.get('fhe_device', 'auto')}")
    lines.append(f"Total wall time: {format_duration(run_meta['total_elapsed_s'])}")
    lines.append("")
    lines.append(
        "Note: svd*_model.py and svd*_model_fast.py write the same artifact paths; "
        "later scripts overwrite earlier FHE/plaintext artifacts. Metrics below are "
        "captured per script run before overwrite."
    )

    lines.append(_section("1. RUN INVENTORY"))
    inv_rows = []
    for r, p in zip(run_results, parsed):
        devices = sorted(set(p.get("devices") or []))
        inv_rows.append([
            r["script"],
            r["status"],
            str(r["exit_code"]),
            format_duration(r["elapsed_s"]),
            ",".join(devices) if devices else "-",
            r["log_path"],
        ])
    lines.append(_table(
        ["Script", "Status", "Exit", "Wall time", "FHE device(s)", "Log file"],
        inv_rows,
    ))

    lines.append(_section("2. MODEL QUALITY METRICS (accuracy / precision / recall / F1)"))
    metric_rows = []
    for p in parsed:
        for m in p["model_metrics"]:
            metric_rows.append([
                p["script"],
                m["config"],
                m["kind"],
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
            ])
    lines.append(_table(
        ["Script", "Config", "Kind", "Accuracy", "Precision", "Recall", "F1"],
        metric_rows,
    ))

    lines.append(_section("3. TRAINING / COMPILE / FHE-SIM WALL TIMES"))
    time_rows = []
    for p in parsed:
        # Prefer summary table rows when present
        if p["summary_rows"]:
            for row in p["summary_rows"]:
                time_rows.append([
                    p["script"],
                    row["config"],
                    f"{row['train_s']:.1f}",
                    f"{row['predict_s']:.1f}",
                    f"{row['compile_s']:.1f}",
                    f"{row['fhe_sim_s']:.1f}",
                ])
        else:
            for tag, vals in sorted(p["phase_times"].items()):
                time_rows.append([
                    p["script"],
                    tag,
                    f"{vals.get('train_s', float('nan')):.1f}" if "train_s" in vals else "-",
                    f"{vals.get('predict_s', float('nan')):.1f}" if "predict_s" in vals else "-",
                    f"{vals.get('compile_s', float('nan')):.1f}" if "compile_s" in vals else "-",
                    f"{vals.get('fhe_sim_s', float('nan')):.1f}" if "fhe_sim_s" in vals else "-",
                ])
    lines.append(_table(
        ["Script", "Config", "Train (s)", "Predict (s)", "Compile (s)", "FHE sim (s)"],
        time_rows,
    ))

    lines.append(_section("4. FHE ENCRYPT / INFERENCE / DECRYPT LATENCY (real client/server)"))
    lat_rows = []
    for p in parsed:
        for lat in p["fhe_latency"]:
            lat_rows.append([
                p["script"],
                lat["config"],
                str(lat["n_records"]),
                f"{lat['keygen_s']:.4f}",
                f"{lat['encrypt_mean_s']:.6f}",
                f"{lat['inference_mean_s']:.6f}",
                f"{lat['decrypt_mean_s']:.6f}",
                f"{lat['e2e_mean_s']:.6f}",
                f"{lat['encrypt_std_s']:.6f}",
                f"{lat['inference_std_s']:.6f}",
                f"{lat['decrypt_std_s']:.6f}",
            ])
        # Fallback: bench summary table if blocks missing
        if not p["fhe_latency"] and p["bench_fhe_rows"]:
            for lat in p["bench_fhe_rows"]:
                lat_rows.append([
                    p["script"],
                    lat["config"],
                    "-",
                    "-",
                    f"{lat['encrypt_mean_s']:.6f}",
                    f"{lat['inference_mean_s']:.6f}",
                    f"{lat['decrypt_mean_s']:.6f}",
                    f"{lat['e2e_mean_s']:.6f}",
                    "-",
                    "-",
                    "-",
                ])
        # Also pull Enc/Inf/Dec from GLOBAL SUMMARY rows when present
        for row in p["summary_rows"]:
            if "encrypt_mean_s" not in row:
                continue
            # Skip if already covered by a latency block for same config+script
            already = any(
                r[0] == p["script"] and r[1] == row["config"]
                for r in lat_rows
            )
            if already:
                continue
            lat_rows.append([
                p["script"],
                row["config"],
                "-",
                "-",
                f"{row['encrypt_mean_s']:.6f}",
                f"{row['inference_mean_s']:.6f}",
                f"{row['decrypt_mean_s']:.6f}",
                f"{row.get('e2e_mean_s', row['encrypt_mean_s'] + row['inference_mean_s'] + row['decrypt_mean_s']):.6f}",
                "-",
                "-",
                "-",
            ])
    lines.append(_table(
        [
            "Script", "Config", "N", "Keygen (s)",
            "Enc mean", "Inf mean", "Dec mean", "E2E mean",
            "Enc std", "Inf std", "Dec std",
        ],
        lat_rows,
    ))

    lines.append(_section("5. PLAINTEXT SINGLE-RECORD INFERENCE LATENCY"))
    pt_rows = []
    for p in parsed:
        for lat in p["plaintext_latency"]:
            pt_rows.append([
                p["script"],
                lat["config"],
                str(lat["n_records"]),
                f"{lat['total_s']:.4f}",
                f"{lat['mean_s']:.6f}",
                f"{lat['std_s']:.6f}",
            ])
    lines.append(_table(
        ["Script", "Config", "N", "Total (s)", "Mean (s)", "Std (s)"],
        pt_rows,
    ))

    lines.append(_section("6. PER-SCRIPT LOG POINTERS"))
    for r in run_results:
        lines.append(f"- {r['script']}: {r['log_path']}  [{r['status']}, {format_duration(r['elapsed_s'])}]")
    lines.append("")
    lines.append(
        "Full raw stdout/stderr for each script is preserved in the log files above. "
        "This report aggregates classification metrics and timing measurements."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(report_path: Path, run_meta: dict, run_results: list[dict]) -> Path:
    parsed = []
    for r in run_results:
        text = Path(r["log_path"]).read_text(encoding="utf-8", errors="replace")
        parsed.append(parse_log(text, r["script"]))

    report = build_report(run_meta, run_results, parsed)
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip-fast",
        action="store_true",
        help="Skip svd100_model_fast.py and svd200_model_fast.py",
    )
    parser.add_argument(
        "--only-fast",
        action="store_true",
        help="Run only the *_fast trainers + best_model_single_record.py",
    )
    parser.add_argument(
        "--only-inference",
        action="store_true",
        help="Run only best_model_single_record.py (expects existing artifacts)",
    )
    parser.add_argument(
        "--scripts",
        nargs="+",
        help="Explicit script list to run (overrides --skip-fast/--only-*)",
    )
    parser.add_argument(
        "--report-only",
        metavar="LOG_DIR",
        help="Do not run scripts; rebuild report.txt from logs in LOG_DIR",
    )
    parser.add_argument(
        "--report",
        default="report.txt",
        help="Output report path (default: report.txt)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for child scripts",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="FHE compile/runtime device (sets PPML_FHE_DEVICE for child scripts). "
             "Default: keep existing env or 'auto'.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    os.chdir(root)

    if args.device is not None:
        os.environ["PPML_FHE_DEVICE"] = args.device
        print(f"PPML_FHE_DEVICE={args.device}")

    if args.report_only:
        log_dir = Path(args.report_only).resolve()
        if not log_dir.is_dir():
            print(f"ERROR: log dir not found: {log_dir}", file=sys.stderr)
            return 1
        run_results = []
        for script in ALL_SCRIPTS:
            log_path = log_dir / f"{Path(script).stem}.log"
            if not log_path.exists():
                continue
            text = log_path.read_text(encoding="utf-8", errors="replace")
            exit_m = re.search(r"# exit_code:\s*(-?\d+)", text)
            time_m = re.search(r"# wall_time_s:\s*([\d.]+)", text)
            exit_code = int(exit_m.group(1)) if exit_m else -1
            elapsed = float(time_m.group(1)) if time_m else 0.0
            run_results.append({
                "script": script,
                "log_path": str(log_path),
                "exit_code": exit_code,
                "elapsed_s": elapsed,
                "status": "OK" if exit_code == 0 else "FAILED",
            })
        if not run_results:
            print(f"ERROR: no script logs found in {log_dir}", file=sys.stderr)
            return 1
        run_meta = {
            "started": "n/a (report-only)",
            "finished": now_brasilia().isoformat(),
            "cwd": str(root),
            "python": args.python,
            "fhe_device": os.environ.get("PPML_FHE_DEVICE", "auto"),
            "total_elapsed_s": sum(r["elapsed_s"] for r in run_results),
        }
        report_path = write_report(root / args.report, run_meta, run_results)
        print(f"Wrote {report_path}")
        return 0

    if args.scripts:
        scripts = args.scripts
    elif args.only_inference:
        scripts = ["best_model_single_record.py"]
    elif args.only_fast:
        scripts = [
            "svd100_model_fast.py",
            "svd200_model_fast.py",
            "best_model_single_record.py",
        ]
    elif args.skip_fast:
        scripts = [
            "svd100_model.py",
            "svd200_model.py",
            "best_model_single_record.py",
        ]
    else:
        scripts = list(DEFAULT_SCRIPTS)

    for script in scripts:
        if not (root / script).is_file():
            print(f"ERROR: script not found: {script}", file=sys.stderr)
            return 1

    stamp = now_brasilia().strftime("%Y%m%d_%H%M%S")
    log_dir = root / "run_logs" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    started = now_brasilia().isoformat()
    t0 = time.time()
    print(f"Run started: {started}")
    print(f"Logs directory: {log_dir}")
    print(f"Scripts: {', '.join(scripts)}")
    sys.stdout.flush()

    run_results = []
    for script in scripts:
        log_path = log_dir / f"{Path(script).stem}.log"
        result = run_script(script, log_path, args.python)
        run_results.append(result)
        # Continue on failure so partial metrics still land in the report,
        # but record the failure clearly.
        if result["exit_code"] != 0:
            print(f"WARNING: {script} failed; continuing with remaining scripts.")

    finished = now_brasilia().isoformat()
    total_elapsed = time.time() - t0
    run_meta = {
        "started": started,
        "finished": finished,
        "cwd": str(root),
        "python": args.python,
        "fhe_device": os.environ.get("PPML_FHE_DEVICE", "auto"),
        "total_elapsed_s": total_elapsed,
    }

    report_path = write_report(root / args.report, run_meta, run_results)
    print("\n" + "#" * 70)
    print(f"#  REPORT WRITTEN: {report_path}")
    print(f"#  LOGS: {log_dir}")
    print(f"#  TOTAL WALL TIME: {format_duration(total_elapsed)}")
    print("#" * 70)

    # Non-zero if any child failed
    return 0 if all(r["exit_code"] == 0 for r in run_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
