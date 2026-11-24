"""Turn native pilot stdout and GPU telemetry into reproducible CSVs and plots."""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def parse(root, kind):
    records = {}
    text = (root / f"{kind}-train.log").read_text(errors="replace")
    status = {}
    for line in text.splitlines():
        if kind == "sf" and (match := re.match(r"^step (\d+): (\{.+\})$", line)):
            step, values = int(match[1]), ast.literal_eval(match[2])
            records[step] = {"step": step, "loss": values["loss"], "lr": values.get("lr"),
                             "grad_norm": values.get("grad_norm"),
                             "samples_per_second": values.get("perf/global_samples_per_second")}
        elif line.startswith("PRC_METRIC "):
            values = json.loads(line.removeprefix("PRC_METRIC "))
            step = int(values["step"])
            records[step] = {"step": step, "loss": values["train/loss"], "lr": values["train/lr"],
                             "grad_norm": values["train/grad_norm"], "elapsed": values["elapsed"],
                             "peak_allocated_bytes": values["peak_allocated_bytes"]}
        elif line.startswith("PRC_AM_DONE "):
            status.update(json.loads(line.removeprefix("PRC_AM_DONE ")))
        elif line.startswith("PRC_SF_TRAIN_START "):
            status["started"] = line.split()[1]
        elif line.startswith("PRC_SF_TRAIN_STOP "):
            status["stopped"] = line.split()[1]
            status["exit"] = int(line.rsplit("=", 1)[1])
        elif line.startswith("PRC_AM_PERSISTED "):
            status["checkpoint"] = line.split(maxsplit=1)[1]
    rows = sorted(records.values(), key=lambda r: r["step"])
    assert len(rows) >= 2, kind
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in rows), kind
    if kind == "sf" and {"started", "stopped"} <= status.keys():
        status["elapsed"] = (datetime.fromisoformat(status["stopped"]) -
                            datetime.fromisoformat(status["started"])).total_seconds()
    if kind == "am":
        for first, last in zip(rows, rows[1:]):
            seconds = last["elapsed"] - first["elapsed"]
            if seconds > 0:
                last["samples_per_second"] = (last["step"] - first["step"]) / seconds
    gpu = root / f"{kind}-gpu.csv"
    mib = [int(parts[1]) for line in gpu.read_text().splitlines()
           if len(parts := line.strip().split(",")) == 3 and parts[1].strip().isdigit()]
    status["sampled_peak_gpu_mib"] = max(mib)
    verified = root / f"{kind}-verify.log"
    if verified.exists():
        for line in verified.read_text().splitlines():
            if line.startswith("PRC_VERIFY "):
                status["forward_smoke"] = json.loads(line.removeprefix("PRC_VERIFY "))
    return rows, status


def summarize(rows, status):
    valid = [r for r in rows if r["grad_norm"] > 0]
    assert len(valid) >= 10
    width = min(50, max(5, len(valid)//10))
    status.update(
        last_logged_step=rows[-1]["step"], logged_rows=len(rows), zero_gradient_logs=len(rows)-len(valid),
        window_size=width, opening_loss_median=median(r["loss"] for r in valid[:width]),
        trailing_loss_median=median(r["loss"] for r in valid[-width:]),
        median_samples_per_second=median(r["samples_per_second"] for r in rows if r.get("samples_per_second")),
        maximum_logged_grad_norm=max(r["grad_norm"] for r in rows),
    )
    ema = None
    for r in rows:
        if r["grad_norm"] > 0:
            ema = r["loss"] if ema is None else ema + (r["loss"] - ema) / 16.0
            r["ema"] = ema
        else:
            r["ema"] = None


def render(root, kind, rows, status):
    color = "#2454A6" if kind == "sf" else "#1C7863"
    title = "SpecForge  |  Qwen2.5-0.5B-Instruct" if kind == "sf" else "NeMo AutoModel  |  Qwen3-0.6B"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#BCc8d4", "axes.labelcolor": "#182a3d",
                         "xtick.color": "#374a5e", "ytick.color": "#374a5e", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8.2, 4.5), dpi=300)
    fig.subplots_adjust(left=.10, right=.98, top=.76, bottom=.21)
    fig.text(.10, .94, title, color="#182a3d", fontsize=17, weight="bold")
    fig.text(.10, .864, "Parallel Reconsideration training loss", color="#44576c", fontsize=10)
    ax.grid(axis="y", color="#e3e8ee", lw=.55)
    ax.set_axisbelow(True)
    valid = [r for r in rows if r["ema"] is not None]
    ax.scatter([r["step"] for r in valid], [r["loss"] for r in valid], s=5, color=color,
               linewidths=0, alpha=.23, label="Logged loss", zorder=3)
    ax.plot([r["step"] for r in valid], [r["ema"] for r in valid], lw=1.9, color=color,
            label="EMA over supervised logs (span 31)", zorder=4)
    empty = [r for r in rows if r["ema"] is None]
    if empty:
        ax.scatter([r["step"] for r in empty], [r["loss"] for r in empty], s=15, marker="|",
                   color="#6a7d8c", alpha=.75, label="Zero-gradient log", zorder=3)
    ax.set(xlabel="Optimizer step", ylabel="Training loss")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="upper right", frameon=True, framealpha=.95, fontsize=8, edgecolor="#e7ebf0")
    fig.text(.10, .063, "Single A800 80 GB  ·  BF16 compute  ·  UltraChat, 2,048 rows per epoch",
             fontsize=8.5, color="#52647a")
    fig.savefig(root / f"{kind}-loss.png", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("kind", choices=("sf", "am"))
    args = parser.parse_args()
    rows, status = parse(args.folder, args.kind)
    summarize(rows, status)
    render(args.folder, args.kind, rows, status)
    keys = dict.fromkeys(key for row in rows for key in row).keys()
    with (args.folder / f"{args.kind}-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.folder / f"{args.kind}-summary.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
