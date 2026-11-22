"""Render the measured MT-Bench serving comparison as a terminal-style table."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARMS = (("ar", "target-only"), ("nex-eagle3", "Nex EAGLE3"), ("tuned-eagle31", "EAGLE3.1 final"))
HEADER = (
    "dataset", "method", "conc", "out_tok/s", "vs AR", "vs Nex",
    "accept_len", "accept_%", "TPOT_ms", "TTFT_ms",
)
WIDTH = (11, 16, 4, 9, 7, 7, 10, 8, 8, 8)


def render_rows(report):
    rows = []
    for arm, label in ARMS:
        for concurrency in (1, 4):
            key = f"mtbench-first-turn-n16-o256-c{concurrency}-r1-stream"
            cell = report["cells"][key]
            measured = cell[arm]["summary"]
            rate = measured["throughput_tok_s"]
            assert measured["n"] == 16
            assert measured["mean_ttft_ms"] is not None
            assert measured["mean_tpot_ms"] is not None
            rows.append({
                "dataset": "MT-Bench-16",
                "method": label,
                "conc": concurrency,
                "out_tok/s": rate,
                "vs AR": rate / cell["ar"]["summary"]["throughput_tok_s"],
                "vs Nex": rate / cell["nex-eagle3"]["summary"]["throughput_tok_s"],
                "accept_len": measured["pooled_accept_length"],
                "accept_%": None if measured["strict_draft_accept_rate"] is None else measured["strict_draft_accept_rate"] * 100,
                "TPOT_ms": measured["mean_tpot_ms"],
                "TTFT_ms": measured["mean_ttft_ms"],
            })
    return rows


def format_line(values):
    fields = []
    for i, (value, width) in enumerate(zip(values, WIDTH)):
        if value is None:
            value = "—"
        elif isinstance(value, float):
            precision = 2 if i in (4, 5, 6) else 1
            value = f"{value:.{precision}f}"
            if i in (4, 5):
                value += "x"
        fields.append(f"{value:<{width}}" if i < 2 else f"{value:>{width}}")
    return "  ".join(fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path, help="Output filename without extension")
    args = parser.parse_args()
    rows = render_rows(json.loads(args.report.read_text()))
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans Mono", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(10.1, 3.65), facecolor="#151515")
    left = 0.025
    fig.text(left, 0.915, "Qwen3-30B-A3B-Instruct-2507  /  EAGLE3.1 serving", fontsize=13,
             fontweight="bold", color="#f5f3ef")
    fig.text(left, 0.835, "2 x Ascend 910B3 (64 GiB)   TP2   BF16   SGLang   Greedy SSE",
             fontsize=10, color="#a8b1b7")
    grid_chars = len(format_line(HEADER))
    line = "─" * grid_chars
    for y, string, color in (
        (0.765, line, "#69696c"),
        (0.704, format_line(HEADER), "#d3d9df"),
        (0.646, line, "#69696c"),
    ):
        fig.text(left, y, string, fontsize=10.6, color=color)
    for y, row in zip((0.586, 0.520, 0.446, 0.380, 0.306, 0.240), rows):
        color = "#97e6cf" if row["method"] == "EAGLE3.1 final" else "#eeece5"
        fig.text(left, y, format_line([row[k] for k in HEADER]), fontsize=10.6, color=color)
    fig.text(left, 0.175, line, fontsize=10.6, color="#69696c")
    fig.text(left, 0.112, "16 held-out first turns per cell, max 256 new tokens. TTFT and TPOT are request means.",
             fontsize=8.7, color="#a8b1b7")
    fig.text(left, 0.052, "Ratios compare measured throughput at the same concurrency. Output tokens may differ.",
             fontsize=8.7, color="#a8b1b7")
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=300, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
