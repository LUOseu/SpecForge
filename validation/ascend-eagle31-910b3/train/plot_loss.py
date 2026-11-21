"""Plot the eight observed train/loss values from each completed smoke run."""

import argparse
import ast
import csv
import math
import re
from pathlib import Path


def read_loss(log):
    series = {}
    for line in log.read_text().splitlines():
        match = re.search(r"\bstep (\d+): (\{.*\})\s*$", line)
        if match:
            row = ast.literal_eval(match[2])
            loss = float(row.get("train/loss", row.get("loss")))
            step = int(match[1])
            if step in series or not math.isfinite(loss):
                raise ValueError(
                    f"{log}: repeated optimizer step or non-finite loss at {step}"
                )
            series[step] = loss
    if sorted(series) != list(range(1, 9)):
        raise ValueError(f"{log}: expected optimizer steps 1–8, found {sorted(series)}")
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_base", type=Path)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["eagle3", "eagle3.1"]

    def observed(method):
        run_id = f"qwen30b-910b3-{method}"
        run = args.run_base / run_id
        if not (run / f"{run_id}-step8" / "training_state.pt").is_file():
            raise FileNotFoundError(f"{run}: missing step-8 checkpoint")
        return read_loss(run / "train.log")

    losses = {method: observed(method) for method in methods}
    with (args.run_base / "train-loss.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["optimizer_step", *methods])
        for step in range(1, 9):
            writer.writerow([step, *(losses[method][step] for method in methods)])
    fig, ax = plt.subplots(figsize=(7.3, 4.2), layout="constrained")
    for method, series in losses.items():
        ax.plot(
            range(1, 9), [series[i] for i in range(1, 9)], "o-", label=method.upper()
        )
    ax.set(
        xlabel="Completed optimizer update",
        ylabel="Training loss (weighted soft cross-entropy)",
        title="Qwen3-30B-A3B-Instruct-2507 · ShareGPT smoke, 8 samples",
        xticks=range(1, 9),
    )
    ax.legend()
    fig.savefig(args.run_base / "train-loss.png", dpi=180)


if __name__ == "__main__":
    main()
