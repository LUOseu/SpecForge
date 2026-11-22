"""Reproduce the matched training curves and any recorded held-out evaluations."""

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
WINDOW = 10
TEXT = "#172B3A"
SECONDARY = "#495B68"
METHODS = {
    "EAGLE3": ("eagle3", "#0072B2", "o", "-"),
    "EAGLE3.1": ("eagle31", "#B64E00", "^", (0, (4, 2))),
}


def read_observations(path, method, allowed_steps):
    points = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            update = int(row["update"])
            loss = float(row["loss"])
            if row["method"] != method or update not in allowed_steps or not math.isfinite(loss):
                raise ValueError(f"Invalid observation in {path} at update {update}")
            if update in points:
                raise ValueError(f"Duplicate update {update} in {path}")
            points[update] = (loss, row)
    if not points:
        raise ValueError(f"No observations in {path}")
    return dict(sorted(points.items()))


def read_results(data_dir, updates):
    steps = range(1, updates + 1)
    training = {}
    held_out = {}
    sample_counts = set()
    for name, (directory, *_) in METHODS.items():
        observed = read_observations(data_dir / directory / "steps.csv", name, steps)
        missing = sorted(set(steps) - observed.keys())
        if missing:
            raise ValueError(f"{name} missing optimizer updates {missing}")
        training[name] = [loss for loss, _ in observed.values()]

        path = data_dir / directory / "validation.csv"
        if path.is_file():
            evaluations = read_observations(path, name, range(updates + 1))
            held_out[name] = {step: loss for step, (loss, _) in evaluations.items()}
            sample_counts.update(int(row["n_samples"]) for _, row in evaluations.values())
    if held_out:
        if held_out.keys() != training.keys() or len({tuple(points) for points in held_out.values()}) != 1:
            raise ValueError("Expected matched held-out evaluations at the same optimizer updates")
        if len(sample_counts) != 1 or min(sample_counts) <= 0:
            raise ValueError("Expected the same nonempty held-out split at each evaluation")
    return training, held_out, next(iter(sample_counts), None)


def trailing_means(losses):
    return [fmean(losses[end - WINDOW : end]) for end in range(WINDOW, len(losses) + 1)]


def style_axis(ax, title, subtitle, updates):
    ax.text(0, 1.17, title, transform=ax.transAxes, fontsize=13, color=TEXT, fontweight="bold")
    ax.text(0, 1.075, subtitle, transform=ax.transAxes, fontsize=9, color=SECONDARY)
    ax.set_xlabel("Optimizer update", fontsize=11, color=TEXT, labelpad=7)
    ax.set_xlim(-0.028 * updates, 1.028 * updates)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.margins(y=0.08)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.65, color="#DAE1E5")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#95A3AC")
        ax.spines[spine].set_linewidth(0.7)
    ax.tick_params(length=3, width=0.7, color="#95A3AC", labelcolor=SECONDARY, labelsize=9)


def render(training, held_out, sample_count, output_path, setup, updates):
    if held_out:
        fig, (train_ax, eval_ax) = plt.subplots(
            1, 2, figsize=(10.5, 4.95), dpi=180, sharey=True,
            gridspec_kw={"width_ratios": [1.85, 1.0], "wspace": 0.095},
        )
    else:
        fig, train_ax = plt.subplots(figsize=(8.4, 4.85), dpi=180)
    left = 0.074 if held_out else 0.09
    fig.subplots_adjust(left=left, bottom=0.135, right=0.968, top=0.681)
    fig.text(left, 0.925, "Context-parallel draft training", color=TEXT,
             fontsize=20, fontweight="bold", va="center")
    fig.text(left, 0.849, setup, fontsize=10.5, color=SECONDARY, va="center")

    handles = []
    for method, losses in training.items():
        _, color, marker, line_style = METHODS[method]
        train_ax.scatter(range(1, updates + 1), losses, s=16, marker=marker,
                         color=color, alpha=0.50, linewidths=0, zorder=2)
        means = trailing_means(losses)
        train_ax.plot(range(WINDOW, updates + 1), means, color=color, linestyle=line_style,
                      linewidth=2.25, zorder=3)
        if held_out:
            points = held_out[method]
            eval_ax.scatter(list(points), list(points.values()), s=42, marker=marker,
                            color=color, edgecolors="white", linewidths=0.45, zorder=3)
        handles.append(Line2D([], [], label=method, color=color, marker=marker,
                              linestyle=line_style, linewidth=2.2, markersize=5))
        print(f"{method}: updates 1–{WINDOW} mean {means[0]:.8f}, "
              f"updates {updates - WINDOW + 1}–{updates} mean {means[-1]:.8f}")

    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.968, 0.925),
               frameon=False, ncol=2, fontsize=10, labelcolor=TEXT,
               borderaxespad=0, handlelength=2.5, columnspacing=1.5)
    style_axis(train_ax, "Training loss", "Points show each update. Lines show 10-update trailing means.", updates)
    train_ax.set_ylabel("Loss", fontsize=11, color=TEXT, labelpad=7)
    train_ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    if held_out:
        style_axis(eval_ax, "Held-out loss", f"{sample_count} conversations per evaluation", updates)
        eval_steps = list(next(iter(held_out.values())))
        if len(eval_steps) <= 6:
            eval_ax.set_xticks(eval_steps)
        else:
            eval_ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE / "training-loss.png")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--setup", default="4 × Ascend 910B3   ·   Ulysses 2 × Ring 2   ·   Maximum input length of 1,536 tokens")
    args = parser.parse_args()
    if args.updates < WINDOW:
        parser.error(f"At least {WINDOW} updates are required for the trailing means")
    training, held_out, sample_count = read_results(args.data_dir, args.updates)
    render(training, held_out, sample_count, args.output, args.setup, args.updates)


if __name__ == "__main__":
    main()
