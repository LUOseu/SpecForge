"""Extract every logged optimizer update and per-depth statistic from a multi-rank run."""

import argparse, ast, csv, fcntl, hashlib, io, json, math, re, statistics
from pathlib import Path


def metrics(run):
    source = (run / "train.log").read_bytes()
    collected = {}
    for match in re.finditer(rb"\bstep (\d+): (\{[^{}\n]+\})", source):
        try:
            row = ast.literal_eval(match[2].decode())
        except (ValueError, SyntaxError, UnicodeDecodeError):
            continue
        if "loss" in row:
            collected.setdefault(int(match[1]), []).append(row)
    if not collected:
        return
    rows = []
    for i, rank_rows in sorted(collected.items()):
        # Training loss/accuracy are DP-reduced upstream. Timing is per rank;
        # take the median across all intact copies of the emitted measurement.
        keys = rank_rows[0].keys()
        row = {"optimizer_step": i, "emitted_rank_rows": len(rank_rows)}
        for key in keys:
            vals = [r[key] for r in rank_rows if key in r]
            if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in vals):
                raise ValueError(f"Invalid {key} at {i}")
            if key == "loss" and max(vals) - min(vals) > 1.0e-5:
                raise ValueError(f"Rank loss disagreement at {i}")
            row[key] = statistics.median(vals)
        rows.append(row)
    return source, rows


def plot(output, rows, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["optimizer_step"] for r in rows]
    fig, axes = plt.subplots(
        2, 1, figsize=(9.2, 6.8), sharex=True, layout="constrained"
    )
    axes[0].plot(
        xs, [r["loss"] for r in rows], lw=0.8, color="#86a3bd", label="Training loss"
    )
    smooth = [
        statistics.mean(r["loss"] for r in rows[max(0, j - 15) : j + 1])
        for j in range(len(rows))
    ]
    axes[0].plot(
        xs, smooth, color="#17677c", lw=1.65, label="Mean of latest 16 logged updates"
    )
    axes[0].set_ylabel("Weighted soft cross-entropy")
    axes[0].legend(fontsize=8)
    for depth in range(7):
        ys = [
            statistics.mean(r[f"acc_{depth}"] for r in rows[max(0, j - 15) : j + 1])
            for j in range(len(rows))
        ]
        axes[1].plot(xs, ys, lw=1.2, label=f"Depth {depth + 1}")
    axes[1].set(
        ylabel="Training accuracy (16-point mean)",
        xlabel="Completed optimizer update",
        ylim=(0, 1),
    )
    axes[1].legend(ncol=4, fontsize=8)
    fig.suptitle(title)
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=170)
    temporary.replace(output)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    out = a.run / "metrics"
    out.mkdir(exist_ok=True)
    metrics_lock = (out / ".write.lock").open("a")
    fcntl.flock(metrics_lock, fcntl.LOCK_EX)
    packed = metrics(a.run)
    if packed is None:
        raise SystemExit("No logged optimizer updates")
    source, rows = packed
    cfg = __import__("yaml").safe_load((a.run / "run.yaml").read_text())
    freq = cfg["training"]["log_interval"]
    save_freq = cfg["training"]["save_interval"]
    text = io.StringIO()
    columns = list(rows[0])
    writer = csv.DictWriter(text, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    tmp = out / "optimizer-steps.csv.tmp"
    tmp.write_text(text.getvalue())
    tmp.replace(out / "optimizer-steps.csv")
    steady = [
        r["perf/optimizer_step_time_s"]
        for r in rows
        if r["optimizer_step"] >= max(16, 2 * freq)
        and r["optimizer_step"] % save_freq != freq
    ]
    summary = {
        "run_id": a.run.name,
        "last_logged_step": rows[-1]["optimizer_step"],
        "logged_steps": len(rows),
        "logging_interval": freq,
        "missing_logged_steps": sorted(
            set(range(freq, rows[-1]["optimizer_step"] + 1, freq))
            - {r["optimizer_step"] for r in rows}
        ),
        "first_loss": rows[0]["loss"],
        "last_loss": rows[-1]["loss"],
        "mean_last_16_logged_losses": statistics.mean(r["loss"] for r in rows[-16:]),
        "last_lr": rows[-1]["lr"],
        "median_steady_seconds_per_update": statistics.median(steady)
        if steady
        else None,
        "excluded_from_steady": "initial logged steps below 16 and first logged window after each checkpoint",
        "raw_train_log_bytes_at_snapshot": len(source),
        "raw_train_log_sha256_at_snapshot": hashlib.sha256(source).hexdigest(),
        "per_depth_training_metrics": "acc_0..6, ploss_0..6, kl_loss_0..6, acceptance_rate_0..6; serving acceptance measured separately",
    }
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(out / "summary.json")
    if a.plot:
        plot(out / "loss-and-depth-accuracy.png", rows, a.run.name)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
