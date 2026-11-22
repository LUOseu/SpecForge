"""Trace real optimizer boundaries and run the configured held-out pass at step zero."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch_npu

from specforge.cli import main
from specforge.eval import Evaluator
from specforge.training.controller import TrainerController, TrainerCore


def append_csv(path, row):
    header = not path.exists()
    with path.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        if header:
            writer.writeheader()
        writer.writerow(row)


def save_weights_digest(module, path):
    parameters = {}
    for name, value in module.named_parameters():
        host = value.detach().cpu().contiguous()
        raw = memoryview(host.view(torch.uint8).numpy())
        parameters[name] = dict(
            shape=list(host.shape), dtype=str(host.dtype), trainable=value.requires_grad,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    path.write_text(json.dumps(parameters, indent=2) + "\n")


def instrument(folder, method, heldout_size):
    train_step = TrainerCore.train_step
    evaluate = TrainerController.evaluate_configured
    fit = TrainerController.fit
    window = []

    def heldout_loss(evaluator, forward_fn, batches):
        # The stock evaluator accumulates FP64 then calls HCCL all_reduce,
        # which rejects kDouble on 910B3. Transfer additive scalar pairs only.
        indices, numerators, denominators = [], [], []
        for batch in batches:
            out = forward_fn(batch)
            n, d = out.loss_terms
            numerators.append(n.detach().float().item())
            denominators.append(d.detach().float().item())
            indices.extend(int(s.rsplit(":", 1)[1]) for s in batch.sample_ids)
        partials = [None] * dist.get_world_size()
        dist.all_gather_object(partials, (indices, numerators, denominators))
        assert all(p[0] == list(range(heldout_size)) for p in partials)
        total = math.fsum(x for _, values, _ in partials for x in values)
        weight = math.fsum(x for _, _, values in partials for x in values)
        assert weight > 0
        return {"eval/avg_loss": total / weight}

    def traced_step(core, batch, ctx=None):
        rank = dist.get_rank()
        optimizer = core.backend.optimizer
        if not window and ctx.global_step == 0 and rank == 0:
            save_weights_digest(core.strategy.trainable_module(), folder / "init_parameters.json")
        window.extend(int(s.rsplit(":", 1)[1]) for s in batch.sample_ids)
        lr_used = optimizer.get_learning_rate()
        result = train_step(core, batch, ctx)
        if result.optimizer_stepped:
            update = ctx.global_step + 1
            assert len(window) == core.accumulation_steps == 4
            adam_step = next(iter(optimizer.optimizer.state.values()))["step"].item()
            assert adam_step == update and lr_used > 0
            metrics = result.materialize_metrics()
            row = dict(method=method, update=update, loss=metrics["loss"],
                       grad_norm=metrics["grad_norm"], lr_used=lr_used,
                       lr_next=optimizer.get_learning_rate(), adam_step=int(adam_step),
                       sample_indices=" ".join(map(str, window)))
            assert math.isfinite(row["loss"]) and 0 < row["grad_norm"] < float("inf")
            if rank == 0:
                row.update({k: metrics[k] for k in sorted(metrics) if k not in row})
                append_csv(folder / "steps.csv", row)
            else:
                append_csv(folder / f"rank{rank}.csv", row)
            window.clear()
        return result

    def traced_evaluate(controller):
        metrics = evaluate(controller)
        if dist.get_rank() == 0:
            loss = metrics["eval/avg_loss"]
            assert math.isfinite(loss)
            append_csv(folder / "validation.csv", dict(
                method=method, update=controller.global_step, loss=loss, n_samples=heldout_size,
            ))
            with (folder / "validation.jsonl").open("a") as fp:
                fp.write(json.dumps(dict(update=controller.global_step, **metrics)) + "\n")
        return metrics

    def traced_fit(controller, data):
        if controller.global_step == 0:
            controller.evaluate_configured()
        return fit(controller, data)

    TrainerCore.train_step = traced_step
    TrainerController.fit = traced_fit
    TrainerController.evaluate_configured = traced_evaluate
    Evaluator.run = heldout_loss


if __name__ == "__main__":
    config = Path(sys.argv[1])
    instrument(config.parent, sys.argv[2], int(sys.argv[3]))
    sys.exit(main(["train", "-c", str(config)]))
