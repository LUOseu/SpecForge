"""Read a step-8 Qwen3-30B-A3B draft and its serving export, entirely on CPU.

Example: python check_checkpoint.py --eagle31 CANONICAL_31 SERVING_31 \
           --eagle3 CANONICAL_3 SERVING_3 > checkpoint-parity.json
Each arm may also be checked as soon as its export is available.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

PIN = "c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def serving_name(key):
    for prefix, namespace in [
        ("backbone.layers.", "layers."),
        ("backbone.final_norm.", "norm."),
        ("fusion.", ""),
        ("vocabulary.", ""),
    ]:
        if key.startswith(prefix):
            return namespace + key[len(prefix) :]
    return key


def read_json(folder):
    return json.loads((folder / "config.json").read_text())


def check_arm(canonical, serving, enabled):
    saved, artifact = read_json(canonical), read_json(serving)
    checkpoint = torch.load(
        canonical / "training_state.pt",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    require(
        checkpoint["global_step"] == 8, "expected the completed eighth optimizer update"
    )
    cfg = checkpoint["canonical_draft_config"]
    require(
        cfg == {k: v for k, v in saved.items() if k != "specforge_artifact"},
        "checkpoint and sidecar config differ",
    )
    require(
        saved["specforge_artifact"]["format"] == "specforge"
        and cfg["architectures"] == ["Eagle3DraftModel"],
        "wrong canonical format",
    )
    require(
        artifact["architectures"] == ["LlamaForCausalLMEagle3"]
        and artifact["specforge_artifact"]["sglang_commit"] == PIN,
        "unexpected serving architecture or revision",
    )
    spec = cfg["spec_config"]
    for key in ("hidden_size", "vocab_size", "draft_vocab_size"):
        require(cfg[key] == artifact[key], "serving dimension differs: " + key)
    require(
        (cfg["hidden_size"], cfg["vocab_size"], cfg["draft_vocab_size"])
        == (2048, 151936, 32000),
        "wrong Qwen draft geometry",
    )
    require(
        spec["target_features"]
        == {"hidden_size": 2048, "target_layer_ids": [1, 23, 44]},
        "wrong canonical feature selection",
    )
    require(
        artifact["eagle_config"]["eagle_aux_hidden_state_layer_ids"] == [1, 23, 44]
        and artifact["num_aux_hidden_states"] == 3
        and artifact["target_hidden_size"] == 2048,
        "wrong serving feature selection",
    )
    require(
        all(
            flag is enabled
            for flag in [
                spec["normalize_features"],
                spec["norm_output"],
                artifact["fc_norm"],
                artifact["norm_output"],
            ]
        ),
        "normalization flags do not match the requested arm",
    )
    require(
        artifact["norm_before_fc"] is False
        and artifact["eagle_config"]["norm_before_fc"] is False,
        "concatenated-feature normalization enabled",
    )
    weights = {serving_name(k): v for k, v in checkpoint["draft_state_dict"].items()}
    require(
        len(weights) == len(checkpoint["draft_state_dict"]), "name mapping collision"
    )
    norm_keys = {f"fc_norm.{i}.weight" for i in range(3)} if enabled else set()
    require(
        {k for k in weights if k.startswith("fc_norm.")} == norm_keys,
        "unexpected canonical feature norm keys",
    )
    norm_keys.add("norm.weight")
    require(
        norm_keys | {"fc.weight", "lm_head.weight", "t2d", "d2t"} <= weights.keys(),
        "required weight or mapping missing",
    )
    seen, summaries, map_hashes = set(), {}, {}
    total_bytes = 0
    torch.set_num_threads(1)
    for shard in sorted(serving.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as file:
            for name in file.keys():
                require(
                    name in weights and name not in seen,
                    "unexpected or duplicate serving tensor: " + name,
                )
                source, dest = weights[name], file.get_tensor(name)
                require(
                    source.shape == dest.shape and source.dtype == dest.dtype,
                    "shape or dtype differs: " + name,
                )
                a, b = source.contiguous().view(torch.uint8), dest.contiguous().view(
                    torch.uint8
                )
                require(torch.equal(a, b), "tensor bytes differ: " + name)
                seen.add(name)
                total_bytes += a.numel()
                if name in norm_keys:
                    require(
                        source.shape == (2048,) and torch.isfinite(source).all(),
                        "invalid norm: " + name,
                    )
                    summaries[name] = dict(
                        dtype=str(source.dtype),
                        elements=source.numel(),
                        sha256=hashlib.sha256(a.numpy()).hexdigest(),
                        nonunit_elements=int((source != 1).sum()),
                    )
                if name in ("t2d", "d2t"):
                    map_hashes[name] = hashlib.sha256(a.numpy()).hexdigest()
    require(
        seen == weights.keys(),
        "serving tensors missing: " + str(sorted(weights.keys() - seen)),
    )
    return dict(
        step=checkpoint["global_step"],
        exact_tensors=len(seen),
        exact_bytes=total_bytes,
        fc_norm=enabled,
        norm_output=enabled,
        norm_weights=summaries,
        vocabulary_sha256=map_hashes,
        sglang_commit=PIN,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("eagle31", "eagle3"):
        parser.add_argument(
            "--" + name, nargs=2, type=Path, metavar=("CHECKPOINT", "SERVING")
        )
    args = parser.parse_args()
    reports = {
        name: check_arm(*pair, enabled=(name == "eagle31"))
        for name, pair in vars(args).items()
        if pair
    }
    if not reports:
        parser.error("supply at least one checkpoint and serving directory pair")
    if len(reports) == 2:
        require(
            reports["eagle31"]["vocabulary_sha256"]
            == reports["eagle3"]["vocabulary_sha256"],
            "arms use different vocabulary maps",
        )
    print(json.dumps(dict(passed=True, arms=reports), indent=2))


if __name__ == "__main__":
    main()
