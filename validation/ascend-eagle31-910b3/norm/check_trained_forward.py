"""Inspect two canonical Qwen step-8 forward passes on CPU using a fixed two-token input.

--make-shells creates metadata-only models on a modern SpecForge install. The legacy
CPU can load those with --legacy-import-adapter. All parameter values come from
training_state.pt or the real target embedding table, including the frozen input.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def make_shells(root, dest):
    import yaml

    from specforge.config.draft.codec import decode_config
    from specforge.modeling import build_draft_model

    dest.mkdir(parents=True, exist_ok=True)
    for name in ("eagle3", "eagle3.1"):
        preset = root / "specforge/config/draft/presets" / f"qwen3-30B-A3B-{name}.yaml"
        cfg = decode_config(yaml.safe_load(preset.read_text()))
        model = build_draft_model(
            cfg, backend="eager", device="meta", dtype=torch.bfloat16
        )
        assert all(p.is_meta for p in model.parameters())
        torch.save(model, dest / (name + ".pt"))


def rms(x, scale, eps):
    return (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    ).to(x.dtype) * scale


def inputs(qpos, kpos):
    from specforge.modeling.draft.layers.attention.masks import AttentionLayout
    from specforge.modeling.draft.layers.inputs import LayerInputs

    mask = (qpos[:, :, None] >= kpos[:, None, :]).unsqueeze(1)
    layout = AttentionLayout(query_positions=qpos, key_positions=kpos, visibility=mask)
    return (LayerInputs(attention_layout=layout, position_ids=qpos),)


def difference(x, y):
    return float((x.float() - y.float()).abs().max())


def verify(checkpoint, shell, embed, enabled):
    from specforge.config.draft.codec import encode_config

    model = (
        torch.load(shell, weights_only=False, map_location="cpu")
        .to_empty(device="cpu")
        .eval()
    )
    saved = torch.load(
        str(checkpoint / "training_state.pt"),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    assert saved["global_step"] == 8 and saved[
        "canonical_draft_config"
    ] == encode_config(model.config.draft)
    model.load_state_dict(saved["draft_state_dict"], strict=True)
    model.embed_tokens.load_target_weight(embed)
    assert (
        model.config.draft.spec_config.norm_output is enabled
        and model.config.draft.spec_config.normalize_features is enabled
    )
    eps = model.config.draft.norm_config.eps
    h = model.config.draft.hidden_size
    gen = torch.Generator().manual_seed(20261121)
    x = (
        torch.randn(1, 2, 3 * h, generator=gen)
        * torch.repeat_interleave(torch.tensor([0.125, 3.0, 12.0]), h)
    ).bfloat16()
    tokens = torch.tensor([[42, 777]])
    positions = torch.tensor([[0, 1]])
    state = {
        "fc_input": [],
        "stage_input": [],
        "raw": [],
        "head_input": [],
        "norm_result": [],
    }

    def pre(tag):
        return lambda m, args: state[tag].append(args[0].detach())

    handles = [
        model.fusion.fc.register_forward_pre_hook(pre("fc_input")),
        model.backbone.layers[0].register_forward_pre_hook(pre("stage_input")),
        model.backbone.layers[0].register_forward_hook(
            lambda m, a, o: state["raw"].append(o.hidden.detach())
        ),
        model.lm_head.register_forward_pre_hook(pre("head_input")),
        model.backbone.final_norm.register_forward_hook(
            lambda m, a, o: state["norm_result"].append(o.detach())
        ),
    ]
    with torch.no_grad():
        first = model(tokens, x, inputs(positions, positions))
        second = model(
            tokens.flip(-1),
            None,
            inputs(positions + 2, torch.cat((positions, positions + 2), -1)),
            recurrent_hidden=first.hidden_states,
            kv_prefix=first.kv,
        )
        for hook in handles:
            hook.remove()
        assert {k: len(v) for k, v in state.items()} == {
            "fc_input": 1,
            "stage_input": 2,
            "raw": 2,
            "head_input": 2,
            "norm_result": 2,
        }
        feature_ref = (
            x
            if not enabled
            else torch.cat(
                [
                    rms(ch, norm.weight, eps)
                    for ch, norm in zip(x.chunk(3, dim=-1), model.fusion.fc_norm)
                ],
                -1,
            )
        )
        pre_fc_error = difference(state["fc_input"][0], feature_ref)
        first_input = state["stage_input"][0][..., h:]
        fc_ref = F.linear(feature_ref, model.fusion.fc.weight)
        fc_error = difference(first_input, fc_ref)
        feature_bypass = difference(first_input, F.linear(x, model.fusion.fc.weight))
        assert pre_fc_error == 0 and fc_error == 0 and (feature_bypass > 0) == enabled
        depth = []
        for idx, out in enumerate((first, second)):
            raw = state["raw"][idx]
            ref = rms(raw, model.backbone.final_norm.weight, eps)
            ref_logits = F.linear(ref, model.lm_head.weight)
            expected_feedback = ref if enabled else raw
            checks = dict(
                final_norm_max_abs=difference(state["norm_result"][idx], ref),
                head_input_max_abs=difference(state["head_input"][idx], ref),
                logits_max_abs=difference(out.logits, ref_logits),
                feedback_max_abs=difference(out.hidden_states, expected_feedback),
            )
            assert all(v == 0 for v in checks.values()), checks
            bypass_delta = difference(out.logits, F.linear(raw, model.lm_head.weight))
            assert bypass_delta > 0 and bool(torch.isfinite(out.logits).all())
            depth.append(
                dict(
                    **checks,
                    bypassed_head_norm_logit_delta=bypass_delta,
                    normalized_vs_raw_state_delta=difference(ref, raw),
                )
            )
        recurrence_error = difference(
            state["stage_input"][1][..., h:], first.hidden_states
        )
        assert recurrence_error == 0
    return dict(
        step=8,
        feature_norm=enabled,
        norm_output=enabled,
        fc_reference_error=fc_error,
        fc_input_reference_error=pre_fc_error,
        feature_norm_bypass_projection_delta=feature_bypass,
        final_norm_calls=2,
        fc_calls=1,
        feedback_as_depth2_input_max_abs=recurrence_error,
        depth=depth,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shells", type=Path, required=True)
    p.add_argument("--make-shells", type=Path, metavar="SPECFORGE_SRC")
    p.add_argument("--eagle31", type=Path)
    p.add_argument("--eagle3", type=Path)
    p.add_argument("--target", type=Path)
    p.add_argument("--legacy-import-adapter", action="store_true")
    args = p.parse_args()
    if args.make_shells:
        make_shells(args.make_shells, args.shells)
        return
    if not (args.target and args.eagle31 and args.eagle3):
        p.error("provide target and both trained checkpoints")
    if args.legacy_import_adapter:
        import legacy_import_adapter

        legacy_import_adapter.install()
    torch.set_num_threads(2)
    key = "model.embed_tokens.weight"
    index = json.loads((args.target / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with safe_open(
        args.target / index[key], framework="pt", device="cpu"
    ) as tensor_file:
        embedding = tensor_file.get_tensor(key)
        reports = {
            name: verify(
                path, args.shells / (name + ".pt"), embedding, name == "eagle3.1"
            )
            for name, path in [("eagle3.1", args.eagle31), ("eagle3", args.eagle3)]
        }
    print(
        json.dumps(
            dict(
                passed=True,
                device="cpu",
                dtype="bfloat16",
                torch=torch.__version__,
                real_target_embeddings=True,
                feature_source="seeded synthetic [1,2,6144], scales 0.125/3/12",
                token_ids=[42, 777],
                arms=reports,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
